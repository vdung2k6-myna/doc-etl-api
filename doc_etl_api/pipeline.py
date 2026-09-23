import hashlib
import logging
import re
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import BinaryIO

import requests
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import EasyOcrOptions, ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core import Document as LlamaDocument
from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from doc_etl_api.config import DEFAULT_USER_AGENT, Settings, VectorStoreBackend, settings
from doc_etl_api.extraction import select_main_content

logger = logging.getLogger(__name__)

# The metadata key carrying the collections a source belongs to.
COLLECTIONS_KEY = "collections"

# The metadata key carrying the name the index stores a source under, which is
# what its content is fetched by. It is written where the source's identity is
# decided rather than derived at read time from `source_name` and `final_url`:
# those two agree with it by accident of which ingestion path ran, and an address
# that silently became a different string would fail as a lookup on the content
# endpoint rather than anywhere near the cause.
ADDRESS_KEY = "address"

# Metadata that is written for filtering and reporting but never read by a model.
# Both the embed and the LLM rendering are excluded, and that is not
# belt-and-braces: metadata is charged against the chunk-size budget before
# splitting, and `MetadataAwareTextSplitter._get_metadata_str` renders a node both
# ways and keeps the longer of the two, so excluding one mode alone would still
# let a collection name shrink the text each chunk can hold. That changes where a
# source splits and therefore its vectors -- measured, tagging one document with
# two collections took it from 14 nodes to 20. Excluding the key from both
# readings is what makes tagging metadata-only: the prefilter still sees the
# collections, no model ever does. The address is excluded for the same reason,
# though it buys less: the address is the same string as `source_name` for a file
# and as `final_url` for a URL, and neither of those is excluded, so the value is
# already charged against every chunk's budget and already rendered into what a
# model would read. What the exclusion removes is the `address: <value>` line
# itself, which would otherwise be a second copy of a filename or a full URL plus
# its label, paid for by every chunk of that source. Measured on the fixture
# below, the extra line was worth 5 tokens and did not move a boundary at
# CHUNK_SIZE=128 -- but the collections precedent shows that a metadata line can
# move one, so the key is kept out rather than trusted to be small. Every key in
# this list is metadata; the rest of the metadata dict is not, and that
# inconsistency is worth revisiting on its own rather than folded into this change.
_MODEL_EXCLUDED_KEYS = [COLLECTIONS_KEY, ADDRESS_KEY]


@dataclass(frozen=True)
class SourceRecord:
    """What a source is, as the catalog reports it.

    ``name`` and ``address`` are two different things, and for a URL that
    redirected they differ. The name is what the source was submitted as -- the
    filename an upload carried, the URL a caller sent -- and it is what the
    catalog displays. The address is what the index stores the source under and
    therefore the only thing that finds one again: a filename, or the final URL a
    submitted URL landed on. Both are reported so a caller can display a source
    and fetch it without one of those needing to be derived from the other.
    """

    name: str
    address: str
    source_type: str
    collections: tuple[str, ...]
    chunk_count: int


@dataclass(frozen=True)
class _Adjacency:
    """Where a hit sits inside its source, and what context surrounds it.

    ``before`` and ``after`` describe the context the source holds, whether or
    not it was asked for; ``neighbours`` carries the part of it that was.
    """

    position: int
    before: int
    after: int
    neighbours: list[dict]


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _node_id(source_key: str, position: int) -> str:
    """A node's identity: its source's identity and its position within it.

    Scoped to the source rather than derived from the node's own text. Text is
    shared between documents -- a boilerplate paragraph appears in two of them --
    and a text-addressed id would be a single entry that both sources claim.
    Deleting one source on re-ingest would then delete the other source's copy
    of that text along with it: measured, a source that still carried the
    paragraph lost it when the other source was re-ingested without it. Scoping
    the id to the source keeps one source's edit from reaching another's content.
    """
    return hashlib.sha256(f"{source_key}\x00{position}".encode()).hexdigest()


def _embedding_input_limit(embedding_model: object) -> int:
    """The token limit of the window the embedding model reads through.

    Nothing else bounds chunk size, and an oversized chunk is truncated by the
    model rather than reported -- which is how chunk tails went missing before.
    A model that does not state its limit is therefore a construction error, not
    a reason to fall back to a guessed constant.
    """
    for attribute in ("max_length", "max_seq_length"):
        limit = getattr(embedding_model, attribute, None)
        if isinstance(limit, int) and limit > 0:
            return limit
    raise ValueError(
        f"Embedding model {type(embedding_model).__name__} states no maximum input "
        "length (max_length or max_seq_length), so chunk sizes cannot be bounded by it."
    )


def _embedding_tokenizer(embedding_model: object):
    """The tokenizer the embedding model reads text with."""
    tokenizer = getattr(getattr(embedding_model, "_model", None), "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            f"Embedding model {type(embedding_model).__name__} exposes no tokenizer, so "
            "chunk sizes cannot be measured in the vocabulary it reads."
        )
    return tokenizer


def _model_token_count(tokenizer) -> Callable[[str], list[int]]:
    """The counter the splitter measures chunk size and overlap with.

    Special tokens are included deliberately. The model's window covers the
    ``[CLS]``/``[SEP]`` it adds itself, so a count that leaves them out overstates
    the room a chunk has: against a 256-token window, 256 tokens of content means
    the last two are truncated away.

    This is returned as a callable rather than handing over the tokenizer object,
    because the ``LlamaSettings.tokenizer`` setter rewrites a
    ``PreTrainedTokenizerBase`` into a count that omits special tokens -- exactly
    the undercount this function exists to prevent.

    ``verbose=False`` silences the tokenizer's "sequence longer than the model's
    maximum" warning. The splitter counts whole documents, and counting one is not
    running it through the model, so the warning would announce truncation on every
    long document when none is happening.
    """
    return partial(tokenizer.encode, add_special_tokens=True, verbose=False)


def _token_count(tokenizer) -> Callable[[str], int]:
    """How many tokens the embedding model reads for text, special tokens included.

    The measure `SentenceSplitter` budgets ``chunk_size`` with, so the node floor
    and the chunk ceiling are expressed in one vocabulary and cannot drift apart
    into different ones.
    """
    encode = _model_token_count(tokenizer)
    return lambda text: len(encode(text))


def _content_token_ids(tokenizer, text: str) -> list[int]:
    """Token ids for text as written, without the special tokens the model adds."""
    return tokenizer.encode(text, add_special_tokens=False)


# A markdown table row carries at least two cell separators. A line with fewer
# is prose that happens to mention a pipe.
_TABLE_CELL_SEPARATORS = 2
_SEPARATOR_CELL = re.compile(r"^:?-+:?$")
# How merged text is rejoined: the splitter's own paragraph break, so a node
# folded together out of two reads as two paragraphs rather than one run-on.
_MERGE_SEPARATOR = "\n\n\n"


def _is_table_line(line: str) -> bool:
    """Whether a line is a markdown table row."""
    return line.count("|") >= _TABLE_CELL_SEPARATORS


def _is_separator_row(line: str) -> bool:
    """Whether a line is a markdown table's ``| --- | --- |`` header underline."""
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(_SEPARATOR_CELL.match(cell) for cell in cells)


# A markdown heading as Docling writes one: one to six hashes, then the heading's
# own text. A line that only looks like a heading -- a hash comment inside a fenced
# code block -- is excluded by the fence tracking in `_split_sections`.
_HEADING_LINE = re.compile(r"^#{1,6}\s+\S")


@dataclass(frozen=True)
class _Unit:
    """One unit of a document: a paragraph of prose, or a run of table rows."""

    kind: str
    text: str


@dataclass(frozen=True)
class _Section:
    """A document's content under one heading, or the content before its first."""

    heading: str | None
    units: tuple[_Unit, ...]


def _is_heading_line(line: str) -> bool:
    """Whether a line opens one of the document's own sections."""
    return _HEADING_LINE.match(line.strip()) is not None


def _split_sections(markdown: str) -> list[_Section]:
    """Split markdown into the document's own sections, each with its units.

    A heading line opens a section and nothing crosses one: that is the document
    stating where one topic ends and the next begins, which is a better boundary
    than any the pipeline could infer. Within a section a blank line ends a unit,
    which is the author's own paragraph break -- and it is the structure a source
    whose headings were dropped upstream still has left.

    A run of consecutive table rows is a unit of its own, so a table reaches
    `_table_parts` whole; a blank line ends that run, which is how Docling writes
    tables. Consecutive non-blank prose lines are one unit, so a paragraph written
    across several lines stays whole, and only a blank line ends it.

    A heading counts only outside a fenced code block, because a ``# comment``
    inside one is a line of the sample rather than a section opening, and cutting
    there would store half a code block. Table detection is deliberately left
    unguarded by the fence: the table path is not what this split changes, and a
    fenced run of pipes reaching `_table_parts` is the behaviour it has always had.
    """
    sections: list[_Section] = []
    heading: str | None = None
    units: list[_Unit] = []
    lines: list[str] = []
    kind: str | None = None
    fenced = False

    def flush_unit() -> None:
        if kind is not None and lines:
            units.append(_Unit(kind, "\n".join(lines)))
        lines.clear()

    def flush_section() -> None:
        nonlocal kind
        flush_unit()
        if heading is not None or units:
            sections.append(_Section(heading, tuple(units)))
        units.clear()
        kind = None

    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
        elif not fenced and _is_heading_line(stripped):
            flush_section()
            heading = stripped
            continue
        if not stripped:
            flush_unit()
            kind = None
            continue
        line_kind = "table" if _is_table_line(line) else "prose"
        if line_kind != kind:
            flush_unit()
            kind = line_kind
        lines.append(line)
    flush_section()
    return sections


def _heading_cost(heading: str | None, count: Callable[[str], int]) -> int:
    """What carrying *heading* costs the budget of every node that carries it.

    Measured rather than assumed: the model reads a node as one sequence, so the
    heading's cost is the difference between counting a text with the heading and
    the blank line above it prepended and counting that text alone. A node packed
    to the chunk size and then given its heading would otherwise sit over the cap
    by exactly this.
    """
    if not heading:
        return 0
    return count(f"{heading}\n\nx") - count("x")


def _splitter_document(text: str, source_key: str, metadata: dict) -> LlamaDocument:
    """A document holding one unit, for the sentence splitter to divide.

    The metadata exclusion has to be stated here too, and not only on the node
    built from the split: the splitter reads this document's metadata for no
    reason but to charge the chunk-size budget before it splits, and the nodes it
    returns are discarded, so without the exclusion tagging a source would change
    where it splits.
    """
    return LlamaDocument(
        text=text,
        id_=source_key,
        metadata=metadata,
        excluded_embed_metadata_keys=list(_MODEL_EXCLUDED_KEYS),
        excluded_llm_metadata_keys=list(_MODEL_EXCLUDED_KEYS),
    )


def _token_spans(tokenizer, text: str) -> list[tuple[int, int]] | None:
    """Where each content token sits in *text*, as far as the tokenizer reports it.

    ``None`` when the tokenizer reports no spans, which is the case for the slow
    tokenizers rather than the fast ones this pipeline is built on.
    """
    try:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, ValueError, NotImplementedError):
        return None
    spans = encoding.get("offset_mapping")
    return [tuple(span) for span in spans] if spans else None


def _even_spans(text: str, token_count: int) -> list[tuple[int, int]]:
    """Character spans standing in for tokens, when the tokenizer reports none.

    Approximate by construction, so a piece cut this way can still be over the
    budget; it is the caller that measures. What it does guarantee is that every
    cut leaves the text shorter, which is what keeps a caller bounding the piece
    from cutting it the same way a second time.
    """
    step = len(text) / token_count
    return [(int(index * step), int((index + 1) * step)) for index in range(token_count)]


def _split_oversized_text(text: str, budget: int, tokenizer) -> list[str]:
    """Cut text with no boundary left to cut on into at most *budget* tokens.

    *budget* is counted in content tokens, without the boundary tokens the model
    adds to whatever it reads: callers that concatenate something onto a piece
    spend the model's overhead from their own budget instead. Callers bounding a
    whole node use `_bounded_parts`, which measures the node rather than
    trusting that the two budgets add up.

    The pieces are slices of the text itself. Cutting by decoding token ids
    cannot return the text it started from: measured on the corpus, a Vietnamese
    table row came back with its diacritics stripped, its case lowered and a
    WordPiece continuation marker left inside a word -- text the source never
    contained, which a query for the original wording could no longer find.

    Reached where a single markdown table row is wider than the embedding
    window: no boundary remains within it, so it is cut on token boundaries.
    """
    token_ids = _content_token_ids(tokenizer, text)
    if len(token_ids) <= budget:
        return [text]
    spans = _token_spans(tokenizer, text)
    if spans is None or len(spans) != len(token_ids):
        spans = _even_spans(text, len(token_ids))
    pieces: list[str] = []
    index = 0
    while index < len(token_ids):
        stop = min(index + budget, len(token_ids))
        first, last = spans[index][0], spans[stop - 1][1]
        boundary = stop
        if stop < len(token_ids):
            # End the piece where the text already breaks rather than inside a word,
            # and start the next piece from there. Nothing is dropped but the space:
            # a piece is always a whole number of tokens, ending on a token boundary.
            whitespace = text.rfind(" ", first, last)
            if whitespace > first:
                boundary = next(
                    (i for i in range(index + 1, stop) if spans[i][0] >= whitespace), stop
                )
                last = spans[boundary - 1][1]
        piece = text[first:last].strip()
        if piece:
            pieces.append(piece)
        index = boundary
    return pieces


def _bounded_parts(text: str, limit: int, count: Callable[[str], int], tokenizer) -> list[str]:
    """Cut *text* until every piece is within *limit* as the model counts it.

    The cut is made in content tokens while the limit counts the boundary tokens
    the model adds itself, so the budget comes from measuring that overhead
    rather than assuming it -- and each piece is then measured again, because
    decoding a cut and re-encoding it can move the boundary by a token. Measured
    on the corpus, a table part cut to the budget came back at 257 tokens against
    a 256-token window, one token of which the model truncated.
    """
    if count(text) <= limit:
        return [text]
    budget = max(limit - count(""), 1)
    pieces: list[str] = []
    for piece in _split_oversized_text(text, budget, tokenizer):
        # Whatever still does not fit is over by construction means more than the
        # budget of content, so cutting it again always makes progress.
        pieces.extend(_bounded_parts(piece, limit, count, tokenizer))
    return pieces


def _canonical_separator(line: str) -> str:
    """A table's separator row in its shortest equivalent form.

    Docling writes the row wide enough to span its widest cell, and a dash run
    costs the tokenizer about two tokens per dash: measured on the corpus, a
    fourteen-row table's separator row was 1171 tokens against a 256-token
    window. Repeating that row in every part put each part past the window on the
    strength of its punctuation alone, and a part the model cannot read whole is
    the truncation this pipeline exists to prevent.

    A separator's width states nothing in markdown -- only the number of columns
    and their alignment do -- so it is stored in the shortest form that states
    both. Measured on the same table, that is 1171 tokens down to 18.
    """
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    shortened = [
        f"{':' if cell.startswith(':') else ''}---{':' if cell.endswith(':') else ''}"
        for cell in cells
    ]
    return f"| {' | '.join(shortened)} |"


def _table_parts(text: str, chunk_size: int, tokenizer) -> list[str]:
    """The parts a markdown table block becomes, each carrying its header.

    A table that fits is stored whole. One that does not is cut at row
    boundaries, repeating the header and separator rows at the top of every part,
    so each stored row names its own columns: measured, the row
    ``| | Thư tịch | Không rõ`` stored on its own is unanswerable, because the
    header saying what the columns mean was cut into a different node.

    Every part is then bounded by the window. That is a backstop rather than the
    usual case: a part is assembled from whole rows that were measured to fit
    beside the header, so it is within the window already -- but only where the
    header itself is, and a table whose own header row is wider than the window
    has no boundary left that would keep it whole.
    """
    count = _token_count(tokenizer)
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return []
    head = lines[:2] if len(lines) > 1 and _is_separator_row(lines[1]) else lines[:1]
    if len(head) == 2:
        head = [head[0], _canonical_separator(head[1])]
    rows_lines = lines[len(head) :]
    whole = "\n".join([*head, *rows_lines])
    if count(whole) <= chunk_size:
        return [whole]

    # The header is repeated in every part, so it is spent from the part's budget
    # rather than added on top of it -- as are the boundary tokens the model adds
    # to the part, which the header's own count has already paid for once.
    header = "\n".join(head)
    budget = max(chunk_size - count(header) - count(""), 1)
    parts: list[str] = []
    rows: list[str] = []
    for row in rows_lines:
        if rows and count("\n".join([*head, *rows, row])) > chunk_size:
            parts.append("\n".join([*head, *rows]))
            rows = []
        if count("\n".join([*head, row])) > chunk_size:
            # A single row that does not fit beside the header: no boundary
            # remains within it, so it is cut by tokens and keeps the header.
            parts.extend(
                "\n".join([header, piece])
                for piece in _split_oversized_text(row, budget, tokenizer)
            )
        else:
            rows.append(row)
    if rows:
        parts.append("\n".join([*head, *rows]))
    if not parts:
        parts = [header]

    bounded: list[str] = []
    for part in parts:
        bounded.extend(_bounded_parts(part, chunk_size, count, tokenizer))
    return bounded


def _join_text(earlier: str, later: str) -> str:
    """Concatenate two nodes' text, keeping one copy of a heading they share.

    A heading the two texts both lead with names the node once, not once per text
    joined into it: the merged node already opens with it, so the second copy says
    nothing the first did not. Measured on one source, merges stored its heading 86
    times over and those copies were 15% of everything the source indexed, with a
    single node carrying 15 of them. A heading only one of the texts carries is
    kept, because it is where the section it opens begins inside the merged node.
    """
    return f"{earlier}{_MERGE_SEPARATOR}{_without_repeated_heading(earlier, later)}"


def _without_repeated_heading(earlier: str, later: str) -> str:
    """*later* without the leading heading line it repeats from *earlier*.

    Only the leading line is compared, and only when it is a heading, so two nodes
    opening with the same ordinary line are left alone -- two nodes holding the
    same text is `_dedupe_texts`'s business, not this function's.
    """
    first_earlier = earlier.split("\n", 1)[0]
    first_later, separator, rest = later.partition("\n")
    if not separator or not rest or first_later != first_earlier:
        return later
    if not _is_heading_line(first_later):
        return later
    return rest.lstrip("\n")


def _merge_below_floor(
    texts: Sequence[str],
    floor: int,
    ceiling: int,
    count: Callable[[str], int],
    origins: Sequence[int] | None = None,
) -> tuple[list[str], int, int]:
    """Fold nodes too small to carry information into a neighbour.

    A node below *floor* is joined to the node before it. The first node of a
    document has no predecessor, so it is offered to the one after it instead.
    Merging concatenates text, so nothing is dropped to satisfy the floor --
    which is the point: a fragment is usually real content that lost the context
    naming it. The one thing not repeated is a heading both nodes already carry,
    which names the merged node once rather than once per text folded into it;
    see `_join_text`.

    A merge that would push the node past *ceiling*, the embedding model's input
    length, is refused and the fragment is offered to its other neighbour. An
    oversized node is the truncation this pipeline exists to prevent, so trading
    that invariant away to satisfy the floor would trade a real loss for a tidy
    node count. A fragment neither neighbour can legally take is stored on its
    own: small, and honestly so.

    When *origins* is given -- one section index per text, in the same order -- a
    fragment is offered the neighbours inside its own section before the ones a
    heading separates it from, so reaching the minimum does not join two of the
    document's sections into one node. Preference rather than prohibition,
    because the floor is a requirement: a fragment whose only legal merge lies
    across a heading is still merged rather than left to stand alone.

    A document whose entire content is below the floor is left as the single node
    it is rather than merged away.

    Returns the surviving texts, how many nodes were merged away, and how many
    fragments no merge was legal for.
    """
    surviving = list(texts)
    # The section each text came from, carried alongside it and shrunk in the same
    # places, so a merge cannot leave a boundary pointing at text that has moved.
    sections = list(origins) if origins is not None else None
    merges = 0
    refusals = 0
    index = 0
    while index < len(surviving):
        if len(surviving) == 1:
            break
        if count(surviving[index]) >= floor:
            index += 1
            continue
        # The preceding node keeps reading order; the first node of a document
        # has none and is offered forward instead.
        neighbours = [n for n in (index - 1, index + 1) if 0 <= n < len(surviving)]
        if sections is not None:
            # Same-section neighbours first, then the ones beyond a heading. The
            # sort is stable, so neighbours of one section keep reading order.
            neighbours.sort(key=lambda other: sections[other] != sections[index])
        for other in neighbours:
            joined = _join_text(surviving[min(index, other)], surviving[max(index, other)])
            if count(joined) > ceiling:
                continue
            surviving[max(index, other)] = joined
            del surviving[min(index, other)]
            if sections is not None:
                # The merged node sits where the earlier of the two did, and
                # keeps that one's section: it is the section it opens in.
                del sections[max(index, other)]
            merges += 1
            # Re-examine the node just grown: it may itself still be below the
            # floor, which keeps the floor a fixpoint rather than a single pass.
            index = min(index, other)
            break
        else:
            refusals += 1
            index += 1
    return surviving, merges, refusals


def _dedupe_texts(texts: Sequence[str]) -> tuple[list[str], int]:
    """Keep one node per distinct text, reporting how many were dropped.

    Applied only to the nodes of a single document. Two sources holding the same
    text must each keep their own node, because the node an edited source deletes
    is located by document identity: a node both sources claimed would be removed
    on behalf of one of them and taken away from the other.

    Returns the surviving texts and how many were removed as repeated.
    """
    distinct: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for text in texts:
        if text in seen:
            duplicates += 1
            continue
        seen.add(text)
        distinct.append(text)
    return distinct, duplicates


def _content_below_heading(text: str) -> str:
    """A node's text without the heading line it leads with, if it leads with one.

    Every node divided out of a section carries that section's heading, so two
    adjacent nodes of one section open with the same line -- and that line is not
    overlap. It is the same heading because both nodes belong to the section it
    names, so comparing the texts as written would score the sentences the
    splitter did repeat, sitting right below the heading, as no overlap at all:
    measured, a section divided into sentence-bounded nodes reported an overlap
    of 0 while every adjacent pair shared its trailing sentence.
    """
    first, separator, rest = text.partition("\n")
    return rest if separator and rest and _is_heading_line(first) else text


def _boundary_overlap(earlier: Sequence[int], later: Sequence[int]) -> int:
    """Tokens the earlier node ends with that the later node also begins with.

    The splitter repeats whole trailing sentences, so the overlap is the longest
    suffix/prefix match. Content tokens are compared rather than the model's full
    input, because the special tokens are added per sequence and belong to no
    node's text -- counting them would credit a boundary with overlap it has not
    got.
    """
    for size in range(min(len(earlier), len(later)), 0, -1):
        if earlier[-size:] == later[:size]:
            return size
    return 0


def _chunking_outcome(
    nodes: Sequence[BaseNode],
    tokenizer,
    *,
    floor_merges: int = 0,
    floor_refusals: int = 0,
    duplicate_nodes: int = 0,
    structural_nodes: int = 0,
    divided_nodes: int = 0,
) -> dict[str, int | None]:
    """What the chunking actually achieved, in the embedding model's tokens.

    ``max_node_tokens`` is measured as the model reads a node, special tokens
    included, so it is directly comparable to the model's input limit and any
    value above it means content is being truncated. ``overlap_tokens`` is the
    smallest overlap across adjacent nodes -- the conservative answer to whether
    the configured overlap was delivered -- and ``None`` when a source produced
    too few nodes to have a boundary at all. It is measured below the heading a
    node leads with, because adjacent nodes of one section share that heading by
    construction and it is not themselves repeated. The counts that follow say what the
    minimum node size and the collapse of repeated text changed, so their effect
    is visible rather than inferred from the node count.

    ``structural_nodes`` and ``divided_nodes`` say where the boundaries came from:
    nodes stored under a boundary the document itself provided, and nodes that
    had to be divided further because one of the document's units exceeded the
    chunk size. They are counted as the boundaries are chosen, before the floor
    and the collapse change what survives, so they describe how the nodes were
    produced rather than how many are stored -- which is what makes a source whose
    structure was too coarse to bound its nodes visible rather than merely
    unusual.
    """
    count = _token_count(tokenizer)
    sizes = [count(node.get_content()) for node in nodes]
    overlaps = [
        _boundary_overlap(
            _content_token_ids(tokenizer, _content_below_heading(earlier.get_content())),
            _content_token_ids(tokenizer, _content_below_heading(later.get_content())),
        )
        for earlier, later in zip(nodes, nodes[1:], strict=False)
    ]
    return {
        "nodes": len(nodes),
        "max_node_tokens": max(sizes, default=0),
        "overlap_tokens": min(overlaps) if overlaps else None,
        "floor_merges": floor_merges,
        "floor_refusals": floor_refusals,
        "duplicate_nodes": duplicate_nodes,
        "structural_nodes": structural_nodes,
        "divided_nodes": divided_nodes,
    }


def _pdf_format_options(languages: Sequence[str]) -> dict[InputFormat, PdfFormatOption]:
    """Docling's options for the PDF pipeline, with OCR in *languages*.

    Docling's own default reads every image with a Chinese/English recogniser, so
    a Vietnamese scan comes back without its diacritics -- and in Vietnamese the
    diacritics are the word, not decoration. Measured on a 162-page scan, that was
    57% of the diacritics lost, so the pages no longer matched a query for their
    own wording. Naming the language is what fixes it.

    Two fields are deliberately left alone:

    * The OCR mode, so OCR applies to a page only where there is no text to read.
      A page carrying its own text layer keeps it rather than having accurate text
      replaced by recognised text. (``force_full_page_ocr`` is the field this used
      to be expressed with; it is deprecated in favour of ``mode``, and the
      default mode is the same behaviour.)
    * ``use_gpu``, which is deprecated in favour of Docling's accelerator options
      and warns at runtime when set. Device selection stays with Docling.

    Only the PDF entry is configured: OCR is performed by the PDF pipeline, so
    DOCX, HTML, XLSX and the rest keep the behaviour they have today. The
    threaded pipeline is the one Docling's own default PDF entry uses, so
    choosing it here keeps page handling as it was and changes only the OCR --
    the base class carries the same settings but selects the single-threaded
    pipeline, which would quietly slow every PDF down.
    """
    pipeline_options = ThreadedPdfPipelineOptions()
    pipeline_options.ocr_options = EasyOcrOptions(lang=list(languages))
    return {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}


class DoclingConverter:
    """Convert files and URLs to structured markdown using Docling."""

    SUPPORTED_FILE_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".html",
        ".htm",
        ".txt",
        ".md",
        ".json",
        ".xml",
    }

    def __init__(self, converter: DocumentConverter | None = None) -> None:
        """Build a converter configured for this deployment, or take the one given.

        Only the fallback is configured. An injected converter is used exactly as
        handed over, so a caller can supply one it controls -- which is what the
        tests do, and why the injection point is kept rather than replaced.
        """
        self._converter = converter or DocumentConverter(
            format_options=_pdf_format_options(settings.pdf_ocr_language_list)
        )

    def is_supported_file(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.SUPPORTED_FILE_EXTENSIONS

    def convert_file(self, file: BinaryIO, filename: str) -> str:
        suffix = Path(filename).suffix.lower() or ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = Path(tmp.name)

        try:
            result: ConversionResult = self._converter.convert(tmp_path)
            if result.status != ConversionStatus.SUCCESS:
                raise RuntimeError(f"Docling failed to convert {filename}: {result.status.value}")
            return result.document.export_to_markdown()
        finally:
            tmp_path.unlink(missing_ok=True)

    def convert_url(
        self, url: str, timeout: int = 30, user_agent: str | None = None
    ) -> tuple[str, str]:
        # The request names this service. Left to the HTTP library's own default,
        # hosts that require a client identity refuse the fetch outright -- a
        # Wikipedia page answers 403 to `python-requests/*` and 200 to a request
        # that says who it is -- so the default here is never "send no header".
        response = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": user_agent or DEFAULT_USER_AGENT},
        )
        response.raise_for_status()
        final_url = response.url

        # Only the page's main content is converted. Left whole, the navigation,
        # sidebars, banners and footer are chunked and embedded alongside it.
        selected = select_main_content(response.content, final_url)

        suffix = ".html"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(selected.encode())
            tmp_path = Path(tmp.name)

        try:
            result: ConversionResult = self._converter.convert(tmp_path)
            if result.status != ConversionStatus.SUCCESS:
                raise RuntimeError(f"Docling failed to convert {url}: {result.status.value}")
            return result.document.export_to_markdown(), final_url
        finally:
            tmp_path.unlink(missing_ok=True)


class IndexPipeline:
    """Orchestrate chunking, embedding, and indexing of parsed content."""

    def __init__(
        self,
        app_settings: Settings,
        converter: DoclingConverter | None = None,
        embedding_model: HuggingFaceEmbedding | None = None,
    ) -> None:
        self._settings = app_settings
        self._converter = converter or DoclingConverter()
        self._embedding_model = embedding_model or HuggingFaceEmbedding(
            model_name=app_settings.embedding_model
        )
        # Chunk sizing has to be expressed in the vocabulary the embedding model
        # reads and bounded by the window it reads through: previously the two
        # were unrelated -- tiktoken-counted 512 against a 256-token window -- so
        # the tail of every chunk was discarded before it was ever embedded.
        self._max_input_tokens = _embedding_input_limit(self._embedding_model)
        self._tokenizer = _embedding_tokenizer(self._embedding_model)
        self._chunk_size = self._resolve_chunk_size(app_settings.chunk_size)
        self._min_chunk_tokens = app_settings.min_chunk_tokens

        LlamaSettings.embed_model = self._embedding_model
        LlamaSettings.tokenizer = _model_token_count(self._tokenizer)
        LlamaSettings.node_parser = SentenceSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
        )
        self._index = VectorStoreIndex(nodes=[])
        # Guards the shared vector store. Ingestion runs in the Starlette
        # threadpool while search runs on the event loop, so both the write in
        # the ingest methods and the read in `search` must hold this lock.
        self._index_lock = threading.Lock()
        # What is indexed, for the readiness endpoint and the source catalog.
        # `_source_records` is only ever touched under the lock, and each write
        # publishes `_source_catalog` as an immutable snapshot, so a reader costs
        # no retrieval, no embedding work, and no lock. Publishing is what keeps a
        # reader from iterating a dict another thread is mutating.
        self._source_records: dict[str, SourceRecord] = {}
        self._source_catalog: tuple[SourceRecord, ...] = ()
        # Each source's node ids mapped to the position each holds within it,
        # written with the source's record so the two cannot disagree. The
        # position is what makes a hit's neighbours findable: it is computed when
        # the ids are assigned and then hashed into the id, so a hit cannot be
        # placed in its source from the hit alone. It is kept here rather than in
        # the node's metadata because metadata not listed as excluded is embedded
        # with the text, and this number would then be part of every stored
        # vector instead of bookkeeping beside the store.
        self._source_positions: dict[str, dict[str, int]] = {}
        self._indexed_sources = 0
        self._indexed_chunks = 0

    @property
    def converter(self) -> DoclingConverter:
        return self._converter

    @property
    def index(self) -> VectorStoreIndex:
        return self._index

    @property
    def indexed_sources(self) -> int:
        return self._indexed_sources

    @property
    def indexed_chunks(self) -> int:
        return self._indexed_chunks

    @property
    def source_catalog(self) -> tuple[SourceRecord, ...]:
        """Every indexed source, in a stable order, safe to read without the lock.

        An immutable snapshot published by the write that produced it, so a
        caller cannot observe a source part-way through being replaced and does
        not wait on ingestion to read it.
        """
        return self._source_catalog

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _resolve_chunk_size(self, configured: int | None) -> int:
        """The chunk size to use, bounded by what the embedding model can read.

        An unset setting follows the model's own input limit, so the default
        cannot drift from the model it feeds. An explicit setting is honoured up
        to that limit; above it the pipeline refuses to start rather than produce
        nodes whose tails the model would truncate.
        """
        if configured is None:
            return self._max_input_tokens
        if configured > self._max_input_tokens:
            raise ValueError(
                f"CHUNK_SIZE={configured} exceeds the maximum input length of embedding "
                f"model '{self._settings.embedding_model}' ({self._max_input_tokens} "
                "tokens). Chunks larger than that would be truncated before they were "
                f"embedded; set CHUNK_SIZE to {self._max_input_tokens} or less."
            )
        return configured

    def _identify_nodes(self, nodes: Sequence[BaseNode], source_key: str) -> None:
        """Give every node an identity derived from its source and its position."""
        for position, node in enumerate(nodes):
            node.id_ = _node_id(source_key, position)

    def _build_node(self, text: str, source_key: str, metadata: dict) -> BaseNode:
        """A node carrying its text, its source's identity, and its metadata."""
        node = TextNode(
            text=text,
            metadata=dict(metadata),
            # Collections are a filter key, not content: the prefilter reads them
            # off the node and the model never does. See `_MODEL_EXCLUDED_KEYS`.
            excluded_embed_metadata_keys=list(_MODEL_EXCLUDED_KEYS),
            excluded_llm_metadata_keys=list(_MODEL_EXCLUDED_KEYS),
        )
        # Set as a relationship rather than a field, because `ref_doc_id` is a
        # read-only property derived from the source relation -- and it is what
        # `delete_ref_doc` locates a source's earlier content by.
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=source_key)
        return node

    def _chunk(
        self, markdown: str, source_key: str, metadata: dict
    ) -> tuple[list[BaseNode], dict[str, int | None]]:
        """Split markdown into the nodes this source will store.

        The document's own structure decides the boundaries: a heading opens a
        section, a blank line ends a unit within it, and a unit that fits the chunk
        size is stored as the node it is -- the size bounds a node rather than
        joining two of the document's units into one, and no node crosses a
        heading. Only a unit that still exceeds the chunk size on its own is
        divided further. The floor then folds away what is too small to carry
        information -- preferring a neighbour inside the fragment's own section,
        and never past the embedding model's input length, since an oversized node
        is the truncation this pipeline exists to prevent -- and text repeated
        inside this one source collapses to a single node. All of it runs before
        any embedding is computed, so what is reported and what is embedded
        describe the same nodes. Positions are numbered after all three, so the
        surviving nodes stay contiguous and identity remains source and position.
        """
        count = _token_count(self._tokenizer)
        texts: list[str] = []
        # Which section each node came from, in the same order, so the floor can
        # prefer a neighbour it does not have to cross a heading to reach.
        origins: list[int] = []
        structural = 0
        divided = 0
        for origin, section in enumerate(_split_sections(markdown)):
            section_texts, section_structural, section_divided = self._section_nodes(
                section, source_key, metadata, count
            )
            texts.extend(section_texts)
            origins.extend([origin] * len(section_texts))
            structural += section_structural
            divided += section_divided

        texts, floor_merges, floor_refusals = _merge_below_floor(
            texts, self._min_chunk_tokens, self._max_input_tokens, count, origins
        )
        texts, duplicate_nodes = _dedupe_texts(texts)

        nodes = [self._build_node(text, source_key, metadata) for text in texts]
        self._identify_nodes(nodes, source_key)
        chunking = _chunking_outcome(
            nodes,
            self._tokenizer,
            floor_merges=floor_merges,
            floor_refusals=floor_refusals,
            duplicate_nodes=duplicate_nodes,
            structural_nodes=structural,
            divided_nodes=divided,
        )
        return nodes, chunking

    def _section_nodes(
        self, section: _Section, source_key: str, metadata: dict, count: Callable[[str], int]
    ) -> tuple[list[str], int, int]:
        """The nodes one section produces, and how their boundaries were chosen.

        A unit the document offers is stored as the node it is: a paragraph the
        document separated from its neighbours is a thought of its own, so units
        that fit are never packed together to fill the chunk size. The heading
        that opened the section leads every node the section produces -- the
        argument `_table_parts` already makes for repeating a table's header,
        since a node holding one part of a section without the heading naming it
        states no topic of its own.

        Only a unit that still exceeds the chunk size on its own is divided
        further: a paragraph with no paragraph break left in it, or a table too
        wide to fit, offers the document's own boundaries no more than that, and a
        sentence boundary is the last resort rather than the first tool.

        Returns the node texts, how many nodes were bounded by the document's own
        structure, and how many had to be divided because one unit exceeded it.
        """
        heading = section.heading
        cap = self._chunk_size
        # A heading is spent from the budget of every node that carries it rather
        # than added on top, or a node packed to the cap and then given its
        # heading would sit over the cap by exactly the heading's own cost.
        budget = max(cap - _heading_cost(heading, count), 1)

        def node_text(units: Sequence[str]) -> str:
            return "\n\n".join(([heading] if heading else []) + list(units))

        texts: list[str] = []
        structural = 0
        divided = 0
        splitter: SentenceSplitter | None = None

        for unit in section.units:
            if unit.kind == "table":
                parts = _table_parts(unit.text, budget, self._tokenizer)
                if len(parts) == 1:
                    # A table is a unit of the document's own making, stored
                    # whole, so it is structurally bounded like any paragraph.
                    texts.append(node_text(parts))
                    structural += 1
                else:
                    # Too wide to fit: its rows are the only boundary it has left.
                    texts.extend(node_text([part]) for part in parts)
                    divided += len(parts)
                continue
            if count(node_text([unit.text])) <= cap:
                # It fits as it stands, so the document's own boundary is the
                # node's boundary and the cap has no boundary to add: joining it
                # to its neighbour would make one node of two thoughts.
                texts.append(node_text([unit.text]))
                structural += 1
                continue
            # A single unit wider than the cap with no paragraph break left inside
            # it, so it is divided where the splitter can: at sentence boundaries,
            # which is the only place the configured overlap still applies.
            if splitter is None:
                splitter = (
                    LlamaSettings.node_parser
                    if budget == cap
                    else SentenceSplitter(
                        chunk_size=budget, chunk_overlap=self._settings.chunk_overlap
                    )
                )
            pieces: list[str] = []
            for node in splitter.get_nodes_from_documents(
                [_splitter_document(unit.text, source_key, metadata)]
            ):
                # A backstop for a single sentence wider than the budget, which is
                # no sentence boundary at all: it is cut by tokens to fit.
                pieces.extend(_bounded_parts(node.get_content(), budget, count, self._tokenizer))
            texts.extend(node_text([piece]) for piece in pieces)
            divided += len(pieces)
        return texts, structural, divided

    def _replace_source(
        self,
        source_key: str,
        nodes: Sequence[BaseNode],
        *,
        name: str,
        source_type: str,
        collections: Sequence[str] = (),
    ) -> None:
        """Store *nodes* as the whole of *source_key*, replacing its earlier content.

        Deletion is by document identity, so a source that has never been
        ingested needs no special case: removing content that is not there is a
        no-op rather than an error. Both halves run inside the index lock, so a
        concurrent search cannot observe a source half-replaced. The counters are
        recomputed from what each source currently holds, which is what keeps
        them describing stored content rather than submitted content.

        The source's collections are replaced by the same write that replaces its
        nodes, so what a source belongs to cannot drift from what it holds: a
        re-ingestion is always the whole of what that source is.
        """
        with self._index_lock:
            self._index.delete_ref_doc(source_key, delete_from_docstore=True)
            self._index.insert_nodes(nodes)
            self._source_records[source_key] = SourceRecord(
                name=name,
                address=source_key,
                source_type=source_type,
                collections=tuple(collections),
                chunk_count=len(nodes),
            )
            # Written from the same nodes the line above counts, so a source's
            # recorded order describes what the store holds and a re-ingestion
            # replaces it rather than leaving positions from the content before.
            self._source_positions[source_key] = {
                node.node_id: position for position, node in enumerate(nodes)
            }
            self._source_catalog = tuple(
                self._source_records[key] for key in sorted(self._source_records)
            )
            self._indexed_sources = len(self._source_records)
            self._indexed_chunks = sum(item.chunk_count for item in self._source_records.values())

    def ingest_file(
        self,
        source_id: str,
        file: BinaryIO,
        filename: str,
        mime_type: str | None = None,
        collections: Sequence[str] = (),
    ) -> tuple[dict, dict[str, float]]:
        timings: dict[str, float] = {}

        parse_start = time.perf_counter()
        markdown = self._converter.convert_file(file, filename)
        timings["parse_ms"] = _elapsed_ms(parse_start)

        # A file's identity is its filename, so re-uploading one replaces the
        # content its earlier upload indexed rather than adding a second copy
        # beside it. Setting it as the document's id is what lets the index
        # locate that earlier content without a registry.
        source_key = filename
        collections = list(collections)
        metadata = {
            "source_id": source_id,
            "source_type": "file",
            "source_name": filename,
            "mime_type": mime_type,
            ADDRESS_KEY: source_key,
            COLLECTIONS_KEY: collections,
        }

        chunk_start = time.perf_counter()
        nodes, chunking = self._chunk(markdown, source_key, metadata)
        timings["chunk_ms"] = _elapsed_ms(chunk_start)

        embed_start = time.perf_counter()
        for node in nodes:
            node.embedding = self._embedding_model.get_text_embedding(node.get_content())
        timings["embed_ms"] = _elapsed_ms(embed_start)

        index_start = time.perf_counter()
        self._replace_source(
            source_key, nodes, name=filename, source_type="file", collections=collections
        )
        timings["index_ms"] = _elapsed_ms(index_start)

        result = {
            "source_id": source_id,
            "source_type": "file",
            "name": filename,
            "mime_type": mime_type,
            "collections": collections,
            "status": "indexed",
            "chunking": chunking,
        }
        logger.info(
            "Indexed file source=%s filename=%s chunking=%s timings=%s",
            source_id,
            filename,
            chunking,
            timings,
        )
        return result, timings

    def ingest_url(
        self,
        source_id: str,
        url: str,
        timeout: int | None = None,
        collections: Sequence[str] = (),
    ) -> tuple[dict, dict[str, float]]:
        timings: dict[str, float] = {}
        timeout = timeout or self._settings.url_fetch_timeout_seconds

        parse_start = time.perf_counter()
        markdown, final_url = self._converter.convert_url(
            url, timeout=timeout, user_agent=self._settings.resolved_user_agent
        )
        timings["parse_ms"] = _elapsed_ms(parse_start)

        # A page's identity is where it finally landed, not where the request
        # pointed. Two URLs redirecting to one page are one source, and a page
        # is replaced when it is re-submitted rather than indexed twice.
        source_key = final_url
        collections = list(collections)
        metadata = {
            "source_id": source_id,
            "source_type": "url",
            "source_name": url,
            "final_url": final_url,
            ADDRESS_KEY: source_key,
            COLLECTIONS_KEY: collections,
        }

        chunk_start = time.perf_counter()
        nodes, chunking = self._chunk(markdown, source_key, metadata)
        timings["chunk_ms"] = _elapsed_ms(chunk_start)

        embed_start = time.perf_counter()
        for node in nodes:
            node.embedding = self._embedding_model.get_text_embedding(node.get_content())
        timings["embed_ms"] = _elapsed_ms(embed_start)

        index_start = time.perf_counter()
        self._replace_source(
            source_key, nodes, name=url, source_type="url", collections=collections
        )
        timings["index_ms"] = _elapsed_ms(index_start)

        result = {
            "source_id": source_id,
            "source_type": "url",
            "name": url,
            "final_url": final_url,
            "collections": collections,
            "status": "indexed",
            "chunking": chunking,
        }
        logger.info(
            "Indexed URL source=%s url=%s chunking=%s timings=%s",
            source_id,
            url,
            chunking,
            timings,
        )
        return result, timings

    def search(
        self,
        query: str,
        top_k: int,
        collections: Sequence[str] | None = None,
        neighbours: int = 0,
    ) -> list[dict]:
        """Rank chunks against *query*, optionally scoped to *collections*.

        The scope is expressed as a single ``ANY`` entry over the collections
        list, which the store applies as a prefilter before the similarity scan.
        That is what makes the requested count a count *within* the scope: the
        top_k is drawn from the filtered set, so a scope holding fewer chunks
        returns fewer results rather than being padded from outside it.

        A source carrying no collections matches neither operator -- a list
        value is tested by membership, and an empty list contains nothing -- so
        an untagged source falls out of every filter with no special case.

        *neighbours* is how many adjacent chunks to attach on each side of every
        result. It is a count, not a flag: no single number expresses "the
        passage around this hit" for every caller. Zero attaches nothing, which
        is what a caller that has not asked for adjacency receives.
        """
        filters = None
        if collections:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key=COLLECTIONS_KEY,
                        value=list(collections),
                        operator=FilterOperator.ANY,
                    )
                ]
            )
        retriever = self._index.as_retriever(similarity_top_k=top_k, filters=filters)
        with self._index_lock:
            nodes = retriever.retrieve(query)
            results = []
            for hit in nodes:
                chunk = hit.node
                adjacency = self._adjacency(chunk, neighbours)
                results.append(
                    {
                        "text": chunk.get_content(),
                        "score": float(hit.score) if hit.score is not None else 0.0,
                        "source_id": chunk.metadata.get("source_id", ""),
                        "source_type": chunk.metadata.get("source_type", ""),
                        "source_name": chunk.metadata.get("source_name", ""),
                        # Read off the chunk the search already returned, so a hit
                        # costs no extra retrieval and no catalog read: the two
                        # values are written with the chunk's identity, and the
                        # caller needs them per hit rather than per source.
                        ADDRESS_KEY: chunk.metadata.get(ADDRESS_KEY, ""),
                        COLLECTIONS_KEY: chunk.metadata.get(COLLECTIONS_KEY, []),
                        "position": adjacency.position,
                        "neighbours_before": adjacency.before,
                        "neighbours_after": adjacency.after,
                        "neighbours": adjacency.neighbours,
                    }
                )
        return results

    def source_content(self, address: str) -> tuple[SourceRecord, list[dict]] | None:
        """Every stored chunk of the source *address* names, in reading order.

        None for an address no source has, which is a different answer from a
        source that stored no chunks: the second is a source, reported with an
        empty list, and collapsing the two would make "you asked for something
        that is not here" read the same as "here it is, and it was empty" -- the
        reason `_adjacency` reports no adjacency rather than a guessed one.

        The read embeds nothing and retrieves nothing: it walks what the store
        already holds, which is why it can answer for a source of any size without
        loading a model. It holds the same lock the writes hold, because a
        replacement rewrites the docstore, the records and the positions together,
        and a reader outside the lock could take one ingestion's nodes against
        another's positions.

        The order comes from the position map rather than from the docstore's own
        iteration order: the docstore is a dict keyed by node id, so its order is
        incidental, and the position a node was stored at is what makes its chunks
        come back as the document reads.
        """
        with self._index_lock:
            record = self._source_records.get(address)
            if record is None:
                return None
            positions = self._source_positions.get(address, {})
            stored = self._index.docstore.docs
            chunks = [
                {"text": node.get_content(), "position": position}
                for node_id, position in sorted(positions.items(), key=lambda item: item[1])
                if (node := stored.get(node_id)) is not None
            ]
        return record, chunks

    def _adjacency(self, chunk: BaseNode, count: int) -> _Adjacency:
        """Place *chunk* in its source and collect the neighbours *count* asks for.

        A chunk whose placement cannot be established reports no adjacency rather
        than a guessed one, so an unknown is never shown as a neighbour that
        exists.

        The caller holds ``_index_lock``: this reads the records an ingest writes,
        so outside it a replacement could be observed half-applied.
        """
        source_key = chunk.ref_doc_id
        record = self._source_records.get(source_key)
        position = self._source_positions.get(source_key, {}).get(chunk.node_id)
        if position is None or record is None:
            return _Adjacency(0, 0, 0, [])
        # A neighbour is the node the same id function names at an adjacent
        # position, which is what makes it the source's own next chunk rather than
        # whatever else the store holds: nothing outside this source is reachable
        # from the hit's position, and the count bounds both sides.
        stored = self._index.docstore.docs
        neighbours = []
        for offset in range(1, count + 1):
            for place in (position - offset, position + offset):
                if 0 <= place < record.chunk_count:
                    adjacent = stored.get(_node_id(source_key, place))
                    if adjacent is not None:
                        neighbours.append({"text": adjacent.get_content(), "position": place})
        # Walking outwards is what fills the side that has room when the other is
        # short; sorting restores the document's order, which is the order a
        # caller splicing a passage reads in.
        neighbours.sort(key=lambda item: item["position"])
        return _Adjacency(
            position,
            position,
            max(record.chunk_count - position - 1, 0),
            neighbours,
        )


def create_pipeline(
    app_settings: Settings | None = None,
    converter: DoclingConverter | None = None,
    embedding_model: HuggingFaceEmbedding | None = None,
) -> IndexPipeline:
    app_settings = app_settings or settings
    if app_settings.vector_store_backend != VectorStoreBackend.SIMPLE:
        raise NotImplementedError(
            f"Vector store backend {app_settings.vector_store_backend.value} is not implemented."
        )
    return IndexPipeline(app_settings, converter=converter, embedding_model=embedding_model)
