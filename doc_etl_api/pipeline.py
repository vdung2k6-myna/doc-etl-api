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
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
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

# Metadata that is written for filtering but never read by a model. Both the
# embed and the LLM rendering are excluded, and that is not belt-and-braces:
# metadata is charged against the chunk-size budget before splitting, and
# `MetadataAwareTextSplitter._get_metadata_str` renders a node both ways and
# keeps the longer of the two, so excluding one mode alone would still let a
# collection name shrink the text each chunk can hold. That changes where a
# source splits and therefore its vectors -- measured, tagging one document with
# two collections took it from 14 nodes to 20. Excluding the key from both
# readings is what makes tagging metadata-only: the prefilter still sees the
# collections, no model ever does.
_MODEL_EXCLUDED_KEYS = [COLLECTIONS_KEY]


@dataclass(frozen=True)
class SourceRecord:
    """What a source is, as the catalog reports it."""

    name: str
    source_type: str
    collections: tuple[str, ...]
    chunk_count: int


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


def _split_blocks(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into ``("table" | "prose", text)`` blocks, in document order.

    The splitter packs sentences, and a table row is not a sentence: it carries
    no sentence punctuation, so the packer sees a table as one undifferentiated
    run of tokens and cuts it at whatever count it happens to reach. Measured on
    the corpus this was built for, a ten-row infobox became a dozen 9-14 token
    scraps such as ``| | Thư tịch | Không rõ``, each detached from the header
    naming its columns.

    A table block is a run of consecutive table rows; everything else is prose.
    A blank line ends a run, which is how Docling writes tables.
    """
    blocks: list[tuple[str, str]] = []
    kind: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if kind is not None and lines:
            blocks.append((kind, "\n".join(lines)))

    for line in markdown.split("\n"):
        line_kind = "table" if _is_table_line(line) else "prose"
        if line_kind != kind:
            flush()
            kind, lines = line_kind, []
        lines.append(line)
    flush()
    return blocks


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
    """Concatenate two nodes' text without dropping either."""
    return f"{earlier}{_MERGE_SEPARATOR}{later}"


def _merge_below_floor(
    texts: Sequence[str], floor: int, ceiling: int, count: Callable[[str], int]
) -> tuple[list[str], int, int]:
    """Fold nodes too small to carry information into a neighbour.

    A node below *floor* is joined to the node before it. The first node of a
    document has no predecessor, so it is offered to the one after it instead.
    Merging concatenates text, so nothing is dropped to satisfy the floor --
    which is the point: a fragment is usually real content that lost the context
    naming it.

    A merge that would push the node past *ceiling*, the embedding model's input
    length, is refused and the fragment is offered to its other neighbour. An
    oversized node is the truncation this pipeline exists to prevent, so trading
    that invariant away to satisfy the floor would trade a real loss for a tidy
    node count. A fragment neither neighbour can legally take is stored on its
    own: small, and honestly so.

    A document whose entire content is below the floor is left as the single node
    it is rather than merged away.

    Returns the surviving texts, how many nodes were merged away, and how many
    fragments no merge was legal for.
    """
    surviving = list(texts)
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
        for other in neighbours:
            joined = _join_text(surviving[min(index, other)], surviving[max(index, other)])
            if count(joined) > ceiling:
                continue
            surviving[max(index, other)] = joined
            del surviving[min(index, other)]
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
) -> dict[str, int | None]:
    """What the chunking actually achieved, in the embedding model's tokens.

    ``max_node_tokens`` is measured as the model reads a node, special tokens
    included, so it is directly comparable to the model's input limit and any
    value above it means content is being truncated. ``overlap_tokens`` is the
    smallest overlap across adjacent nodes -- the conservative answer to whether
    the configured overlap was delivered -- and ``None`` when a source produced
    too few nodes to have a boundary at all. The three counts that follow say
    what the minimum node size and the collapse of repeated text changed, so
    their effect is visible rather than inferred from the node count.
    """
    count = _token_count(tokenizer)
    sizes = [count(node.get_content()) for node in nodes]
    overlaps = [
        _boundary_overlap(
            _content_token_ids(tokenizer, earlier.get_content()),
            _content_token_ids(tokenizer, later.get_content()),
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
    }


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
        self._converter = converter or DocumentConverter()

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

        Blocks are chunked first (a table by rows, prose by sentence), then the
        floor folds away what is too small to carry information -- never past the
        embedding model's input length, since an oversized node is the truncation
        this pipeline exists to prevent -- then text repeated inside this one
        source collapses to a single node. All three run before any embedding is
        computed, so what is reported and what is embedded describe the same
        nodes. Positions are numbered after both, so the surviving nodes stay
        contiguous and identity remains source and position.
        """
        count = _token_count(self._tokenizer)
        texts: list[str] = []
        for kind, block in _split_blocks(markdown):
            if kind == "table":
                texts.extend(_table_parts(block, self._chunk_size, self._tokenizer))
                continue
            texts.extend(
                node.get_content()
                for node in LlamaSettings.node_parser.get_nodes_from_documents(
                    [
                        LlamaDocument(
                            text=block,
                            id_=source_key,
                            metadata=metadata,
                            # The splitter reads this document's metadata only to
                            # charge the chunk-size budget -- the nodes it returns
                            # are discarded and rebuilt by `_build_node` -- so the
                            # exclusion has to be stated here as well, or tagging a
                            # source would change how it splits.
                            excluded_embed_metadata_keys=list(_MODEL_EXCLUDED_KEYS),
                            excluded_llm_metadata_keys=list(_MODEL_EXCLUDED_KEYS),
                        )
                    ]
                )
            )

        texts, floor_merges, floor_refusals = _merge_below_floor(
            texts, self._min_chunk_tokens, self._max_input_tokens, count
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
        )
        return nodes, chunking

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
                source_type=source_type,
                collections=tuple(collections),
                chunk_count=len(nodes),
            )
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
        self, query: str, top_k: int, collections: Sequence[str] | None = None
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
        for node in nodes:
            results.append(
                {
                    "text": node.node.get_content(),
                    "score": float(node.score) if node.score is not None else 0.0,
                    "source_id": node.node.metadata.get("source_id", ""),
                    "source_type": node.node.metadata.get("source_type", ""),
                    "source_name": node.node.metadata.get("source_name", ""),
                }
            )
        return results


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
