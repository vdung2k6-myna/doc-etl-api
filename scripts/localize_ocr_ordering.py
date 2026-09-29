"""Localize the stage that reorders text in a scanned document.

A scanned Vietnamese PDF has its diacritics read correctly and its text assembled
out of order: measured, EasyOCR reads a sentence's words correctly and the phrase
comes out as ``Hãy đây`` with ``nghe`` appended later. Recognition was fixed in
``doc-etl-api-vietnamese-pdf-ocr``; assembly was left out of scope there, and
nothing established which stage is responsible.

Three stages can move a word, and the fixes for them are disjoint -- so the choice
cannot be made before one is named:

* **the detector.** In the configured OCR mode, the page is not read whole. One
  crop per layout cluster is read, and the detector returns one box per line it
  finds inside that crop. Every detector knob (``text_threshold``, ``low_text``,
  ``link_threshold``, ``canvas_size``, ``mag_ratio``) is unreachable: Docling
  calls ``readtext`` with no arguments, so nothing above it can set them.
* **Docling's cell handling.** What the detector returned is turned into the
  page's textline cells. Reachable only by patching the library.
* **the cluster partition and the order rules.** Which crops were read at all,
  in what order, and the rule-based reading order applied to the assembled
  elements afterwards. This is reachable configuration -- a layout model spec on
  one line.

This script attributes a mis-ordered phrase to exactly one of them, and reports
the evidence. It is a diagnostic, not a fix: it runs offline, is reachable from
no route, and changes nothing about the service.

Four arms
---------

**A -- the partition.** The layout clusters, and the OCR rects the OCR stage
actually used, after the library's own dilation and dedup. The rects are recorded
as the stage asks for them, because the clusters they came from are not the ones
left on the page once the run ends -- the layout post-processing stage replaces
the layout prediction after OCR has been queued, so a partition recomputed from
the loaded page is a different partition. Then which rect, or
which rects in what order, cover the phrase. *Settles:* whether the detector is
reachable at all. Two or more crops over one phrase exonerate it by construction
-- two independent ``readtext`` calls can both be perfect while the phrase is
assembled wrongly -- so this arm runs first and decides whether the arms below
carry any weight.

**B -- the layer-by-layer replay.** For each covering crop, EasyOCR is re-run on
the byte-identical image the pipeline gave it, and its raw output is printed
beside Docling's textline cells for the same words and the exported markdown. The
first layer that differs from the one before it names the stage:

    raw reordered                     -> the detector
    raw ordered, cells reordered      -> Docling's cell handling
    both ordered, markdown reordered  -> the order rules

Under a partitioned read there is no single raw layer, since the phrase is not
one crop's output, so the walk starts at the cells and a reorder first visible
there is the partition's doing. A phrase that comes out in order at every layer
it has is reported as not reproduced at these conditions rather than as a stage:
a null result is a finding, and it moves the question downstream of the parse.

**C -- the whole-page pass.** The same page read in one page-sized crop, which
removes the partition entirely. Cross-checked against arm B: a phrase that is
wrong per-crop and right whole-page implicates the partition rather than a rule.
This arm is a discriminator only -- the whole-page mode replaces a digital PDF's
accurate text layer with recognised text, which is why the service does not use
it -- so an on-screen result for a born-digital document is reported as
inconclusive instead of as evidence.

**D -- the born-digital control.** Every document given is classified by whether
the phrase came from OCR or from the document's own text layer, and the outcomes
are combined: a control that orders correctly while the scan does not implicates
the OCR cells' geometry, since the order rules are content-agnostic; both
reordering points at the rules themselves, which no OCR setting repairs.

Usage
-----

    .venv/Scripts/python.exe scripts/localize_ocr_ordering.py SCAN.pdf
    .venv/Scripts/python.exe scripts/localize_ocr_ordering.py SCAN.pdf CONTROL.pdf
    .venv/Scripts/python.exe scripts/localize_ocr_ordering.py SCAN.pdf --marker "..." --pages 4-6

Give the scan and the born-digital control together to get arm D's combined
implication; a document path is the only thing a report needs. Each document is
converted twice -- once in the service's configured mode, once whole-page for arm
C -- so pass ``--pages`` to bound the cost on a long scan.

The probe imports the service's own ``_pdf_format_options`` rather than
re-declaring the OCR options, so the parse being diagnosed is the service's parse.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, Page
from docling.datamodel.pipeline_options import OcrMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.stages.ocr.easyocr_model import EasyOcrModel
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.page import TextCell
from numpy import array as numpy_array

from doc_etl_api.config import settings
from doc_etl_api.pipeline import _pdf_format_options

# The phrase the OCR change measured coming out as ``Hãy đây`` with ``nghe``
# appended later. A short default keeps the subsequence comparison tight; pass
# the whole sentence with ``--marker`` when the page is known.
DEFAULT_MARKER = "Hãy nghe đây"

# Verdicts for one layer's token order.
ABSENT = "absent"
ORDERED = "ordered"
REORDERED = "reordered"

# Attributions. One of the first four names the stage; the last two say why no
# stage can be named.
BY_PARTITION = "the cluster partition and the order rules that follow it"
BY_DETECTOR = "EasyOCR's detector output"
BY_CELLS = "Docling's cell handling"
BY_RULES = "the reading-order rules"
NOT_REPRODUCED = "not reproduced at these conditions"
INCONCLUSIVE = "inconclusive"

# A rect that contains a cluster's box within this many page units counts as
# that cluster's crop. The library's dedup dilates before it merges, so a
# merged rect is a little larger than the boxes it came from.
CONTAINMENT_EPSILON = 2.0


def _tokens(text: str) -> list[str]:
    """The comparable words of *text*.

    Diacritics are kept: they are the word in Vietnamese, and a comparison that
    folded them would call a mis-recognised word correctly ordered. Only
    punctuation and case are dropped, because neither moves a word.
    """
    return re.findall(r"\w+", text.lower(), re.UNICODE)


def _verdict(order: Sequence[str], marker: Sequence[str]) -> str:
    """Whether *order* carries every word of *marker*, and in the marker's order.

    A subsequence test rather than a substring one, because the phrase being
    diagnosed is rarely contiguous once the text is assembled: other words from
    the same line sit between its own.
    """
    if not marker:
        return ABSENT
    seen = set(order)
    if not all(word in seen for word in marker):
        return ABSENT
    index = 0
    for word in order:
        if word == marker[index]:
            index += 1
            if index == len(marker):
                return ORDERED
    return REORDERED


def _top_left(box: BoundingBox, page: Page) -> BoundingBox:
    """*box* in top-left coordinates, which is how clusters are reported.

    Textline cells arrive in either origin depending on whether they came from
    OCR or from a PDF's own text layer, and the two cannot be compared until
    they agree.
    """
    assert page.size is not None
    return box.to_top_left_origin(page_height=page.size.height)


def _cell_box(cell: TextCell, page: Page) -> BoundingBox:
    return _top_left(cell.rect.to_bounding_box(), page)


def _contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    """Whether *outer* holds *inner*, allowing a small tolerance."""
    return (
        outer.l - CONTAINMENT_EPSILON <= inner.l
        and outer.t - CONTAINMENT_EPSILON <= inner.t
        and outer.r + CONTAINMENT_EPSILON >= inner.r
        and outer.b + CONTAINMENT_EPSILON >= inner.b
    )


@dataclass(frozen=True)
class Cluster:
    """One layout detection."""

    index: int
    label: str
    box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class Crop:
    """One OCR input rect, and the clusters it was derived from."""

    index: int
    box: BoundingBox
    source_clusters: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class Partition:
    """One page's partition, as the OCR stage asked for it.

    Captured while the run is in progress rather than reconstructed from the
    converted page, because the two are not the same partition: the layout
    post-processing stage hands the page a rewritten set of clusters, and it runs
    after OCR has been queued. A partition rebuilt from the loaded page therefore
    describes clusters the stage never partitioned, and rects it never read.
    """

    page_no: int
    clusters: tuple[Cluster, ...]
    crops: tuple[Crop, ...]


@dataclass
class DocumentReport:
    """Everything one document's arms produced, kept for the combined verdict."""

    path: Path
    lines: list[str] = field(default_factory=list)
    pages_with_marker: list[int] = field(default_factory=list)
    attribution: str = INCONCLUSIVE
    marker_source: str = ABSENT
    full_page_verdict: str = ABSENT
    full_page_cell_verdict: str = ABSENT
    partition_verdict: str = "unknown"
    layer_verdicts: dict[str, str] = field(default_factory=dict)
    near_miss: tuple[int, str] | None = None


def _converter(mode: OcrMode | None = None) -> tuple[DocumentConverter, dict[int, Partition]]:
    """The service's converter, and the OCR partitions it will ask for.

    The options come from the service so the probe cannot drift from what is
    deployed. Only the OCR mode is overridden for arm C, and only on that arm's
    own copy of the options object.

    Three things have to be arranged on top of the service's options, because the
    evidence the arms read exists only while the parse is loaded -- and, for the
    partition, only while the OCR stage is running.

    ``generate_parsed_pages`` retains the textline cells, which are dropped once
    a document is assembled. The page backend -- which arm B renders the crop
    through -- is released when the run ends, in the pipeline's own ``finally``.
    The pipeline's ``keep_backend`` only covers an earlier mid-run cleanup, so the
    release itself is turned into a no-op on the probe's instance: skipping a
    release is a memory decision, not a result, and the returned markdown is the
    same parse either way. Without both the probe would report a partition
    computed by the fallback path -- with the backend gone, ``get_ocr_rects``
    stops asking the PDF which clusters already hold text and returns every layout
    cluster, which is not what the service does -- and no crop to replay.

    The partition is the third: it is recorded from the pipeline's own OCR model
    as that model is asked for its rects, because ``page.predictions.layout`` does
    not survive the run unchanged. The layout post-processing stage replaces the
    prediction's clusters with the post-processed ones, and it runs after OCR has
    been queued -- so a partition rebuilt from the converted page describes
    clusters the stage never partitioned and crops it never read. The returned
    mapping is that record, keyed by page number.
    """
    format_options = _pdf_format_options(settings.pdf_ocr_language_list)
    pdf = format_options[InputFormat.PDF]
    assert isinstance(pdf, PdfFormatOption)
    pdf.pipeline_options.generate_parsed_pages = True
    if mode is not None:
        pdf.pipeline_options.ocr_options.mode = mode
    converter = DocumentConverter(format_options=format_options)
    converter.initialize_pipeline(InputFormat.PDF)
    partitioned: dict[int, Partition] = {}
    for pipeline in converter.initialized_pipelines.values():
        pipeline.keep_backend = True
        pipeline._unload = lambda conv_res: conv_res
        _record_partitions(pipeline.ocr_model, partitioned)
    return converter, partitioned


def _record_partitions(model: BaseOcrModel, into: dict[int, Partition]) -> None:
    """Record each partition at the moment the OCR stage asks for it.

    ``get_ocr_rects`` is wrapped rather than called afterwards, so the rects and
    the clusters they were deduced from are read in the same call and describe the
    same partition. The clusters are snapshotted there too, because containment --
    which is how a merged rect is traced back to the detections that went into it
    -- has to be measured against the boxes the rects were actually built from.
    """
    original = model.get_ocr_rects

    def recording(page: Page) -> list[BoundingBox]:
        rects = original(page)
        clusters = tuple(_clusters(page))
        into[page.page_no] = Partition(
            page_no=page.page_no,
            clusters=clusters,
            crops=_crops_from(rects, clusters, page),
        )
        return rects

    model.get_ocr_rects = recording


def _ocr_model(mode: OcrMode) -> EasyOcrModel:
    """The OCR model the replay reads with, and arm C borrows for its rects.

    Built from the service's options, so arm C's mode flip stays off the
    pipeline's own model and one instance serves both -- a second one would be a
    second EasyOCR load. Its reader is not the object the OCR stage called: it is
    another EasyOCR from the same options, so what a replay's fidelity rests on is
    the image being the one the stage was handed. That is what capturing the
    partition at OCR time is for.
    """
    options = _pdf_format_options(settings.pdf_ocr_language_list)[InputFormat.PDF]
    assert isinstance(options, PdfFormatOption)
    ocr_options = options.pipeline_options.ocr_options
    ocr_options.mode = mode
    return EasyOcrModel(
        enabled=True,
        artifacts_path=None,
        options=ocr_options,
        accelerator_options=options.pipeline_options.accelerator_options,
    )


def _clusters(page: Page) -> list[Cluster]:
    """The layout detections for *page*, in the order the layout stage produced."""
    prediction = page.predictions.layout
    if prediction is None:
        return []
    return [
        Cluster(
            index=index,
            label=str(cluster.label),
            box=_top_left(cluster.bbox, page),
            confidence=float(cluster.confidence),
        )
        for index, cluster in enumerate(prediction.clusters)
    ]


def _crops(
    page: Page,
    model: EasyOcrModel,
    known_clusters: Sequence[Cluster],
    mode: OcrMode | None = None,
) -> list[Crop]:
    """The OCR rects for *page*, each naming the clusters it was derived from.

    This asks the model for the rects now, which is only sound for a pass whose
    partition cannot have changed since -- arm C's whole-page conversion, whose
    single rect is the page and does not depend on the clusters at all. The
    partition the service's own run used is the one ``_record_partitions``
    captured as the stage asked for it, and that is what arms A and B read.

    *mode* overrides the model's own mode for this call, so arm C's rect report
    describes the pass arm C ran rather than the default one. It is a temporary
    flip rather than a second model, because a second model is a second EasyOCR
    load and the mode is the only thing that differs.
    """
    previous = model.options.mode
    if mode is not None:
        model.options.mode = mode
    try:
        rects = model.get_ocr_rects(page)
    finally:
        model.options.mode = previous
    return list(_crops_from(rects, known_clusters, page))


def _crops_from(
    rects: Sequence[BoundingBox],
    known_clusters: Sequence[Cluster],
    page: Page,
) -> tuple[Crop, ...]:
    """*rects* as crops, each naming the clusters it was derived from.

    The library does not report which detections went into a merged rect, so that
    is recovered by containment: a merged rect holds the boxes it came from. The
    clusters have to be the ones the rects were built from, which is why the
    recorder snapshots them in the same call instead of reading the page later.
    """
    crops = []
    for index, rect in enumerate(rects):
        box = _top_left(rect, page)
        crops.append(
            Crop(
                index=index,
                box=box,
                source_clusters=tuple(
                    (cluster.index, cluster.label)
                    for cluster in known_clusters
                    if _contains(box, cluster.box)
                ),
            )
        )
    return tuple(crops)


def _marker_cells(page: Page, marker: Sequence[str]) -> list[TextCell]:
    """The page's cells that carry any word of *marker*, in Docling's own order.

    Any word rather than all of them, because a phrase split across two boxes --
    which is one of the things being diagnosed -- has no single cell carrying it.
    ``page.cells`` is empty unless the parse retained its parsed pages, which
    ``_converter`` asks for; an empty list here means the cells were released or
    the page has none, and either way there is nothing to compare.
    """
    wanted = set(marker)
    return [
        cell
        for cell in sorted(page.cells, key=lambda cell: cell.index)
        if wanted & set(_tokens(cell.text or ""))
    ]


def _union(cells: Sequence[TextCell], page: Page) -> BoundingBox | None:
    """The box enclosing *cells*, or None when there are none."""
    if not cells:
        return None
    boxes = [_cell_box(cell, page) for cell in cells]
    left = min(box.l for box in boxes)
    top = min(box.t for box in boxes)
    right = max(box.r for box in boxes)
    bottom = max(box.b for box in boxes)
    return BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT)


def _covering_crops(crops: Sequence[Crop], area: BoundingBox | None) -> list[Crop]:
    """The crops that overlap *area*, in the order the OCR stage would read them."""
    if area is None:
        return []
    return [crop for crop in crops if crop.box.intersection_area_with(area) > 0]


def _covers(cells: Sequence[TextCell], marker: Sequence[str]) -> bool:
    """Whether every word of *marker* is somewhere among *cells*.

    The union of the cells' words, not one cell's: a phrase split across two
    boxes -- one of the things being diagnosed -- is still the phrase.
    """
    words = {word for cell in cells for word in _tokens(cell.text or "")}
    return set(marker) <= words


def _marker_source(cells: Sequence[TextCell]) -> str:
    """Whether the phrase came from OCR or from the document's own text layer."""
    if not cells:
        return ABSENT
    from_ocr = [cell.from_ocr for cell in cells]
    if all(from_ocr):
        return "OCR"
    if not any(from_ocr):
        return "the document's text layer"
    return "mixed OCR and text layer"


def _replay(
    model: EasyOcrModel, page: Page, crop: Crop
) -> tuple[tuple[int, int], list[tuple[str, float]]]:
    """EasyOCR's own output for the image the pipeline handed the OCR stage.

    This is the OCR stage's own lines, in its own order: render the crop through
    the page backend at the model's scale, take it as an array, hand it to the
    reader. ``page.get_image`` would be nearly the same call, but it takes a
    different route once a crop is involved and takes a different type, and either
    difference is enough to make a matching replay meaningless: the arm would
    compare our rendering against Docling's rather than the detector against
    Docling's use of it. The reader is the probe's own (see ``_ocr_model``), so
    the crop is the part that has to be the stage's -- which is why it is taken
    from the partition captured at OCR time and not from the converted page.
    """
    backend = page._backend
    if backend is None:
        return (0, 0), []
    image = backend.get_page_image(scale=model.scale, cropbox=crop.box)
    results = model.reader.readtext(numpy_array(image))
    return image.size, [(text, float(confidence)) for _, text, confidence in results]


def _convert(
    path: Path, mode: OcrMode | None, page_range: tuple[int, int]
) -> tuple[ConversionResult, dict[int, Partition]]:
    """Convert *path*, and report the partitions the OCR stage asked for.

    The two come back together because the partitions belong to the run that
    produced them: they are a record of that conversion, not of the document.
    """
    converter, partitioned = _converter(mode)
    result = converter.convert(path, raises_on_error=False, page_range=page_range)
    return result, partitioned


def _report_arms(
    report: DocumentReport,
    result: ConversionResult,
    model: EasyOcrModel,
    marker: Sequence[str],
    partitions: dict[int, Partition],
) -> None:
    """Arms A and B for one document's default-mode conversion.

    The partition each page is reported against is the one *partitions* recorded
    while the run was in progress, so arm A describes the crops the OCR stage
    asked for and arm B replays the image those crops were. ``model`` is the
    probe's own reader, built from the service's OCR options, for that replay.
    """
    markdown_words = _tokens(result.document.export_to_markdown()) if result.document else []

    for page in result.pages:
        assert page.size is not None
        cells = _marker_cells(page, marker)
        if not cells:
            continue
        # Every word of the marker has to be somewhere among these cells for the
        # page to carry the phrase. Sharing one word is not carrying it -- and a
        # page that shares a word is remembered instead, because a phrase the
        # recogniser read differently otherwise looks identical to a phrase that
        # was never on the page.
        if not _covers(cells, marker):
            text = " ".join(cell.text or "" for cell in cells)
            report.near_miss = (page.page_no, text)
            continue
        report.pages_with_marker.append(page.page_no)

        # Which route read the phrase is a property of the page, not of the
        # partition, so it is settled before the partition is looked up: a page
        # with no recorded partition still has an answer for arm D, and the
        # summary reads it.
        source = _marker_source(cells)
        if report.marker_source == ABSENT:
            report.marker_source = source
        elif report.marker_source != source:
            # A document read by both routes has no single answer for arm D, and
            # arm C is only comparable where OCR did the reading.
            report.marker_source = "mixed OCR and text layer"

        partition = partitions.get(page.page_no)
        if partition is None:
            # The OCR stage asks every page it is handed for its rects, so a page
            # it never asked about was not read by OCR in this run: there is no
            # crop to replay, and none the cells could have come from.
            report.attribution = INCONCLUSIVE
            report.lines.append("")
            report.lines.append(f"  page {page.page_no} -- carries the marker")
            report.lines.append(
                "    arm A: the OCR stage asked for no partition of this page, so "
                "this run did not read it by OCR -- no crop to compare the cells "
                "against"
            )
            report.lines.append(f"    arm D: the marker came from {source}")
            continue

        clusters = list(partition.clusters)
        crops = list(partition.crops)
        area = _union(cells, page)
        covering = _covering_crops(crops, area)

        report.lines.append("")
        report.lines.append(f"  page {page.page_no} -- carries the marker")
        report.lines.append(
            f"    arm A: {len(clusters)} layout clusters, "
            f"{len(crops)} OCR rects after dilation and dedup, "
            "as the OCR stage asked for them"
        )
        for cluster in clusters:
            report.lines.append(
                f"      cluster [{cluster.index}] {cluster.label} "
                f"conf={cluster.confidence:.2f} box={_box(cluster.box)}"
            )
        for crop in crops:
            report.lines.append(
                f"      rect    [{crop.index}] box={_box(crop.box)} "
                f"from clusters: {_cluster_list(crop)}"
            )
        count = len(covering)
        if count == 1:
            wording = "one rect covers the marker"
        elif count > 1:
            wording = f"{count} (two or more) rects cover the marker"
        else:
            wording = "no rect covers the marker"
        listed = ", ".join(
            f"rect [{crop.index}] (clusters: {_cluster_list(crop)})" for crop in covering
        )
        report.lines.append(
            f"    arm A: {wording}"
            + (f": {listed}" if listed else "")
            + ("  <- partitioned" if count > 1 else "")
        )

        report.lines.append(f"    arm D: the marker came from {source}")
        # Every cell carrying a marker word, with its own box: a line the OCR
        # stage read as one box and Docling's cells present as two, or the
        # reverse, is the difference the layer verdicts below are about.
        for cell in cells:
            report.lines.append(
                f"      cell  [{cell.index}] from_ocr={cell.from_ocr} "
                f"conf={cell.confidence or 0:.2f} box={_box(_cell_box(cell, page))} "
                f"{_quote(cell.text or '')}"
            )

        cell_text = " ".join(cell.text or "" for cell in cells)
        cell_verdict = _verdict(_tokens(cell_text), marker)
        markdown_verdict = _verdict(markdown_words, marker)
        report.layer_verdicts = {"cells": cell_verdict, "markdown": markdown_verdict}

        if len(covering) > 1:
            report.partition_verdict = "partitioned"
            report.lines.append(
                "    arm B: the phrase is read in more than one crop, so there is no "
                "single raw layer to compare -- the replay is reported per rect as "
                "evidence only, because two crops can both read correctly."
            )
            for crop in covering:
                size, boxes = _replay(model, page, crop)
                report.lines.append(
                    f"      rect [{crop.index}] crop={size[0]}x{size[1]}px "
                    f"raw={_quote(' '.join(text for text, _ in boxes))}"
                )
            report.lines.append(f"      docling cells   {_quote(cell_text)}")
            report.lines.append(
                f"    arm B: layer verdicts cells={cell_verdict} markdown={markdown_verdict}, "
                "raw: none -- the phrase is not one crop's output"
            )
            report.attribution = _attribute_partitioned(cell_verdict, markdown_verdict)
            report.lines.append(f"    arm B: attribution -> {report.attribution}")
        elif covering:
            report.partition_verdict = "single crop"
            crop = covering[0]
            threshold = model.options.confidence_threshold
            size, boxes = _replay(model, page, crop)
            raw_text = " ".join(text for text, _ in boxes)
            raw_verdict = _verdict(_tokens(raw_text), marker)
            report.layer_verdicts = {
                "raw": raw_verdict,
                "cells": cell_verdict,
                "markdown": markdown_verdict,
            }
            report.attribution = _attribute(raw_verdict, cell_verdict, markdown_verdict)
            report.lines.append(
                f"    arm B: rect [{crop.index}] crop={size[0]}x{size[1]}px, "
                f"{len(boxes)} easyocr line(s)"
            )
            for text, confidence in boxes:
                dropped = " dropped by the confidence filter" if confidence < threshold else ""
                report.lines.append(f"      raw  conf={confidence:.2f} {_quote(text)}{dropped}")
            report.lines.append(f"      docling cells   {_quote(cell_text)}")
            report.lines.append(
                f"    arm B: layer verdicts raw={raw_verdict} cells={cell_verdict} "
                f"markdown={markdown_verdict}"
            )
            report.lines.append(f"    arm B: attribution -> {report.attribution}")
        else:
            report.partition_verdict = "no crop covers the marker"
            report.attribution = INCONCLUSIVE
            report.lines.append(
                "    arm B: no OCR rect covers the marker, so no layer can be "
                "attributed -- the cells are not where the crops are."
            )


def _attribute_partitioned(cells: str, markdown: str) -> str:
    """The stage a reorder under a partitioned read belongs to.

    The walk starts at the cells rather than at the raw layer, because the phrase
    was read in more than one crop: there is no single detector result to compare
    against, and a replay that reads every crop correctly coexists with wrongly
    assembled text. The cells are where two crops' output is joined, so a reorder
    first visible there is the partition's doing rather than the cell handling of
    any one crop -- and a reorder that appears only in the markdown is above both,
    in the rules.
    """
    if cells == ABSENT:
        return INCONCLUSIVE
    if cells == REORDERED:
        return BY_PARTITION
    if markdown == REORDERED:
        return BY_RULES
    if markdown == ABSENT:
        return INCONCLUSIVE
    return NOT_REPRODUCED


def _attribute(raw: str, cells: str, markdown: str) -> str:
    """The first layer that turns reordered names the stage.

    Absences short-circuit the walk: a layer without the phrase cannot be
    compared to the one before it, and guessing past it would name a stage on no
    evidence.
    """
    if raw == ABSENT or cells == ABSENT:
        return INCONCLUSIVE
    if raw == REORDERED:
        return BY_DETECTOR
    if cells == REORDERED:
        return BY_CELLS
    if markdown == REORDERED:
        return BY_RULES
    if markdown == ABSENT:
        return INCONCLUSIVE
    return NOT_REPRODUCED


def _full_page(
    report: DocumentReport,
    path: Path,
    model: EasyOcrModel,
    marker: Sequence[str],
    page_range: tuple[int, int],
) -> ConversionResult:
    """Arm C: the same pages read in one page-sized crop.

    The second conversion is what removes the partition, so this arm costs a
    second OCR pass over the range. Its verdict is only comparable when the
    document is read by OCR at all: whole-page mode replaces a text layer, so a
    born-digital page is reported inconclusive rather than read as evidence.
    """
    result = _convert(path, OcrMode.FULL_PAGE, page_range)[0]
    report.lines.append("")
    report.lines.append(f"  arm C: whole-page pass (mode={OcrMode.FULL_PAGE.value})")
    if report.marker_source != "OCR":
        report.full_page_verdict = INCONCLUSIVE
        report.lines.append(
            "    inconclusive: the marker does not come from OCR here, and this mode "
            "re-types a text layer, so the two passes are not comparable."
        )
        return result

    markdown_verdict = (
        _verdict(_tokens(result.document.export_to_markdown()), marker)
        if result.document
        else ABSENT
    )
    for page in result.pages:
        if page.page_no not in report.pages_with_marker:
            continue
        crops = _crops(page, model, _clusters(page), mode=OcrMode.FULL_PAGE)
        report.lines.append(f"    page {page.page_no}: {len(crops)} rect(s)")
        for crop in crops:
            report.lines.append(f"      rect [{crop.index}] box={_box(crop.box)}")
        cells = _marker_cells(page, marker)
        cell_text = " ".join(cell.text or "" for cell in cells)
        report.full_page_verdict = markdown_verdict
        report.full_page_cell_verdict = _verdict(_tokens(cell_text), marker)
        report.lines.append(f"    arm C: assembled text {_quote(cell_text)}")
        report.lines.append(
            f"    arm C: marker verdict cells={report.full_page_cell_verdict} "
            f"markdown={markdown_verdict}"
        )
    report.lines.append(f"    arm C: cross-check -> {_cross_check(report)}")
    return result


def _release(results: Sequence[ConversionResult]) -> None:
    """Let go of the backends the probe asked the pipeline to keep.

    The arms read the live backend, so the pipeline's own release was turned into
    a no-op for the run. This is that release, done once every report is printed:
    without it the parser's finalizer reports the documents as left open, which
    reads as a leak in a probe that only wanted a crop.
    """
    for result in results:
        for page in result.pages:
            if page._backend is not None:
                page._backend.unload()
        if result.input._backend is not None:
            result.input._backend.unload()


def _cross_check(report: DocumentReport) -> str:
    """Whether arm C's outcome agrees with arm B's attribution."""
    verdict = report.full_page_verdict
    if verdict == INCONCLUSIVE:
        return "inconclusive, see above"
    if report.attribution == NOT_REPRODUCED:
        return "nothing to explain: neither pass reorders the phrase"
    if verdict == ABSENT:
        return "inconclusive: the phrase is absent from the whole-page pass"
    if report.attribution == BY_RULES:
        if verdict == REORDERED:
            return "agrees: the rules reorder the phrase with one crop as well as many"
        return "conflicts: the rules were implicated, yet one crop reads in order"
    if verdict == ORDERED:
        if report.attribution == BY_CELLS:
            return (
                "agrees: one crop reads the phrase in order, so the cells are built "
                "from the crop that was read -- the cell layer still holds the reorder"
            )
        if report.attribution == BY_DETECTOR:
            return (
                "agrees: one crop reads the phrase in order, so the detector's own "
                "output depends on the crop it is given"
            )
        return "agrees: removing the per-crop partition reads the phrase in order"
    reason = f"agrees: the phrase stays reordered with one crop, as {report.attribution} predicts"
    if report.full_page_cell_verdict == ORDERED:
        reason += (
            " -- and the one-crop cells are in order, so what reorders it sits above "
            "the cells, in the order rules"
        )
    return reason


def _box(box: BoundingBox) -> str:
    return f"(l={box.l:.1f} t={box.t:.1f} r={box.r:.1f} b={box.b:.1f})"


def _cluster_list(crop: Crop) -> str:
    """The clusters a rect was derived from, by index and label.

    More than one means the library's dilation and dedup merged them into a
    single crop, which is the case worth seeing rather than inferring.
    """
    if not crop.source_clusters:
        return "none"
    return ", ".join(f"[{index}] {label}" for index, label in crop.source_clusters)


def _quote(text: str) -> str:
    collapsed = " ".join(text.split())
    return f'"{collapsed}"'


def _parse_pages(value: str) -> tuple[int, int]:
    """``3`` or ``3-7`` as the 1-based inclusive range Docling takes."""
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(f"not a page or page range: {value!r}")
    first = int(match.group(1))
    last = int(match.group(2)) if match.group(2) else first
    if first < 1 or last < first:
        raise argparse.ArgumentTypeError(f"not a usable page range: {value!r}")
    return (first, last)


def _summarise(reports: Sequence[DocumentReport]) -> None:
    """Arm D: what each document's outcome implies when they are read together."""
    print("")
    print("=" * 78)
    print("arm D -- combined outcome")
    for report in reports:
        print(
            f"  {report.path.name}: marker from {report.marker_source}, "
            f"attribution={report.attribution}, whole-page={report.full_page_verdict}"
        )
    scans = [r for r in reports if r.marker_source == "OCR"]
    controls = [r for r in reports if r.marker_source == "the document's text layer"]
    print("")
    if not scans:
        print(
            "  No document's marker came from OCR, so no OCR stage is implicated by "
            "this run. If the indexed text was wrong anyway, the defect is downstream "
            "of the parse -- in chunking or the markdown export -- which is a different "
            "problem from the one this probe localizes."
        )
    for scan in scans:
        if scan.attribution == NOT_REPRODUCED:
            print(
                f"  {scan.path.name}: the phrase is ordered at every layer on the scan, "
                "so the defect did not reproduce at these conditions."
            )
            continue
        if not controls:
            print(
                f"  {scan.path.name}: attributed to {scan.attribution}. Add a "
                "born-digital control with the same phrase to separate the order rules "
                "from the OCR cells' geometry."
            )
            continue
        for control in controls:
            if control.layer_verdicts.get("markdown") == ORDERED and scan.attribution in (
                BY_DETECTOR,
                BY_CELLS,
                BY_PARTITION,
            ):
                print(
                    f"  {scan.path.name} reorders where {control.path.name} does not. The "
                    "order rules are content-agnostic and read the control correctly, so "
                    "the OCR cells' geometry is implicated rather than a rule."
                )
            elif control.layer_verdicts.get("markdown") == REORDERED:
                print(
                    f"  Both {scan.path.name} and {control.path.name} reorder the phrase. "
                    "The rules themselves are wrong for this layout -- no OCR setting "
                    "repairs a rule, so the fix is above the recogniser."
                )
            else:
                print(
                    f"  {scan.path.name} and {control.path.name} do not form a decisive "
                    f"pair: scan={scan.attribution}, control markdown="
                    f"{control.layer_verdicts.get('markdown')}."
                )


def main(argv: Sequence[str] | None = None) -> int:
    # The subject of the report is Vietnamese text, so the report has to survive
    # the console: the platform default on Windows is a codepage that turns every
    # diacritic in the evidence into a replacement character.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="localize_ocr_ordering.py",
        description=(
            "Attribute out-of-order text in a scanned document to the OCR detector, "
            "Docling's cell handling, or the cluster partition and order rules."
        ),
    )
    parser.add_argument(
        "documents",
        nargs="+",
        type=Path,
        metavar="DOCUMENT",
        help="a document to diagnose; give a born-digital one as the control",
    )
    parser.add_argument(
        "--marker",
        default=DEFAULT_MARKER,
        help=f"the phrase to follow through each layer (default: {DEFAULT_MARKER!r})",
    )
    parser.add_argument(
        "--pages",
        type=_parse_pages,
        default=(1, sys.maxsize),
        metavar="A-B",
        help="only convert these pages, which bounds the cost of both passes",
    )
    args = parser.parse_args(argv)

    missing = [path for path in args.documents if not path.exists()]
    if missing:
        for path in missing:
            print(f"no such document: {path}", file=sys.stderr)
        return 2

    marker = _tokens(args.marker)
    if not marker:
        print(f"the marker names no words: {args.marker!r}", file=sys.stderr)
        return 2

    print("=" * 78)
    print(f"localizing OCR text order -- marker {_quote(args.marker)}")
    service_options = _pdf_format_options(settings.pdf_ocr_language_list)[InputFormat.PDF]
    assert isinstance(service_options, PdfFormatOption)
    ocr_options = service_options.pipeline_options.ocr_options
    print(
        f"OCR engine: {type(ocr_options).__name__} from the service's "
        f"_pdf_format_options, langs={list(ocr_options.lang)}, "
        f"mode={ocr_options.mode.value} (page range {args.pages[0]}-{args.pages[1]})"
    )

    model = _ocr_model(OcrMode.DEFAULT)
    reports: list[DocumentReport] = []
    results: list[ConversionResult] = []
    for path in args.documents:
        report = DocumentReport(path=path)
        print("")
        print("-" * 78)
        print(f"document: {path}")

        default_result, partitions = _convert(path, None, args.pages)
        if default_result.document is None:
            report.lines.append("  the conversion produced no document")
            report.attribution = INCONCLUSIVE
        else:
            _report_arms(report, default_result, model, marker, partitions)
            if not report.pages_with_marker:
                report.lines.append("  no page carries the marker: nothing to localize")
                if report.near_miss is not None:
                    page_no, text = report.near_miss
                    report.lines.append(
                        f"  page {page_no} comes closest, reading {_quote(text)} -- pass "
                        "--marker with the words recognised there if that is the sentence"
                    )
                report.attribution = INCONCLUSIVE

        # Arm C converts again, and appends to the same report, so it is run
        # before the report is printed rather than after it.
        results.append(default_result)
        if report.pages_with_marker:
            results.append(_full_page(report, path, model, marker, args.pages))
        for line in report.lines:
            print(line)
        reports.append(report)

    _summarise(reports)
    _release(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
