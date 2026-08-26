import logging
import tempfile
import time
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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from doc_etl_api.config import Settings, VectorStoreBackend, settings

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


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

    def convert_url(self, url: str, timeout: int = 30) -> tuple[str, str]:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        final_url = response.url

        suffix = ".html"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(response.content)
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
        LlamaSettings.embed_model = self._embedding_model
        LlamaSettings.node_parser = SentenceSplitter(
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
        )
        self._index = VectorStoreIndex(nodes=[])

    @property
    def converter(self) -> DoclingConverter:
        return self._converter

    @property
    def index(self) -> VectorStoreIndex:
        return self._index

    def ingest_file(
        self,
        source_id: str,
        file: BinaryIO,
        filename: str,
        mime_type: str | None = None,
    ) -> tuple[dict, dict[str, float]]:
        timings: dict[str, float] = {}

        parse_start = time.perf_counter()
        markdown = self._converter.convert_file(file, filename)
        timings["parse_ms"] = _elapsed_ms(parse_start)

        document = LlamaDocument(
            text=markdown,
            metadata={
                "source_id": source_id,
                "source_type": "file",
                "source_name": filename,
                "mime_type": mime_type,
            },
        )

        chunk_start = time.perf_counter()
        nodes = LlamaSettings.node_parser.get_nodes_from_documents([document])
        timings["chunk_ms"] = _elapsed_ms(chunk_start)

        embed_start = time.perf_counter()
        for node in nodes:
            node.embedding = self._embedding_model.get_text_embedding(node.get_content())
        timings["embed_ms"] = _elapsed_ms(embed_start)

        index_start = time.perf_counter()
        self._index.insert_nodes(nodes)
        timings["index_ms"] = _elapsed_ms(index_start)

        result = {
            "source_id": source_id,
            "source_type": "file",
            "name": filename,
            "mime_type": mime_type,
            "status": "indexed",
        }
        logger.info(
            "Indexed file source=%s filename=%s timings=%s",
            source_id,
            filename,
            timings,
        )
        return result, timings

    def ingest_url(
        self, source_id: str, url: str, timeout: int | None = None
    ) -> tuple[dict, dict[str, float]]:
        timings: dict[str, float] = {}
        timeout = timeout or self._settings.url_fetch_timeout_seconds

        parse_start = time.perf_counter()
        markdown, final_url = self._converter.convert_url(url, timeout=timeout)
        timings["parse_ms"] = _elapsed_ms(parse_start)

        document = LlamaDocument(
            text=markdown,
            metadata={
                "source_id": source_id,
                "source_type": "url",
                "source_name": url,
                "final_url": final_url,
            },
        )

        chunk_start = time.perf_counter()
        nodes = LlamaSettings.node_parser.get_nodes_from_documents([document])
        timings["chunk_ms"] = _elapsed_ms(chunk_start)

        embed_start = time.perf_counter()
        for node in nodes:
            node.embedding = self._embedding_model.get_text_embedding(node.get_content())
        timings["embed_ms"] = _elapsed_ms(embed_start)

        index_start = time.perf_counter()
        self._index.insert_nodes(nodes)
        timings["index_ms"] = _elapsed_ms(index_start)

        result = {
            "source_id": source_id,
            "source_type": "url",
            "name": url,
            "final_url": final_url,
            "status": "indexed",
        }
        logger.info(
            "Indexed URL source=%s url=%s timings=%s",
            source_id,
            url,
            timings,
        )
        return result, timings

    def search(self, query: str, top_k: int) -> list[dict]:
        retriever = self._index.as_retriever(similarity_top_k=top_k)
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
