from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from docling.datamodel.base_models import ConversionStatus

from doc_etl_api.config import Settings
from doc_etl_api.pipeline import DoclingConverter, IndexPipeline


@pytest.fixture
def settings():
    return Settings(
        vector_store_backend="simple",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=128,
        chunk_overlap=10,
        default_top_k=2,
    )


def test_docling_converter_supported_files():
    converter = DoclingConverter()
    assert converter.is_supported_file("report.pdf")
    assert converter.is_supported_file("notes.docx")
    assert converter.is_supported_file("page.html")
    assert not converter.is_supported_file("archive.zip")


def test_docling_converter_file(settings, tmp_path):
    converter = DoclingConverter()
    sample = tmp_path / "hello.txt"
    sample.write_text("Hello, world!")

    with patch.object(
        converter._converter,
        "convert",
        return_value=MagicMock(
            status=ConversionStatus.SUCCESS,
            document=MagicMock(export_to_markdown=lambda: "# Hello\n\nHello, world!"),
        ),
    ):
        with sample.open("rb") as f:
            result = converter.convert_file(f, "hello.txt")

    assert "Hello, world!" in result


def test_docling_converter_url(settings, tmp_path):
    converter = DoclingConverter()
    html = b"<html><body><h1>Title</h1><p>Body text.</p></body></html>"

    with patch("doc_etl_api.pipeline.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            url="https://example.com/page",
            content=html,
            raise_for_status=MagicMock(),
        )
        with patch.object(
            converter._converter,
            "convert",
            return_value=MagicMock(
                status=ConversionStatus.SUCCESS,
                document=MagicMock(export_to_markdown=lambda: "# Title\n\nBody text."),
            ),
        ):
            markdown, final_url = converter.convert_url("https://example.com/page")

    assert "# Title" in markdown
    assert final_url == "https://example.com/page"


def test_index_pipeline_ingest_file(settings):
    pipeline = IndexPipeline(settings)
    source_id = "source-file-1"

    with patch.object(
        pipeline.converter,
        "convert_file",
        return_value="# Doc\n\nThis is the content.",
    ):
        with patch.object(pipeline._index, "insert_nodes") as mock_insert:
            result, timings = pipeline.ingest_file(
                source_id=source_id,
                file=BytesIO(b"content"),
                filename="doc.txt",
                mime_type="text/plain",
            )

    assert result["source_id"] == source_id
    assert result["source_type"] == "file"
    assert result["name"] == "doc.txt"
    assert result["mime_type"] == "text/plain"
    assert result["status"] == "indexed"
    assert "parse_ms" in timings
    assert "chunk_ms" in timings
    assert "embed_ms" in timings
    assert "index_ms" in timings
    mock_insert.assert_called_once()


def test_index_pipeline_ingest_url(settings):
    pipeline = IndexPipeline(settings)
    source_id = "source-url-1"

    with patch.object(
        pipeline.converter,
        "convert_url",
        return_value=("# Page\n\nContent from URL.", "https://example.com/final"),
    ):
        with patch.object(pipeline._index, "insert_nodes") as mock_insert:
            result, timings = pipeline.ingest_url(
                source_id=source_id,
                url="https://example.com/page",
            )

    assert result["source_id"] == source_id
    assert result["source_type"] == "url"
    assert result["name"] == "https://example.com/page"
    assert result["final_url"] == "https://example.com/final"
    assert result["status"] == "indexed"
    assert "parse_ms" in timings
    assert "chunk_ms" in timings
    assert "embed_ms" in timings
    assert "index_ms" in timings
    mock_insert.assert_called_once()
