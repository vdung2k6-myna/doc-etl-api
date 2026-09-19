from pathlib import Path

import pytest
from pydantic import ValidationError

from doc_etl_api.config import Settings, VectorStoreBackend


def test_default_settings():
    s = Settings()
    assert s.vector_store_backend == VectorStoreBackend.SIMPLE
    assert s.default_top_k == 5


def test_supported_backend_from_env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VECTOR_STORE_BACKEND", "simple")
        s = Settings()
        assert s.vector_store_backend == VectorStoreBackend.SIMPLE


def test_unknown_backend_raises():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VECTOR_STORE_BACKEND", "qdrant")
        with pytest.raises(ValueError) as exc_info:
            Settings()
    assert "Unsupported vector store backend" in str(exc_info.value)
    assert "simple" in str(exc_info.value)


def test_knowledge_corpus_defaults_are_empty():
    s = Settings()
    assert s.knowledge_corpus_dir == ""
    assert s.knowledge_corpus_urls == ""
    assert s.knowledge_corpus_path is None
    assert s.knowledge_corpus_url_list == []


def test_knowledge_corpus_from_env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KNOWLEDGE_CORPUS_DIR", "knowledge")
        mp.setenv(
            "KNOWLEDGE_CORPUS_URLS",
            "https://example.com/a, https://example.com/b ,",
        )
        s = Settings()

    assert s.knowledge_corpus_path == Path("knowledge")
    assert s.knowledge_corpus_url_list == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_user_agent_defaults_to_the_service_identity():
    assert Settings().resolved_user_agent == "doc-etl-api/0.1.0"


def test_user_agent_from_env_is_used_verbatim():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("USER_AGENT", "doc-etl-api/0.1.0 (contact: ops@example.com)")
        s = Settings()

    assert s.resolved_user_agent == "doc-etl-api/0.1.0 (contact: ops@example.com)"


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_user_agent_falls_back_to_the_default(blank):
    """A blank value must not become an empty header, which such hosts also refuse."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("USER_AGENT", blank)
        s = Settings()

    assert s.resolved_user_agent == "doc-etl-api/0.1.0"


def test_log_level_defaults_to_info():
    assert Settings().log_level == "INFO"


def test_log_level_from_env_is_normalized():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOG_LEVEL", "debug")
        s = Settings()

    assert s.log_level == "DEBUG"


def test_unknown_log_level_is_rejected():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOG_LEVEL", "chatty")
        with pytest.raises(ValidationError):
            Settings()
