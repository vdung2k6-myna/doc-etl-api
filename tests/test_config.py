import pytest

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
