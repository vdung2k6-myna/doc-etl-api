from pathlib import Path

import pytest
from pydantic import ValidationError

from doc_etl_api.config import (
    MAX_COLLECTIONS_PER_REQUEST,
    Settings,
    VectorStoreBackend,
    validate_collection_name,
    validate_collections,
)


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
    assert s.knowledge_corpus_collections == ""
    assert s.knowledge_corpus_collection_list == []


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


# --- Collection names -------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("csharp", "csharp"),
        ("csharp-12", "csharp-12"),
        ("dotnet_8", "dotnet_8"),
        ("a1-b2_c3", "a1-b2_c3"),
        ("9", "9"),
        # Surrounding whitespace is trimmed rather than rejected: a form field
        # pads a value without the caller meaning anything by it.
        ("  csharp  ", "csharp"),
        ("\tcsharp\n", "csharp"),
    ],
)
def test_accepted_collection_names(value, expected):
    assert validate_collection_name(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "CSharp",  # uppercase
        "café",  # diacritics
        "my collection",  # embedded space
        "c" * 65,  # over-length
        "c/sharp",  # separator outside the grammar
        "c.sharp",
        "c:sharp",
        "c#sharp",
        "c sharp",
        "-csharp",  # a separator with nothing on one side of it
        "csharp-",
        "c--sharp",  # a doubled separator
        "",
        "   ",  # whitespace only trims to nothing
    ],
)
def test_rejected_collection_names(value):
    """A name outside the grammar is refused, and the refusal names it.

    Normalizing instead would store a name the caller never wrote and cannot
    guess, which is why nothing here is silently rewritten.
    """
    with pytest.raises(ValueError) as exc_info:
        validate_collection_name(value)

    assert repr(value) in str(exc_info.value)


def test_a_repeated_collection_name_is_recorded_once():
    assert validate_collections(["csharp", "dotnet", "csharp"]) == ["csharp", "dotnet"]


def test_collections_at_the_cap_are_accepted():
    names = [f"collection-{index}" for index in range(MAX_COLLECTIONS_PER_REQUEST)]

    assert validate_collections(names) == names


def test_collections_above_the_cap_are_rejected():
    names = [f"collection-{index}" for index in range(MAX_COLLECTIONS_PER_REQUEST + 1)]

    with pytest.raises(ValueError) as exc_info:
        validate_collections(names)

    assert "Too many collections" in str(exc_info.value)
    assert str(MAX_COLLECTIONS_PER_REQUEST) in str(exc_info.value)


def test_a_repeated_name_does_not_count_against_the_cap():
    """The cap bounds the collections stored, and a repeat is one of them."""
    names = ["csharp"] * (MAX_COLLECTIONS_PER_REQUEST * 2)

    assert validate_collections(names) == ["csharp"]


# --- Corpus collections -----------------------------------------------------


def test_corpus_collections_are_unset_by_default():
    assert Settings().knowledge_corpus_collection_list == []


def test_corpus_collections_from_env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KNOWLEDGE_CORPUS_COLLECTIONS", "csharp, dotnet-8 ,")
        s = Settings()

    assert s.knowledge_corpus_collection_list == ["csharp", "dotnet-8"]


@pytest.mark.parametrize("blank", ["", "   ", ",", " , "])
def test_blank_corpus_collections_are_empty(blank):
    """Empty, whitespace-only and separator-only values all mean "no collections"."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KNOWLEDGE_CORPUS_COLLECTIONS", blank)
        s = Settings()

    assert s.knowledge_corpus_collection_list == []


def test_a_malformed_corpus_collection_stops_the_settings_being_built():
    """A configuration typo must fail at startup, not empty out the index.

    Rejected while the settings are built, so the service does not start at all
    rather than starting and failing every corpus source one by one.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KNOWLEDGE_CORPUS_COLLECTIONS", "csharp, C# 12")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

    assert "C# 12" in str(exc_info.value)
