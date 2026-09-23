import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

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


# --- OCR languages ----------------------------------------------------------


def test_ocr_language_defaults_to_vietnamese():
    """Unconfigured must still read the service's own content correctly.

    A default of "none configured" would leave the defect this setting exists to
    fix in place until an operator opted in, so the default names the language
    the content is written in.
    """
    assert Settings().pdf_ocr_language_list == ["vi"]


def test_ocr_languages_from_env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDF_OCR_LANGUAGES", "vi, en ,")
        s = Settings()

    assert s.pdf_ocr_language_list == ["vi", "en"]


@pytest.mark.parametrize("blank", ["", "   ", ",", " , "])
def test_an_ocr_language_setting_naming_no_language_is_rejected(blank):
    """An empty language list does not mean "no OCR".

    It means the OCR engine's own default, which is a set of European languages
    -- so accepting it would reproduce the defect while looking deliberate.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDF_OCR_LANGUAGES", blank)
        with pytest.raises(ValidationError) as exc_info:
            Settings()

    message = str(exc_info.value)
    assert "names no language" in message
    if blank.strip():
        assert repr(blank) in message


def test_a_malformed_ocr_language_stops_the_settings_being_built():
    """A typo must fail at startup, not the first document that needs OCR."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDF_OCR_LANGUAGES", "vi, vie tnam")
        with pytest.raises(ValidationError) as exc_info:
            Settings()

    assert "vie tnam" in str(exc_info.value)


# --- Corpus file collections ------------------------------------------------


def test_corpus_file_collections_are_unset_by_default():
    assert Settings().knowledge_corpus_file_collections == {}


def test_corpus_file_collections_from_env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(
            "KNOWLEDGE_CORPUS_FILE_COLLECTIONS",
            json.dumps({"101-Truyen-Cuoi-Dan-Gian-Viet-Nam.txt": ["truyen-cuoi"]}),
        )
        s = Settings()

    assert s.knowledge_corpus_file_collections == {
        "101-Truyen-Cuoi-Dan-Gian-Viet-Nam.txt": ["truyen-cuoi"]
    }


def test_corpus_file_collections_are_normalized_like_an_uploads():
    """The tag stored for a corpus file is the tag a filter reads.

    Padded and repeated names are normalized here exactly as an upload's are, so
    a corpus file and an upload naming the same collection meet as one
    collection rather than as two spellings of it.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(
            "KNOWLEDGE_CORPUS_FILE_COLLECTIONS",
            json.dumps({"alpha.txt": [" csharp ", "csharp", "dotnet"]}),
        )
        s = Settings()

    assert s.knowledge_corpus_file_collections == {"alpha.txt": ["csharp", "dotnet"]}


@pytest.mark.parametrize(
    "filename",
    [
        "sub/alpha.txt",  # a path, on any platform
        "sub\\alpha.txt",  # a path on Windows, refused on every platform
        "https://example.com/page",  # a URL is not a corpus file
    ],
)
def test_a_corpus_file_entry_naming_a_path_is_rejected(filename):
    """A path can never match, so it fails where the setting is read.

    The corpus directory is scanned without recursing and an entry matches a
    file by its filename alone. Refusing it here names the setting; letting it
    through would tag nothing and report only that a file was missing.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KNOWLEDGE_CORPUS_FILE_COLLECTIONS", json.dumps({filename: ["csharp"]}))
        with pytest.raises(ValidationError) as exc_info:
            Settings()

    assert repr(filename) in str(exc_info.value)


def test_a_corpus_file_entry_naming_no_collection_is_rejected():
    """Tagging nothing is the one thing an entry cannot be for.

    Every filtered search is scoped by collection, so a source in no collection
    cannot be reached by one. An entry naming no collection is therefore read as
    a mistake rather than as a way to say "no collections" -- that is written by
    configuring no entry for the file and no corpus collections.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KNOWLEDGE_CORPUS_FILE_COLLECTIONS", json.dumps({"alpha.txt": []}))
        with pytest.raises(ValidationError) as exc_info:
            Settings()

    assert "alpha.txt" in str(exc_info.value)


@pytest.mark.parametrize("filename", ["", "   "])
def test_a_corpus_file_entry_needs_a_filename(filename):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KNOWLEDGE_CORPUS_FILE_COLLECTIONS", json.dumps({filename: ["csharp"]}))
        with pytest.raises(ValidationError) as exc_info:
            Settings()

    assert "filename is empty" in str(exc_info.value)


def test_a_malformed_collection_in_a_corpus_file_entry_stops_the_settings_being_built():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KNOWLEDGE_CORPUS_FILE_COLLECTIONS", json.dumps({"alpha.txt": ["C# 12"]}))
        with pytest.raises(ValidationError) as exc_info:
            Settings()

    # The entry is named as well as the collection, so the message points at the
    # part of the configuration to change.
    assert "alpha.txt" in str(exc_info.value)
    assert "C# 12" in str(exc_info.value)


def test_an_unparseable_corpus_file_collections_value_stops_the_settings_being_built():
    """The value is read as JSON, and JSON that does not parse is refused here.

    Refused as the settings are built rather than at the bootstrap, where it
    would instead fail every corpus source one at a time and leave the index
    empty -- the same reasoning as the collection validator above.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KNOWLEDGE_CORPUS_FILE_COLLECTIONS", "{csharp")
        with pytest.raises(SettingsError):
            Settings()
