"""Shared test doubles.

`IndexPipeline` reads its chunk-sizing bound and its token counter off the
embedding model it is constructed with, so a double standing in for the real
model has to report both rather than inherit a stand-in for them.
"""

from types import SimpleNamespace

from llama_index.core.embeddings import MockEmbedding
from transformers import AutoTokenizer

# The embedder the service is configured with, so the double counts tokens in the
# same vocabulary and is bounded by the same window as the model it replaces.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MAX_TOKENS = 256
EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


class StubEmbedding(MockEmbedding):
    """A deterministic embedder carrying a real model's limit and tokenizer."""

    max_length: int = EMBEDDING_MAX_TOKENS

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = SimpleNamespace(tokenizer=EMBEDDING_TOKENIZER)
