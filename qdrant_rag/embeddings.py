"""Embedding service – wraps sentence-transformers or the OpenAI API."""

from __future__ import annotations

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]


class EmbeddingService:
    """Generate dense vector embeddings for a list of text strings.

    Parameters
    ----------
    model_name:
        A `sentence-transformers` model name (e.g. ``"all-MiniLM-L6-v2"``)
        **or** the string ``"openai"`` to use the OpenAI embedding API.
    openai_api_key:
        Required when *model_name* is ``"openai"``.
    openai_embedding_model:
        OpenAI embedding model identifier (default: ``text-embedding-3-small``).
    """

    OPENAI_SENTINEL = "openai"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        openai_api_key: str | None = None,
        openai_embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.model_name = model_name
        self._openai_api_key = openai_api_key
        self._openai_embedding_model = openai_embedding_model
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors, one per input text."""
        if not texts:
            return []
        if self.model_name == self.OPENAI_SENTINEL:
            return self._embed_openai(texts)
        return self._embed_sentence_transformers(texts)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the configured model."""
        if self.model_name == self.OPENAI_SENTINEL:
            # text-embedding-3-small → 1536, text-embedding-3-large → 3072
            if "large" in self._openai_embedding_model:
                return 3072
            return 1536
        model = self._get_st_model()
        return model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_st_model(self):
        if self._model is None:
            if SentenceTransformer is None:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        model = self._get_st_model()
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        if not self._openai_api_key:
            raise ValueError(
                "openai_api_key must be set when using the OpenAI embedding backend."
            )
        if OpenAI is None:  # pragma: no cover
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install it with: pip install openai"
            )

        client = OpenAI(api_key=self._openai_api_key)
        response = client.embeddings.create(
            model=self._openai_embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
