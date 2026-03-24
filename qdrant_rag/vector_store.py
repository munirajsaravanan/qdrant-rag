"""Qdrant vector-store wrapper."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from qdrant_rag.document_loader import Document


class VectorStore:
    """Thin wrapper around :class:`qdrant_client.QdrantClient`.

    Parameters
    ----------
    url:
        Full URL of the Qdrant instance (e.g. ``http://localhost:6333``).
    api_key:
        Optional API key for Qdrant Cloud.
    collection_name:
        Name of the collection to use (created automatically if absent).
    vector_dimension:
        Dimensionality of the embedding vectors.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "rag_documents",
        vector_dimension: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self._client = QdrantClient(url=url, api_key=api_key)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.vector_dimension,
                    distance=rest.Distance.COSINE,
                ),
            )

    def delete_collection(self) -> None:
        """Delete the collection (and all its data)."""
        self._client.delete_collection(self.collection_name)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def upsert(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Insert or update *documents* with their precomputed *embeddings*."""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents ({len(documents)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        points = [
            rest.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": doc.text, **doc.metadata},
            )
            for doc, embedding in zip(documents, embeddings)
        ]

        self._client.upsert(collection_name=self.collection_name, points=points)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most similar documents to *query_vector*.

        Returns a list of dicts with keys ``text``, ``score``, and
        ``metadata`` (a dict of the remaining payload fields).
        """
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []
        for hit in results:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            hits.append(
                {
                    "text": text,
                    "score": hit.score,
                    "metadata": payload,
                }
            )
        return hits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of vectors stored in the collection."""
        result = self._client.count(collection_name=self.collection_name)
        return result.count
