"""Core RAG pipeline: retrieval + generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

from qdrant_rag.config import Config
from qdrant_rag.document_loader import Document, DocumentLoader
from qdrant_rag.embeddings import EmbeddingService
from qdrant_rag.vector_store import VectorStore


@dataclass
class RAGResult:
    """The answer produced by the RAG pipeline together with its sources."""

    answer: str
    sources: list[dict[str, Any]]
    query: str


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Parameters
    ----------
    config:
        A :class:`~qdrant_rag.config.Config` instance.  If *None*, a
        default config is constructed from environment variables.
    """

    _SYSTEM_PROMPT = (
        "You are a helpful assistant. "
        "Answer the user's question using ONLY the context passages provided below. "
        "If the context does not contain enough information to answer the question, "
        "say so clearly.\n\n"
        "Context:\n{context}"
    )

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._embedding_service = EmbeddingService(
            model_name=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key,
        )
        self._vector_store = VectorStore(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            collection_name=self.config.collection_name,
            vector_dimension=self._embedding_service.dimension,
        )
        self._document_loader = DocumentLoader(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(self, path: str) -> int:
        """Ingest a single file and return the number of chunks stored."""
        return self._ingest_documents(self._document_loader.load_file(path))

    def ingest_directory(self, directory: str) -> int:
        """Ingest all supported files under *directory* recursively."""
        return self._ingest_documents(self._document_loader.load_directory(directory))

    def _ingest_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        self._vector_store.ensure_collection()
        texts = [doc.text for doc in documents]
        embeddings = self._embedding_service.embed(texts)
        self._vector_store.upsert(documents, embeddings)
        return len(documents)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResult:
        """Answer *question* using retrieved context from Qdrant."""
        query_embedding = self._embedding_service.embed([question])[0]
        sources = self._vector_store.search(
            query_vector=query_embedding,
            top_k=self.config.top_k,
        )
        answer = self._generate(question, sources)
        return RAGResult(answer=answer, sources=sources, query=question)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, question: str, sources: list[dict[str, Any]]) -> str:
        """Call the LLM with the retrieved context and return the answer."""
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}] {s['text']}" for i, s in enumerate(sources)
        )
        system_message = self._SYSTEM_PROMPT.format(context=context)

        if OpenAI is None:  # pragma: no cover
            raise ImportError(
                "openai is required for answer generation. "
                "Install it with: pip install openai"
            )

        if not self.config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set to use the generation step. "
                "Set it in your .env file or as an environment variable."
            )

        client = OpenAI(api_key=self.config.openai_api_key)
        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""
