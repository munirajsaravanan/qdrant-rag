# qdrant-rag

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline powered by [Qdrant](https://qdrant.tech/).

```
Document files ──► DocumentLoader ──► EmbeddingService ──► VectorStore (Qdrant)
                                                                    │
User question ──► EmbeddingService ──► VectorStore.search ──► OpenAI LLM ──► Answer
```

## Features

* Ingest plain-text (`.txt`, `.md`, `.rst`) and PDF files with configurable chunking
* Local sentence-transformers embeddings **or** OpenAI embeddings
* Qdrant for fast vector similarity search
* OpenAI (`gpt-4o-mini` or any model) for answer generation
* Simple CLI tools (`qdrant-ingest` / `qdrant-query`)
* Fully unit-tested with no live services required

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | *(empty)* | Qdrant Cloud API key (optional) |
| `OPENAI_API_KEY` | – | Required for answer generation |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model used for generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name, or `openai` |
| `QDRANT_COLLECTION` | `rag_documents` | Qdrant collection name |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap characters between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve per query |

### 4. Ingest documents

```bash
# Single file
python ingest.py path/to/document.txt

# Directory (recursive)
python ingest.py path/to/docs/

# Override collection or chunk settings on the fly
python ingest.py docs/ --collection my_col --chunk-size 1000 --chunk-overlap 100
```

### 5. Query

```bash
# With LLM answer generation
python query.py "What is the main topic of the documents?"

# Skip generation – just retrieve context chunks (no OpenAI key needed)
python query.py --no-generate "What is the main topic?"

# JSON output
python query.py --json "Summarise the key points."
```

## Python API

```python
from qdrant_rag import Config, RAGPipeline

config = Config()               # reads from .env / environment
pipeline = RAGPipeline(config)

# Ingest
n = pipeline.ingest_file("docs/report.pdf")
print(f"Stored {n} chunks.")

# Query
result = pipeline.query("What are the key findings?")
print(result.answer)
for src in result.sources:
    print(src["score"], src["metadata"]["source"])
```

## Project layout

```
qdrant_rag/
├── config.py           # Configuration (env vars / .env)
├── document_loader.py  # File reading + text chunking
├── embeddings.py       # sentence-transformers / OpenAI embedding service
├── vector_store.py     # Qdrant client wrapper
└── rag_pipeline.py     # End-to-end RAG pipeline

ingest.py               # CLI: ingest documents
query.py                # CLI: query with RAG

tests/
├── test_config.py
├── test_document_loader.py
├── test_embeddings.py
├── test_vector_store.py
└── test_rag_pipeline.py
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
