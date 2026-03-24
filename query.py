#!/usr/bin/env python3
"""CLI for querying the Qdrant RAG pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from qdrant_rag import Config, RAGPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qdrant-query",
        description="Ask a question against the Qdrant RAG collection.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask.  If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        metavar="NAME",
        help="Override the Qdrant collection name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help="Number of context chunks to retrieve.",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output the full result as JSON.",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help=(
            "Skip the LLM generation step and only print the retrieved "
            "context chunks (useful without an OpenAI key)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    question = args.question
    if not question:
        if sys.stdin.isatty():
            print("Enter your question: ", end="", flush=True)
        question = sys.stdin.readline().strip()
    if not question:
        print("Error: no question provided.", file=sys.stderr)
        return 1

    config = Config()
    if args.collection:
        config.collection_name = args.collection
    if args.top_k is not None:
        config.top_k = args.top_k

    pipeline = RAGPipeline(config=config)

    try:
        if args.no_generate:
            from qdrant_rag.embeddings import EmbeddingService

            svc = EmbeddingService(
                model_name=config.embedding_model,
                openai_api_key=config.openai_api_key,
            )
            vec = svc.embed([question])[0]
            sources = pipeline._vector_store.search(vec, top_k=config.top_k)

            if args.output_json:
                print(json.dumps({"query": question, "sources": sources}, indent=2))
            else:
                for i, s in enumerate(sources, 1):
                    print(f"\n[{i}] score={s['score']:.4f}")
                    print(s["text"])
        else:
            result = pipeline.query(question)

            if args.output_json:
                print(
                    json.dumps(
                        {
                            "query": result.query,
                            "answer": result.answer,
                            "sources": result.sources,
                        },
                        indent=2,
                    )
                )
            else:
                print(f"\nAnswer:\n{result.answer}")
                print(f"\n--- {len(result.sources)} source chunk(s) retrieved ---")
                for i, s in enumerate(result.sources, 1):
                    print(f"\n[{i}] score={s['score']:.4f}  {s['metadata']}")
                    print(s["text"][:200] + ("…" if len(s["text"]) > 200 else ""))

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
