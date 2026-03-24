#!/usr/bin/env python3
"""CLI for ingesting documents into the Qdrant RAG collection."""

from __future__ import annotations

import argparse
import sys

from qdrant_rag import Config, RAGPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qdrant-ingest",
        description="Ingest documents into the Qdrant RAG vector store.",
    )
    parser.add_argument(
        "path",
        help="Path to a file or directory to ingest.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        metavar="NAME",
        help="Override the Qdrant collection name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Maximum characters per text chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="N",
        help="Overlap characters between consecutive chunks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = Config()
    if args.collection:
        config.collection_name = args.collection
    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        config.chunk_overlap = args.chunk_overlap

    pipeline = RAGPipeline(config=config)

    import os

    path = args.path
    try:
        if os.path.isdir(path):
            count = pipeline.ingest_directory(path)
        else:
            count = pipeline.ingest_file(path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error during ingestion: {exc}", file=sys.stderr)
        return 1

    print(f"Successfully ingested {count} chunk(s) from '{path}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
