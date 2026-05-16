#!/usr/bin/env python3
"""
Build a persistent ChromaDB collection from the parsed istmat.org JSONL dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


DEFAULT_DATASET = Path("data/istmat_reforms_50.jsonl")
DEFAULT_DB_DIR = Path("chroma_db")
DEFAULT_COLLECTION = "istmat_reforms"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = normalize_space(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            sentence_end = max(chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "))
            paragraph_end = chunk.rfind("\n")
            cut = max(sentence_end, paragraph_end)
            if cut > chunk_size * 0.55:
                end = start + cut + 1
                chunk = text[start:end]

        chunk = normalize_space(chunk)
        if chunk:
            chunks.append(chunk)

        next_start = max(end - overlap, start + 1)
        start = next_start

    return chunks


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=180)
    parser.add_argument("--reset", action="store_true", help="delete the collection before adding chunks")
    args = parser.parse_args()

    rows = load_rows(args.dataset)
    if not rows:
        raise SystemExit(f"No rows found in {args.dataset}")

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=args.model)
    client = chromadb.PersistentClient(path=str(args.db_dir))

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for row in rows:
        chunks = split_text(row.get("text", ""), args.chunk_size, args.overlap)
        for index, chunk in enumerate(chunks):
            ids.append(f"{row['id']}_chunk_{index:04d}")
            documents.append(chunk)
            metadatas.append(
                {
                    "document_id": row.get("id", ""),
                    "title": row.get("title", ""),
                    "date": row.get("date", ""),
                    "topic": row.get("topic", ""),
                    "source_url": row.get("source_url", ""),
                    "chunk_index": index,
                }
            )

    if not documents:
        raise SystemExit("No chunks were created. Check the dataset text field.")

    batch_size = 128
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Added chunks: {min(end, len(documents))}/{len(documents)}")

    print("\nChromaDB is ready")
    print(f"Documents: {len(rows)}")
    print(f"Chunks: {len(documents)}")
    print(f"Database: {args.db_dir}")
    print(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()
