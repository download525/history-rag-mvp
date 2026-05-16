#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def ok(label: str, value: str = "") -> None:
    print(f"[OK] {label}{': ' + value if value else ''}")


def fail(label: str, error: Exception | str) -> None:
    print(f"[FAIL] {label}: {error}")


def main() -> None:
    try:
        import chromadb

        ok("chromadb")
    except Exception as exc:
        fail("chromadb", exc)

    try:
        import sentence_transformers

        ok("sentence-transformers")
    except Exception as exc:
        fail("sentence-transformers", exc)

    try:
        from dotenv import dotenv_values

        env = dotenv_values(".env")
        ok(".env", "found" if Path(".env").exists() else "missing")
        ok("GIGACHAT_CREDENTIALS", "set" if env.get("GIGACHAT_CREDENTIALS") else "missing")
        ok("GIGACHAT_ACCESS_TOKEN", "set" if env.get("GIGACHAT_ACCESS_TOKEN") else "not used")
        ok("GIGACHAT_VERIFY_SSL", env.get("GIGACHAT_VERIFY_SSL", "not set"))
    except Exception as exc:
        fail("python-dotenv / .env", exc)

    db_path = Path("chroma_db")
    if db_path.exists():
        ok("chroma_db", str(db_path.resolve()))
    else:
        fail("chroma_db", "folder is missing, run `python build_chroma.py --reset`")

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_collection(
            name="istmat_reforms",
            embedding_function=embedding_function,
        )
        ok("Chroma collection", f"istmat_reforms, chunks={collection.count()}")
    except Exception as exc:
        fail("Chroma collection", exc)


if __name__ == "__main__":
    main()
