#!/usr/bin/env python3
"""
Ask questions over the local ChromaDB collection and generate answers with GigaChat.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

try:
    from gigachat import GigaChat
except ImportError:
    GigaChat = None


DEFAULT_DB_DIR = Path("chroma_db")
DEFAULT_COLLECTION = "istmat_reforms"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


def bool_from_env(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "да"}


def build_prompt(question: str, documents: list[str], metadatas: list[dict]) -> str:
    context_parts = []
    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        title = metadata.get("title", "Без названия")
        date = metadata.get("date", "")
        url = metadata.get("source_url", "")
        context_parts.append(
            f"[Фрагмент {index}]\n"
            f"Название: {title}\n"
            f"Дата: {date}\n"
            f"Источник: {url}\n"
            f"Текст:\n{document}"
        )

    context = "\n\n".join(context_parts)
    return f"""Ты помогаешь анализировать исторические документы.
Отвечай только на основе приведённых фрагментов. Если данных недостаточно, прямо скажи об этом.
В конце добавь краткий список использованных источников с названиями документов.

Вопрос:
{question}

Фрагменты корпуса:
{context}
"""


def diversify_results(results: dict, top_k: int) -> tuple[list[str], list[dict]]:
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    selected_documents: list[str] = []
    selected_metadatas: list[dict] = []
    seen_sources: set[str] = set()

    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source_url") or metadata.get("document_id") or metadata.get("title")
        if source in seen_sources:
            continue
        seen_sources.add(source)
        selected_documents.append(document)
        selected_metadatas.append(metadata)
        if len(selected_documents) >= top_k:
            return selected_documents, selected_metadatas

    for document, metadata in zip(documents, metadatas):
        if len(selected_documents) >= top_k:
            break
        selected_documents.append(document)
        selected_metadatas.append(metadata)

    return selected_documents, selected_metadatas


def ask_gigachat_with_access_token(
    prompt: str,
    access_token: str,
    model: str,
    verify_ssl: bool,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1400,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        GIGACHAT_CHAT_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    context = None if verify_ssl else ssl._create_unverified_context()

    try:
        with urlopen(request, timeout=60, context=context) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GigaChat API error {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot connect to GigaChat API: {exc}") from exc

    return result["choices"][0]["message"]["content"]


def ask_gigachat_with_credentials(prompt: str, credentials: str, model: str) -> str:
    if GigaChat is None:
        raise RuntimeError(
            "Package `gigachat` is not installed. Run `pip install -r requirements_rag.txt` "
            "or set GIGACHAT_ACCESS_TOKEN in .env."
        )

    giga = GigaChat(
        credentials=credentials,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        model=model,
        verify_ssl_certs=bool_from_env(os.getenv("GIGACHAT_VERIFY_SSL"), True),
    )
    response = giga.chat(prompt)
    return response.choices[0].message.content


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="question for the RAG system")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
        help="number of ChromaDB matches to inspect before keeping diverse sources",
    )
    args = parser.parse_args()

    question = args.question or input("Вопрос: ").strip()
    if not question:
        raise SystemExit("Question is empty.")

    access_token = os.getenv("GIGACHAT_ACCESS_TOKEN")
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    model = os.getenv("GIGACHAT_MODEL", "GigaChat")
    if not access_token and not credentials:
        raise SystemExit("Set GIGACHAT_ACCESS_TOKEN or GIGACHAT_CREDENTIALS in .env first.")

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=args.embedding_model)
    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(
        name=args.collection,
        embedding_function=embedding_function,
    )

    candidate_k = max(args.candidate_k, args.top_k)
    results = collection.query(query_texts=[question], n_results=candidate_k)
    documents, metadatas = diversify_results(results, args.top_k)

    prompt = build_prompt(question, documents, metadatas)

    verify_ssl = bool_from_env(os.getenv("GIGACHAT_VERIFY_SSL"), True)

    if credentials:
        answer = ask_gigachat_with_credentials(prompt, credentials, model)
    else:
        answer = ask_gigachat_with_access_token(prompt, access_token, model, verify_ssl)

    print(answer)

    print("\n--- Найденные фрагменты ---")
    for index, metadata in enumerate(metadatas, start=1):
        print(f"{index}. {metadata.get('title', '')} | {metadata.get('source_url', '')}")


if __name__ == "__main__":
    main()
