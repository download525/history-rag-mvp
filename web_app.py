#!/usr/bin/env python3
"""
Tiny local web chat for the istmat.org RAG MVP.

Run:
    python web_app.py
Then open:
    http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

from rag_gigachat import (
    DEFAULT_COLLECTION,
    DEFAULT_DB_DIR,
    DEFAULT_MODEL,
    ask_gigachat_with_access_token,
    ask_gigachat_with_credentials,
    bool_from_env,
    build_prompt,
    diversify_results,
)

import os


APP_TITLE = "Исторический RAG"

HTML = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Исторический RAG</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1f2623;
      --muted: #66736d;
      --line: #d9d2c3;
      --accent: #27605b;
      --accent-2: #a84f2a;
      --soft: #e7efe7;
      --shadow: 0 20px 60px rgba(31, 38, 35, .13);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, rgba(39,96,91,.08), transparent 260px),
        var(--bg);
      color: var(--ink);
      font: 16px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .shell {
      width: min(1180px, calc(100vw - 32px));
      height: min(860px, calc(100vh - 32px));
      margin: 16px auto;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    aside {
      border-right: 1px solid var(--line);
      padding: 24px 20px;
      background: #fbf7ee;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    .brand h1 {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.1;
      letter-spacing: 0;
    }

    .brand p, .hint, .source-empty {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }

    .examples {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .examples button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      padding: 10px 12px;
      text-align: left;
      cursor: pointer;
      font: inherit;
      font-size: 14px;
    }

    .examples button:hover {
      border-color: var(--accent);
      background: var(--soft);
    }

    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      background: #fffdf8;
    }

    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 99px;
      background: #4d9a5c;
    }

    .chat {
      padding: 22px 24px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-height: 0;
    }

    .message {
      max-width: 820px;
      border-radius: 8px;
      padding: 14px 16px;
      white-space: pre-wrap;
      word-wrap: break-word;
      overflow-wrap: anywhere;
    }

    .message.user {
      align-self: flex-end;
      background: var(--accent);
      color: white;
    }

    .message.assistant {
      align-self: flex-start;
      background: #f3efe5;
      border: 1px solid var(--line);
    }

    .message.answer {
      width: min(820px, 100%);
      max-height: min(52vh, 520px);
      overflow-y: auto;
      padding-right: 18px;
      scrollbar-color: var(--accent) #e5dfd1;
      scrollbar-width: thin;
    }

    .message.answer::-webkit-scrollbar {
      width: 10px;
    }

    .message.answer::-webkit-scrollbar-track {
      background: #e5dfd1;
      border-radius: 99px;
    }

    .message.answer::-webkit-scrollbar-thumb {
      background: var(--accent);
      border-radius: 99px;
      border: 2px solid #e5dfd1;
    }

    .sources {
      max-width: 820px;
      align-self: flex-start;
      display: grid;
      gap: 8px;
      width: 100%;
      max-height: 180px;
      overflow-y: auto;
      padding-right: 4px;
    }

    .source {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
      font-size: 14px;
    }

    .source a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
    }

    .source a:hover { text-decoration: underline; }

    form {
      padding: 18px 24px;
      border-top: 1px solid var(--line);
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      background: #fbf7ee;
    }

    textarea {
      width: 100%;
      min-height: 48px;
      max-height: 150px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      background: white;
      color: var(--ink);
      font: inherit;
      outline: none;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(39,96,91,.12);
    }

    .send {
      border: 0;
      border-radius: 8px;
      padding: 0 20px;
      min-width: 112px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
    }

    .send:hover { background: #1e514c; }
    .send:disabled { opacity: .6; cursor: wait; }

    @media (max-width: 820px) {
      .shell {
        width: 100vw;
        height: 100vh;
        margin: 0;
        border-radius: 0;
        grid-template-columns: 1fr;
      }

      aside { display: none; }
      header { padding: 14px 16px; }
      .chat { padding: 16px; }
      .message.answer {
        max-height: 50vh;
      }
      form {
        padding: 12px 16px;
        grid-template-columns: 1fr;
      }
      .send {
        min-height: 44px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <section class="brand">
        <h1>Исторический RAG</h1>
        <p>Чат по корпусу документов istmat.org с поиском в ChromaDB и ответом через GigaChat.</p>
      </section>

      <section>
        <p class="hint">Примеры вопросов</p>
        <div class="examples">
          <button type="button">Как в документах описывается организация финансового управления на местах после революции?</button>
          <button type="button">Какие функции получало Центральное налоговое управление Народного комиссариата финансов?</button>
          <button type="button">Как советская власть регулировала финансирование государственных предприятий в 1919-1920 годах?</button>
          <button type="button">Какие трудности кредитной и налоговой реформ начала 1930-х годов отражены в документах?</button>
        </div>
      </section>

      <p class="hint">MVP ограничен текущим корпусом и найденными фрагментами. Если данных мало, система должна сказать об этом.</p>
    </aside>

    <main>
      <header>
        <strong>Чат с историческими документами</strong>
        <span class="status"><span class="dot"></span><span id="statusText">готов</span></span>
      </header>

      <section id="chat" class="chat">
        <div class="message assistant">Задай вопрос по корпусу. Лучше всего работают темы: финансовое управление, государственные предприятия, продовольствие Петрограда, кредитная и налоговая реформы.</div>
      </section>

      <form id="form">
        <textarea id="question" placeholder="Например: какие функции получало Центральное налоговое управление?" required></textarea>
        <button id="send" class="send" type="submit">Спросить</button>
      </form>
    </main>
  </div>

  <script>
    const form = document.querySelector("#form");
    const chat = document.querySelector("#chat");
    const question = document.querySelector("#question");
    const send = document.querySelector("#send");
    const statusText = document.querySelector("#statusText");

    function addMessage(text, type) {
      const node = document.createElement("div");
      node.className = `message ${type}`;
      node.textContent = text;
      chat.appendChild(node);
      chat.scrollTop = chat.scrollHeight;
      return node;
    }

    function markAsAnswer(node) {
      node.classList.add("answer");
      node.setAttribute("tabindex", "0");
      node.setAttribute("aria-label", "Ответ с прокруткой");
      node.scrollTop = 0;
      node.focus({ preventScroll: true });
    }

    function addSources(sources) {
      const wrap = document.createElement("div");
      wrap.className = "sources";
      if (!sources.length) {
        wrap.innerHTML = '<p class="source-empty">Источники не найдены.</p>';
      } else {
        sources.forEach((source, index) => {
          const item = document.createElement("div");
          item.className = "source";
          const link = document.createElement("a");
          link.href = source.url;
          link.target = "_blank";
          link.rel = "noreferrer";
          link.textContent = `${index + 1}. ${source.title || "Документ"}`;
          item.appendChild(link);
          wrap.appendChild(item);
        });
      }
      chat.appendChild(wrap);
      chat.scrollTop = chat.scrollHeight;
    }

    async function ask(text) {
      addMessage(text, "user");
      const pending = addMessage("Ищу фрагменты в корпусе и формирую ответ...", "assistant");
      statusText.textContent = "думаю";
      send.disabled = true;

      try {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text })
        });
        const raw = await response.text();
        let data;
        try {
          data = JSON.parse(raw);
        } catch {
          data = { error: raw || "Пустой ответ сервера" };
        }
        if (!response.ok) {
          throw new Error(data.error || "Ошибка запроса");
        }
        pending.textContent = data.answer;
        markAsAnswer(pending);
        addSources(data.sources || []);
        pending.scrollIntoView({ block: "nearest" });
      } catch (error) {
        pending.textContent = `Не получилось получить ответ: ${error.message}`;
      } finally {
        statusText.textContent = "готов";
        send.disabled = false;
        question.focus();
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;
      question.value = "";
      ask(text);
    });

    question.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        form.requestSubmit();
      }
    });

    document.querySelectorAll(".examples button").forEach((button) => {
      button.addEventListener("click", () => {
        question.value = button.textContent;
        question.focus();
      });
    });
  </script>
</body>
</html>
"""


class RagEngine:
    def __init__(
        self,
        db_dir: Path,
        collection_name: str,
        embedding_model: str,
        top_k: int,
        candidate_k: int,
    ) -> None:
        load_dotenv()
        self.top_k = top_k
        self.candidate_k = max(candidate_k, top_k)
        self.model = os.getenv("GIGACHAT_MODEL", "GigaChat")
        self.access_token = os.getenv("GIGACHAT_ACCESS_TOKEN")
        self.credentials = os.getenv("GIGACHAT_CREDENTIALS")
        self.verify_ssl = bool_from_env(os.getenv("GIGACHAT_VERIFY_SSL"), True)

        if not self.access_token and not self.credentials:
            raise RuntimeError("Set GIGACHAT_ACCESS_TOKEN or GIGACHAT_CREDENTIALS in .env")

        embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        client = chromadb.PersistentClient(path=str(db_dir))
        self.collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    def ask(self, question: str) -> dict:
        results = self.collection.query(query_texts=[question], n_results=self.candidate_k)
        documents, metadatas = diversify_results(results, self.top_k)
        prompt = build_prompt(question, documents, metadatas)

        if self.credentials:
            answer = ask_gigachat_with_credentials(prompt, self.credentials, self.model)
        else:
            answer = ask_gigachat_with_access_token(
                prompt,
                self.access_token,
                self.model,
                self.verify_ssl,
            )

        sources = []
        seen = set()
        for metadata in metadatas:
            url = metadata.get("source_url", "")
            if url in seen:
                continue
            seen.add(url)
            sources.append(
                {
                    "title": metadata.get("title", ""),
                    "url": url,
                    "date": metadata.get("date", ""),
                    "topic": metadata.get("topic", ""),
                }
            )

        return {"answer": answer, "sources": sources}


class AppHandler(BaseHTTPRequestHandler):
    engine: RagEngine

    def log_message(self, format: str, *args) -> None:
        return

    def send_text(self, body: str, content_type: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict, status: int = 200) -> None:
        self.send_text(json.dumps(payload, ensure_ascii=False), "application/json", status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self.send_text(HTML, "text/html")
        elif path == "/health":
            self.send_json({"ok": True, "title": APP_TITLE})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/ask":
            self.send_json({"error": "Not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            question = str(payload.get("question", "")).strip()
            if not question:
                self.send_json({"error": "Question is empty"}, 400)
                return
            self.send_json(self.engine.ask(question))
        except Exception as exc:
            print("\n/api/ask error:")
            traceback.print_exc()
            self.send_json({"error": str(exc)}, 500)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    args = parser.parse_args()

    try:
        AppHandler.engine = RagEngine(
            db_dir=args.db_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
    except Exception:
        print("Не удалось запустить RAG-движок:")
        traceback.print_exc()
        raise

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"{APP_TITLE}: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
