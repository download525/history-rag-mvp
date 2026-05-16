#!/usr/bin/env python3
"""
Small dataset parser for istmat.org.

It builds an MVP corpus for a RAG project about reforms and reformers:
1. scans document listing pages;
2. scores titles by historical-reform keywords;
3. downloads the best 50 document pages;
4. saves JSONL and CSV datasets.

Usage:
    python parse_istmat.py --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

try:
    from lxml import html
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: lxml. Install it with `pip install lxml` "
        "or run the script with the Codex bundled Python."
    ) from exc


BASE_URL = "https://istmat.org"
LISTING_URL = f"{BASE_URL}/documents?page={{page}}"
REQUEST_TIMEOUT = 15

REFORM_KEYWORDS = {
    "реформ": 7,
    "столып": 7,
    "витте": 7,
    "лорис-меликов": 6,
    "александр ii": 5,
    "петр": 4,
    "пётр": 4,
    "манифест": 5,
    "указ": 4,
    "положение": 4,
    "закон": 4,
    "устав": 4,
    "постановление": 4,
    "декрет": 4,
    "государствен": 3,
    "управлен": 3,
    "земск": 5,
    "судеб": 5,
    "крестьян": 5,
    "выкуп": 5,
    "воинск": 4,
    "аграр": 5,
    "финанс": 4,
    "промышлен": 3,
    "железн": 3,
    "нэп": 5,
    "перестрой": 5,
    "индустриализац": 3,
    "коллективизац": 3,
}

NEGATIVE_KEYWORDS = {
    "внешняя политика": -4,
    "соглашение между": -3,
    "договор между": -3,
    "конвенция между": -3,
    "меморандум": -2,
}


@dataclass(frozen=True)
class Candidate:
    title: str
    date: str
    url: str
    tags: str
    score: int


def fetch(url: str, timeout: int | None = None) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; history-rag-mvp-parser/1.0)"
        },
    )
    with urlopen(request, timeout=timeout or REQUEST_TIMEOUT) as response:
        raw = response.read()
        content_type = response.headers.get("content-type", "")
        match = re.search(r"charset=([\w-]+)", content_type, re.I)
        encoding = match.group(1) if match else "utf-8"
        return raw.decode(encoding, errors="replace")


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_of(element) -> str:
    return normalize_space(" ".join(part.strip() for part in element.itertext()))


def score_text(text: str) -> int:
    value = 0
    lower = text.lower().replace("ё", "е")
    for keyword, weight in REFORM_KEYWORDS.items():
        value += lower.count(keyword.replace("ё", "е")) * weight
    for keyword, weight in NEGATIVE_KEYWORDS.items():
        value += lower.count(keyword.replace("ё", "е")) * weight
    return value


def parse_listing_page(page_html: str) -> list[Candidate]:
    doc = html.fromstring(page_html)
    candidates: list[Candidate] = []

    for link in doc.xpath("//a[starts-with(@href, '/node/')]"):
        title = text_of(link)
        href = link.get("href") or ""
        if not title or not re.search(r"\d", href):
            continue

        row_nodes = link.xpath("ancestor::tr[1]")
        row = row_nodes[0] if row_nodes else link.getparent()
        row_text = text_of(row)
        if not row_text:
            continue

        date_match = re.search(r"\b(\d{3,4}(?:\.\d{1,2})?(?:\.\d{1,2})?)\b", row_text)
        date = date_match.group(1) if date_match else ""
        tags = row_text.replace(title, " ")
        score = score_text(f"{title} {tags}")

        if score <= 0:
            continue

        candidates.append(
            Candidate(
                title=title,
                date=date,
                url=urljoin(BASE_URL, href),
                tags=normalize_space(tags),
                score=score,
            )
        )

    return candidates


def clean_page_text(page_html: str) -> tuple[str, str]:
    doc = html.fromstring(page_html)

    for bad in doc.xpath("//script|//style|//nav|//form|//noscript"):
        bad.drop_tree()

    title = ""
    title_nodes = doc.xpath("//h1")
    if title_nodes:
        title = text_of(title_nodes[0])

    body = doc.xpath("//body")
    full_text = text_of(body[0] if body else doc)

    if title and title in full_text:
        full_text = full_text.split(title, 1)[1]

    stop_markers = [
        "Страницы",
        "Статистика",
        "Облако меток",
        "Добавить комментарий",
        "Форма поиска",
    ]
    for marker in stop_markers:
        if marker in full_text:
            full_text = full_text.split(marker, 1)[0]

    lines = []
    for line in full_text.splitlines():
        line = normalize_space(line)
        if not line:
            continue
        if line in {"Главная", "Документы", "Библиотека", "Вход", "Регистрация"}:
            continue
        lines.append(line)

    text = normalize_space("\n".join(lines))
    return title, text


def unique_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    result: list[Candidate] = []
    for item in sorted(candidates, key=lambda c: (-c.score, c.date, c.title)):
        if item.url in seen:
            continue
        seen.add(item.url)
        result.append(item)
    return result


def collect_candidates(pages: int, pause: float) -> list[Candidate]:
    collected: list[Candidate] = []
    for page in range(pages):
        url = LISTING_URL.format(page=page)
        try:
            page_html = fetch(url)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[warn] cannot fetch listing {url}: {exc}", file=sys.stderr)
            continue

        found = parse_listing_page(page_html)
        collected.extend(found)
        print(f"[list] page={page:03d}, matched={len(found)}, total={len(collected)}")
        time.sleep(pause)

    return unique_candidates(collected)


def build_dataset(candidates: list[Candidate], limit: int, pause: float) -> list[dict]:
    rows: list[dict] = []
    for candidate in candidates:
        if len(rows) >= limit:
            break

        try:
            page_html = fetch(candidate.url)
            page_title, text = clean_page_text(page_html)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[warn] cannot fetch document {candidate.url}: {exc}", file=sys.stderr)
            continue

        title = page_title or candidate.title
        if len(text) < 500:
            print(f"[skip] too short: {candidate.url}")
            continue

        row = {
            "id": f"istmat_{len(rows) + 1:03d}",
            "title": title,
            "date": candidate.date,
            "topic": infer_topic(f"{title} {candidate.tags} {text[:1000]}"),
            "source_url": candidate.url,
            "score": candidate.score,
            "text": text,
        }
        rows.append(row)
        print(f"[doc] {len(rows):02d}/{limit}: {title[:90]}")
        time.sleep(pause)

    return rows


def infer_topic(text: str) -> str:
    lower = text.lower().replace("ё", "е")
    checks = [
        ("аграрная политика", ("столып", "аграр", "крестьян", "землеустрой")),
        ("государственное управление", ("государствен", "управлен", "совет министров", "министер")),
        ("право и суд", ("судеб", "юстиц", "прокуратур", "закон")),
        ("экономика и финансы", ("витте", "финанс", "промышлен", "железн", "налог")),
        ("военная реформа", ("воинск", "военн", "арм", "артиллер")),
        ("советская политика", ("нэп", "совнарком", "вкп", "цк", "перестрой")),
    ]
    for topic, words in checks:
        if any(word in lower for word in words):
            return topic
    return "реформы и государственная политика"


def save_dataset(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "istmat_reforms_50.jsonl"
    csv_path = out_dir / "istmat_reforms_50.csv"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "title", "date", "topic", "source_url", "score", "text"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} documents:")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")


def main() -> None:
    global REQUEST_TIMEOUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="number of documents to save")
    parser.add_argument(
        "--pages",
        type=int,
        default=120,
        help="number of istmat listing pages to scan before downloading documents",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.4,
        help="pause between requests, seconds",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="directory for JSONL and CSV output",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="request timeout, seconds",
    )
    args = parser.parse_args()
    REQUEST_TIMEOUT = args.timeout

    candidates = collect_candidates(args.pages, args.pause)
    if not candidates:
        raise SystemExit("No matching documents found. Try increasing --pages.")

    print(f"\nSelected candidate pool: {len(candidates)}")
    dataset = build_dataset(candidates, args.limit, args.pause)

    if len(dataset) < args.limit:
        print(
            f"[warn] saved only {len(dataset)} documents out of requested {args.limit}. "
            "Try increasing --pages.",
            file=sys.stderr,
        )

    save_dataset(dataset, args.out_dir)


if __name__ == "__main__":
    main()
