# ingest_web.py

import hashlib
import io
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

import chromadb
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from ingest_utils import chunk_text
from llm_client import embed_texts

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "research_docs"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def normalize_url(url: str) -> str:
    raw = url.strip()
    if not raw:
        raise ValueError("URL is empty")

    parsed = urlparse(raw)
    if not parsed.scheme:
        parsed = urlparse(f"https://{raw}")

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are supported")

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",  # drop fragment
        )
    )


def _web_where_filter(url: str) -> dict[str, Any]:
    return {
        "$and": [
            {"source": {"$eq": "web"}},
            {"url": {"$eq": url}},
        ]
    }


def _safe_web_id(url: str, chunk_index: int, chunk_text_value: str) -> str:
    url_digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    chunk_digest = hashlib.sha1(chunk_text_value.encode("utf-8")).hexdigest()[:12]
    return f"web::{url_digest}::{chunk_index:05d}::{chunk_digest}"


def _extract_text_from_pdf_bytes(content: bytes) -> tuple[str, int, int]:
    reader = PdfReader(io.BytesIO(content))

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise RuntimeError("Could not decrypt PDF from URL") from exc

    texts: list[str] = []
    non_empty_pages = 0
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            non_empty_pages += 1
            texts.append(text)

    return "\n\n".join(texts), len(reader.pages), non_empty_pages


def _extract_text_from_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def fetch_page_text(url: str, timeout_seconds: int = 30) -> str:
    normalized = normalize_url(url)

    try:
        resp = requests.get(
            normalized,
            timeout=timeout_seconds,
            headers=REQUEST_HEADERS,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch URL: {normalized}. Error: {exc}") from exc

    content_type = (resp.headers.get("content-type") or "").lower()

    if "application/pdf" in content_type or normalized.lower().endswith(".pdf"):
        pdf_text, _, _ = _extract_text_from_pdf_bytes(resp.content)
        return pdf_text

    if "text/plain" in content_type:
        return resp.text

    return _extract_text_from_html(resp.text)


def _upsert_documents(collection, ids: list[str], docs: list[str], metadatas: list[dict[str, Any]]):
    embeddings = embed_texts(docs)
    try:
        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return
    except AttributeError:
        collection.delete(ids=ids)
        collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings,
        )


def _delete_existing_web_chunks(collection, normalized_url: str):
    collection.delete(where=_web_where_filter(normalized_url))


def ingest_web_page(
    url: str,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, Any]:
    normalized_url = normalize_url(url)

    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
    )

    raw_text = fetch_page_text(normalized_url)
    chunks = chunk_text(raw_text)

    if not chunks:
        _delete_existing_web_chunks(collection, normalized_url)
        return {
            "url": normalized_url,
            "chunks_added": 0,
            "status": "skipped",
            "reason": "No text found on page",
        }

    # Replace previous chunks for this URL (deterministic re-ingestion).
    _delete_existing_web_chunks(collection, normalized_url)

    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        doc_id = _safe_web_id(normalized_url, i, chunk)
        ids.append(doc_id)
        docs.append(chunk)
        metadatas.append({"url": normalized_url, "source": "web", "chunk": i})

    _upsert_documents(collection, ids, docs, metadatas)

    return {
        "url": normalized_url,
        "chunks_added": len(docs),
        "status": "ok",
    }


if __name__ == "__main__":
    test_url = "https://www.python.org/"
    result = ingest_web_page(test_url)
    print(result)
