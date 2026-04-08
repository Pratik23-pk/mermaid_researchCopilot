# ingest_pdf.py

import hashlib
import os
from typing import Any

import chromadb
from PyPDF2 import PdfReader

from ingest_utils import chunk_text
from llm_client import embed_texts

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "research_docs"
PDF_DIR = "pdfs"


def _safe_pdf_id(filename: str, chunk_index: int, chunk_text_value: str) -> str:
    digest = hashlib.sha1(chunk_text_value.encode("utf-8")).hexdigest()[:12]
    return f"pdf::{filename}::{chunk_index:05d}::{digest}"


def _pdf_where_filter(filename: str) -> dict[str, Any]:
    return {
        "$and": [
            {"source": {"$eq": "pdf"}},
            {"filename": {"$eq": filename}},
        ]
    }


def extract_text_from_pdf(path: str, max_pages: int | None = None) -> tuple[str, int, int]:
    reader = PdfReader(path)

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise RuntimeError(f"Could not decrypt PDF: {path}") from exc

    all_text: list[str] = []
    pages_total = len(reader.pages)
    pages_to_read = pages_total if max_pages is None else min(max_pages, pages_total)
    non_empty_pages = 0

    for i in range(pages_to_read):
        page = reader.pages[i]
        text = (page.extract_text() or "").strip()
        if text:
            non_empty_pages += 1
            all_text.append(text)

    return "\n\n".join(all_text), pages_to_read, non_empty_pages


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


def _delete_existing_pdf_chunks(collection, filename: str):
    collection.delete(where=_pdf_where_filter(filename))


def ingest_pdfs(
    pdf_dir: str = PDF_DIR,
    max_pages_per_pdf: int | None = None,
    file_paths: list[str] | None = None,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, Any]:
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
    )

    pdf_entries: list[tuple[str, str]] = []
    if file_paths is not None:
        for raw_path in file_paths:
            path = os.path.abspath(raw_path)
            fname = os.path.basename(path)
            if not fname.lower().endswith(".pdf"):
                continue
            if not os.path.exists(path):
                continue
            pdf_entries.append((fname, path))
    else:
        pdf_entries = [
            (fname, os.path.join(pdf_dir, fname))
            for fname in sorted(os.listdir(pdf_dir))
            if fname.lower().endswith(".pdf")
        ]

    summary: dict[str, Any] = {
        "files_total": len(pdf_entries),
        "files_ingested": 0,
        "chunks_added": 0,
        "skipped": [],
        "failed": [],
    }

    for fname, path in pdf_entries:

        try:
            raw_text, pages_read, non_empty_pages = extract_text_from_pdf(
                path,
                max_pages=max_pages_per_pdf,
            )
        except Exception as exc:
            summary["failed"].append({"file": fname, "error": str(exc)})
            continue

        chunks = chunk_text(raw_text)
        if not chunks:
            summary["skipped"].append(
                {
                    "file": fname,
                    "reason": "No extractable text found",
                    "pages_read": pages_read,
                    "non_empty_pages": non_empty_pages,
                }
            )
            # Remove any stale chunks from prior runs for this file.
            _delete_existing_pdf_chunks(collection, fname)
            continue

        # Re-ingesting the same PDF should replace existing chunks for that file.
        _delete_existing_pdf_chunks(collection, fname)

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            doc_id = _safe_pdf_id(fname, i, chunk)
            ids.append(doc_id)
            docs.append(chunk)
            metadatas.append(
                {
                    "filename": fname,
                    "source": "pdf",
                    "chunk": i,
                    "pages_read": pages_read,
                    "non_empty_pages": non_empty_pages,
                }
            )

        try:
            _upsert_documents(collection, ids, docs, metadatas)
        except Exception as exc:
            summary["failed"].append({"file": fname, "error": str(exc)})
            continue

        summary["files_ingested"] += 1
        summary["chunks_added"] += len(docs)

    return summary


if __name__ == "__main__":
    result = ingest_pdfs()
    print(result)
