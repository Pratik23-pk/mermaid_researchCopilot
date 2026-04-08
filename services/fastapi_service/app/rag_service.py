import hashlib
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from .rag.ingest_pdf import ingest_pdfs
from .rag.ingest_web import ingest_web_page
from .rag.llm_client import llm_chat
from .rag.recommender import RecommendationResult, retrieve_recommended_context
from .session_store import COLLECTION_NAME, user_chroma_dir, user_pdf_dir, validate_staged_pdf_path

SYSTEM_PROMPT = (
    "You are Mermaid, a research assistant. "
    "Use the provided context documents to answer the question. "
    "If the context is insufficient, say you are unsure instead of guessing."
)

COPY_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_FILE_BYTES = int(os.getenv("MAX_UPLOAD_FILE_BYTES", str(150 * 1024 * 1024)))
MAX_TOTAL_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_TOTAL_BYTES", str(500 * 1024 * 1024)))


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "uploaded.pdf")
    chars = []
    for ch in base:
        if ch.isalnum() or ch in {".", "_", "-"}:
            chars.append(ch)
        else:
            chars.append("_")
    safe = "".join(chars)
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    return safe


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_SIZE), b""):
            if chunk:
                digest.update(chunk)
    return digest.hexdigest()


def _limit_error_message(limit_bytes: int) -> str:
    limit_mb = max(1, limit_bytes // (1024 * 1024))
    return f"Upload exceeds the configured limit of {limit_mb} MB."


def _stream_to_temp_path(stream, temp_path: Path) -> tuple[int, str]:
    if hasattr(stream, "seek"):
        stream.seek(0)

    total_bytes = 0
    digest = hashlib.sha1()

    with temp_path.open("wb") as out:
        while True:
            chunk = stream.read(COPY_CHUNK_SIZE)
            if not chunk:
                break

            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_FILE_BYTES:
                raise ValueError(_limit_error_message(MAX_UPLOAD_FILE_BYTES))

            digest.update(chunk)
            out.write(chunk)

    return total_bytes, digest.hexdigest()


def _finalize_saved_pdf(
    *,
    temp_path: Path,
    pdf_dir: Path,
    original_name: str,
    digest: str,
) -> tuple[Path, str, bool]:
    base, ext = os.path.splitext(original_name)
    primary_path = pdf_dir / original_name

    if not primary_path.exists():
        temp_path.replace(primary_path)
        return primary_path, original_name, True

    if _hash_file(primary_path) == digest:
        temp_path.unlink(missing_ok=True)
        return primary_path, original_name, False

    candidate_name = f"{base}_{digest[:8]}{ext}"
    candidate_path = pdf_dir / candidate_name
    counter = 1

    while candidate_path.exists():
        if _hash_file(candidate_path) == digest:
            temp_path.unlink(missing_ok=True)
            return candidate_path, candidate_name, False
        candidate_name = f"{base}_{digest[:8]}_{counter}{ext}"
        candidate_path = pdf_dir / candidate_name
        counter += 1

    temp_path.replace(candidate_path)
    return candidate_path, candidate_name, True


def save_uploaded_pdfs(user_id: str, files: list[UploadFile]) -> dict[str, Any]:
    pdf_dir = user_pdf_dir(user_id)

    saved_paths: list[str] = []
    saved_files: list[str] = []
    unchanged_files: list[str] = []
    created_paths: list[Path] = []
    total_request_bytes = 0

    try:
        for upload in files:
            name = _safe_filename(upload.filename or "uploaded.pdf")
            temp_path = pdf_dir / f".upload_{uuid4().hex}.part"

            try:
                file_bytes, digest = _stream_to_temp_path(upload.file, temp_path)
                total_request_bytes += file_bytes
                if total_request_bytes > MAX_TOTAL_UPLOAD_BYTES:
                    raise ValueError(_limit_error_message(MAX_TOTAL_UPLOAD_BYTES))

                target_path, target_name, was_created = _finalize_saved_pdf(
                    temp_path=temp_path,
                    pdf_dir=pdf_dir,
                    original_name=name,
                    digest=digest,
                )
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise

            if was_created:
                created_paths.append(target_path)
                saved_files.append(target_name)
            else:
                unchanged_files.append(target_name)

            saved_paths.append(str(target_path))
    except Exception:
        for created_path in created_paths:
            created_path.unlink(missing_ok=True)
        raise

    return {
        "paths": saved_paths,
        "saved_files": saved_files,
        "unchanged_files": unchanged_files,
    }


def save_staged_pdfs(user_id: str, staged_paths: list[str]) -> dict[str, Any]:
    pdf_dir = user_pdf_dir(user_id)

    saved_paths: list[str] = []
    saved_files: list[str] = []
    unchanged_files: list[str] = []
    created_paths: list[Path] = []
    total_request_bytes = 0

    try:
        for raw_path in staged_paths:
            staged_path = validate_staged_pdf_path(raw_path)
            name = _safe_filename(staged_path.name)
            temp_path = pdf_dir / f".stage_{uuid4().hex}.part"

            try:
                with staged_path.open("rb") as source:
                    file_bytes, digest = _stream_to_temp_path(source, temp_path)

                total_request_bytes += file_bytes
                if total_request_bytes > MAX_TOTAL_UPLOAD_BYTES:
                    raise ValueError(_limit_error_message(MAX_TOTAL_UPLOAD_BYTES))

                target_path, target_name, was_created = _finalize_saved_pdf(
                    temp_path=temp_path,
                    pdf_dir=pdf_dir,
                    original_name=name,
                    digest=digest,
                )
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise

            if was_created:
                created_paths.append(target_path)
                saved_files.append(target_name)
            else:
                unchanged_files.append(target_name)

            saved_paths.append(str(target_path))
    except Exception:
        for created_path in created_paths:
            created_path.unlink(missing_ok=True)
        raise

    return {
        "paths": saved_paths,
        "saved_files": saved_files,
        "unchanged_files": unchanged_files,
    }


def ingest_uploaded_pdfs(user_id: str, file_paths: list[str]) -> dict[str, Any]:
    chroma_dir = user_chroma_dir(user_id)
    pdf_dir = user_pdf_dir(user_id)

    return ingest_pdfs(
        file_paths=file_paths,
        pdf_dir=str(pdf_dir),
        chroma_dir=str(chroma_dir),
        collection_name=COLLECTION_NAME,
    )


def ingest_staged_pdfs(user_id: str, staged_paths: list[str]) -> dict[str, Any]:
    save_report = save_staged_pdfs(user_id, staged_paths)
    ingest_report = ingest_uploaded_pdfs(user_id, save_report["paths"])

    return {
        "saved": save_report,
        "ingest": ingest_report,
    }


def ingest_url(user_id: str, url: str) -> dict[str, Any]:
    chroma_dir = user_chroma_dir(user_id)
    return ingest_web_page(
        url,
        chroma_dir=str(chroma_dir),
        collection_name=COLLECTION_NAME,
    )


def ask_question(user_id: str, question: str) -> tuple[str, RecommendationResult]:
    chroma_dir = user_chroma_dir(user_id)
    rec = retrieve_recommended_context(
        question,
        chroma_dir=str(chroma_dir),
        collection_name=COLLECTION_NAME,
    )

    if rec.context:
        user_content = (
            f"Context documents:\n{rec.context}\n\n"
            f"User question: {question}\n\n"
            "Answer using only the information from the context when possible. "
            "If something is not in the context, say you are unsure."
        )
    else:
        user_content = (
            "No context documents are currently indexed or retrievable for this user.\n\n"
            f"User question: {question}\n\n"
            "State that you are unsure and ask the user to ingest documents."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    answer = llm_chat(messages)
    return answer, rec
