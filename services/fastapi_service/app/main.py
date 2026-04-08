from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag_service import (
    ask_question,
    ingest_staged_pdfs,
    ingest_uploaded_pdfs,
    ingest_url,
    save_uploaded_pdfs,
)
from .session_store import cleanup_user, ensure_roots, list_user_data_counts, sanitize_user_id

app = FastAPI(title="Mermaid FastAPI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _public_save_report(save_report: dict[str, Any]) -> dict[str, Any]:
    saved_files = list(save_report.get("saved_files") or [])
    unchanged_files = list(save_report.get("unchanged_files") or [])
    return {
        "saved_files": saved_files,
        "unchanged_files": unchanged_files,
        "saved_count": len(saved_files),
        "unchanged_count": len(unchanged_files),
    }


class URLIngestRequest(BaseModel):
    url: str = Field(min_length=5)


class StagedPDFIngestRequest(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=20)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2)


@app.on_event("startup")
def _on_startup() -> None:
    ensure_roots()


def _get_user_id(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header is required")

    try:
        return sanitize_user_id(x_user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest/pdfs")
def ingest_pdfs_endpoint(
    files: list[UploadFile] = File(...),
    x_user_id: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = _get_user_id(x_user_id)

    pdf_files = [f for f in files if (f.filename or "").lower().endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF files provided")

    try:
        save_report = save_uploaded_pdfs(user_id, pdf_files)
        ingest_report = ingest_uploaded_pdfs(user_id, save_report["paths"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "saved": _public_save_report(save_report),
        "ingest": ingest_report,
    }


@app.post("/ingest/url")
def ingest_url_endpoint(
    payload: URLIngestRequest,
    x_user_id: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = _get_user_id(x_user_id)
    try:
        result = ingest_url(user_id, payload.url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/ingest/staged-pdfs")
def ingest_staged_pdfs_endpoint(
    payload: StagedPDFIngestRequest,
    x_user_id: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = _get_user_id(x_user_id)

    try:
        result = ingest_staged_pdfs(user_id, payload.paths)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "saved": _public_save_report(result["saved"]),
        "ingest": result["ingest"],
    }


@app.post("/chat")
def chat_endpoint(
    payload: ChatRequest,
    x_user_id: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = _get_user_id(x_user_id)

    try:
        answer, rec = ask_question(user_id, payload.question)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    retrieval = [
        {
            "label": c.label,
            "score": c.score,
            "vector": c.vector_score,
            "lexical": c.lexical_score,
            "source": c.metadata.get("source"),
        }
        for c in rec.chunks
    ]

    return {
        "answer": answer,
        "retrieval": retrieval,
        "stats": rec.stats,
    }


@app.post("/session/cleanup")
def cleanup_endpoint(x_user_id: str | None = Header(default=None)) -> dict[str, str]:
    user_id = _get_user_id(x_user_id)
    cleanup_user(user_id)
    return {"status": "deleted"}


@app.get("/session/stats")
def stats_endpoint(x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    user_id = _get_user_id(x_user_id)
    return list_user_data_counts(user_id)
