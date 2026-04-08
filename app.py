# app.py

import atexit
import hashlib
import io
import os
import re
import shutil
import time
import uuid
from html import escape
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from PyPDF2 import PdfReader

from ingest_pdf import ingest_pdfs
from ingest_web import ingest_web_page
from llm_client import llm_chat
from recommender import RecommendationResult, retrieve_recommended_context

MAX_PREVIEW_CHARS = 260
COLLECTION_NAME = "research_docs"

RUNTIME_BASE_DIR = Path("runtime_sessions")
RUNTIME_PROCESS_ID = f"run_{os.getpid()}"

SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Use the provided context documents to answer the question. "
    "If the context is insufficient, say you are unsure instead of guessing."
)


def _initialize_runtime_root() -> Path:
    base_dir = RUNTIME_BASE_DIR.resolve()
    process_dir = base_dir / RUNTIME_PROCESS_ID

    if os.environ.get("RC_RUNTIME_READY") != "1":
        base_dir.mkdir(parents=True, exist_ok=True)

        # On a new runtime start, clear prior runtime folders to avoid stale user data.
        for child in base_dir.iterdir():
            if child.is_dir() and child.name != RUNTIME_PROCESS_ID:
                shutil.rmtree(child, ignore_errors=True)

        process_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RC_RUNTIME_READY"] = "1"
    else:
        process_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("RC_RUNTIME_ATEXIT") != "1":
        def _cleanup_process_runtime():
            shutil.rmtree(process_dir, ignore_errors=True)

        atexit.register(_cleanup_process_runtime)
        os.environ["RC_RUNTIME_ATEXIT"] = "1"

    return process_dir


def _get_session_paths(runtime_root: Path) -> tuple[str, Path, Path, Path]:
    if "rc_session_id" not in st.session_state:
        st.session_state["rc_session_id"] = uuid.uuid4().hex[:12]

    if "rc_uploader_nonce" not in st.session_state:
        st.session_state["rc_uploader_nonce"] = 0

    session_id = st.session_state["rc_session_id"]
    session_root = runtime_root / "sessions" / session_id
    pdf_dir = session_root / "pdfs"
    chroma_dir = session_root / "chroma_db"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    return session_id, session_root, pdf_dir, chroma_dir


def _reset_session_data(session_root: Path, chroma_dir: Path):
    # Explicitly purge vector collection before deleting files.
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    except Exception:
        pass

    for _ in range(4):
        try:
            shutil.rmtree(session_root)
            break
        except FileNotFoundError:
            break
        except Exception:
            # Chroma internals may briefly keep file handles open.
            time.sleep(0.2)

    st.session_state["rc_session_id"] = uuid.uuid4().hex[:12]
    st.session_state["rc_uploader_nonce"] = st.session_state.get("rc_uploader_nonce", 0) + 1


def _inject_sidebar_styles():
    st.markdown(
        """
<style>
.upload-card {
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  padding: 0.75rem;
  margin-bottom: 0.65rem;
  background: linear-gradient(150deg, #f8fafc 0%, #eef2ff 100%);
}
.upload-title {
  font-size: 0.9rem;
  font-weight: 700;
  color: #1e293b;
}
.upload-meta {
  font-size: 0.75rem;
  color: #334155;
  margin-top: 0.2rem;
}
.upload-snippet {
  font-size: 0.76rem;
  color: #1f2937;
  margin-top: 0.45rem;
  line-height: 1.3;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "uploaded.pdf")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"


def _extract_preview(pdf_bytes: bytes) -> dict[str, Any]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        first_page_text = ""
        if page_count > 0:
            first_page_text = (reader.pages[0].extract_text() or "").strip()
        normalized = " ".join(first_page_text.split())
        snippet = normalized[:MAX_PREVIEW_CHARS]
        if len(normalized) > MAX_PREVIEW_CHARS:
            snippet += "..."
        if not snippet:
            snippet = "No extractable text preview from the first page."
        return {"pages": page_count, "snippet": snippet, "error": None}
    except Exception as exc:
        return {
            "pages": 0,
            "snippet": "Preview unavailable for this file.",
            "error": str(exc),
        }


def _render_uploaded_pdf_previews(uploaded_files: list[Any]):
    st.sidebar.markdown("#### Uploaded PDF Preview")

    for up in uploaded_files:
        data = up.getvalue()
        preview = _extract_preview(data)

        title = escape(_sanitize_filename(up.name))
        size = _human_size(up.size if hasattr(up, "size") else len(data))
        pages = preview["pages"]
        snippet = escape(preview["snippet"])

        st.sidebar.markdown(
            f"""
<div class="upload-card">
  <div class="upload-title">📄 {title}</div>
  <div class="upload-meta">{size} | {pages} page(s)</div>
  <div class="upload-snippet">{snippet}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def _save_uploaded_pdfs(uploaded_files: list[Any], pdf_dir: Path) -> dict[str, Any]:
    pdf_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    saved_files: list[str] = []
    unchanged_files: list[str] = []

    for up in uploaded_files:
        raw = up.getvalue()
        name = _sanitize_filename(up.name)
        base, ext = os.path.splitext(name)

        target_name = name
        target_path = pdf_dir / target_name

        if target_path.exists():
            existing_bytes = target_path.read_bytes()

            if existing_bytes == raw:
                unchanged_files.append(target_name)
                saved_paths.append(str(target_path))
                continue

            suffix = hashlib.sha1(raw).hexdigest()[:8]
            target_name = f"{base}_{suffix}{ext}"
            target_path = pdf_dir / target_name

            counter = 1
            while target_path.exists():
                existing_bytes = target_path.read_bytes()
                if existing_bytes == raw:
                    break
                target_name = f"{base}_{suffix}_{counter}{ext}"
                target_path = pdf_dir / target_name
                counter += 1

        if not target_path.exists():
            target_path.write_bytes(raw)
            saved_files.append(target_name)
        else:
            unchanged_files.append(target_name)

        saved_paths.append(str(target_path))

    return {
        "paths": saved_paths,
        "saved_files": saved_files,
        "unchanged_files": unchanged_files,
    }


def answer_with_rag(question: str, chroma_dir: Path) -> tuple[str, RecommendationResult]:
    rec = retrieve_recommended_context(
        question,
        k=4,
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
            "No context documents are currently indexed or retrievable.\n\n"
            f"User question: {question}\n\n"
            "State that you are unsure and ask the user to ingest documents."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return llm_chat(messages), rec


def _render_pdf_ingest_result(result: dict):
    st.sidebar.success(
        "PDF ingestion complete. "
        f"Ingested {result['files_ingested']}/{result['files_total']} files, "
        f"added {result['chunks_added']} chunks."
    )

    if result["skipped"]:
        st.sidebar.warning(
            f"Skipped {len(result['skipped'])} file(s) with no extractable text."
        )

    if result["failed"]:
        st.sidebar.error(f"Failed {len(result['failed'])} file(s).")
        with st.sidebar.expander("PDF ingestion errors"):
            for item in result["failed"]:
                st.write(f"- {item['file']}: {item['error']}")


def _render_upload_ingest_result(save_report: dict[str, Any], ingest_result: dict[str, Any]):
    st.sidebar.success(
        "Uploaded PDFs processed. "
        f"Saved {len(save_report['saved_files'])}, "
        f"reused {len(save_report['unchanged_files'])}."
    )
    _render_pdf_ingest_result(ingest_result)


def _render_retrieval_debug(rec: RecommendationResult):
    if not rec.chunks:
        st.info("No retrievable context found in this session's vector DB.")
        return

    with st.expander("Retrieved Context (Recommender)", expanded=False):
        st.write(
            "Candidates: "
            f"vector={rec.stats['vector_candidates']}, "
            f"sparse={rec.stats['sparse_candidates']}, "
            f"selected={rec.stats['selected']}"
        )

        for idx, chunk in enumerate(rec.chunks, start=1):
            st.markdown(
                f"**{idx}. {chunk.label}**  "
                f"score={chunk.score} | vector={chunk.vector_score} | lexical={chunk.lexical_score}"
            )


def main():
    st.set_page_config(page_title="Mermaid", page_icon="🧜")
    _inject_sidebar_styles()

    runtime_root = _initialize_runtime_root()
    session_id, session_root, pdf_dir, chroma_dir = _get_session_paths(runtime_root)

    st.title("Mermaid (RAG + Recommender)")

    st.sidebar.header("Session")
    st.sidebar.caption(f"Session ID: `{session_id}`")
    st.sidebar.caption(
        "Data is isolated per session. Use logout to delete this user's PDFs/URLs. "
        "Previous runtime data is cleaned on server restart."
    )

    if st.sidebar.button("Logout + Delete My Data"):
        with st.sidebar:
            with st.spinner("Deleting session data..."):
                _reset_session_data(session_root, chroma_dir=chroma_dir)
        st.sidebar.success("Session data deleted.")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Ingestion")

    uploader_key = f"pdf_uploader_{st.session_state['rc_uploader_nonce']}"
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key=uploader_key,
        help="Upload PDFs for this session only. Files are saved to this session storage.",
    )

    if uploaded_files:
        _render_uploaded_pdf_previews(uploaded_files)

    if st.sidebar.button("Save + Ingest Uploaded PDFs"):
        if not uploaded_files:
            st.sidebar.warning("Upload at least one PDF first.")
        else:
            with st.sidebar:
                with st.spinner("Saving and ingesting uploaded PDFs..."):
                    try:
                        save_report = _save_uploaded_pdfs(uploaded_files, pdf_dir=pdf_dir)
                        ingest_result = ingest_pdfs(
                            file_paths=save_report["paths"],
                            pdf_dir=str(pdf_dir),
                            chroma_dir=str(chroma_dir),
                            collection_name=COLLECTION_NAME,
                        )
                    except Exception as exc:
                        st.error(f"Uploaded PDF ingestion failed: {exc}")
                    else:
                        _render_upload_ingest_result(save_report, ingest_result)

    if st.sidebar.button("Ingest PDFs from this session folder"):
        with st.sidebar:
            with st.spinner("Ingesting PDFs from this session..."):
                try:
                    result = ingest_pdfs(
                        pdf_dir=str(pdf_dir),
                        chroma_dir=str(chroma_dir),
                        collection_name=COLLECTION_NAME,
                    )
                except Exception as exc:
                    st.error(f"PDF ingestion failed: {exc}")
                else:
                    _render_pdf_ingest_result(result)

    web_url = st.sidebar.text_input("Web URL to ingest", "")
    if st.sidebar.button("Ingest this web page"):
        if not web_url.strip():
            st.sidebar.warning("Enter a URL first.")
        else:
            with st.sidebar:
                with st.spinner("Ingesting web page..."):
                    try:
                        result = ingest_web_page(
                            web_url,
                            chroma_dir=str(chroma_dir),
                            collection_name=COLLECTION_NAME,
                        )
                    except Exception as exc:
                        st.error(f"Web ingestion failed: {exc}")
                    else:
                        if result.get("status") == "ok":
                            st.success(
                                "Web page ingestion complete. "
                                f"Added {result['chunks_added']} chunks from {result['url']}."
                            )
                        else:
                            st.warning(
                                f"Skipped URL ({result['url']}): {result.get('reason', 'No text found')}"
                            )

    st.markdown(
        "Ask questions grounded in your session's uploaded PDFs and ingested web pages. "
        "Answers use a recommender reranking layer before generation."
    )

    question = st.text_area("Your question", height=120)

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            try:
                answer, rec = answer_with_rag(question.strip(), chroma_dir=chroma_dir)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                return

        st.subheader("Answer")
        st.write(answer)
        _render_retrieval_debug(rec)


if __name__ == "__main__":
    main()
