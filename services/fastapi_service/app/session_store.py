import os
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = "research_docs"


def _can_write_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _resolve_data_root() -> Path:
    configured = os.getenv("RESEARCH_COPILOT_DATA_ROOT") or os.getenv("RC_DATA_ROOT")

    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        [
            BASE_DIR / "data",
            Path.home() / ".research_copilot" / "fastapi_data",
            Path(tempfile.gettempdir()) / "research_copilot_fastapi_data",
        ]
    )

    for candidate in candidates:
        if _can_write_dir(candidate):
            return candidate

    joined = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"No writable data directory available. Checked: {joined}")


def _resolve_staging_root() -> Path:
    configured = os.getenv("RESEARCH_COPILOT_STAGING_ROOT") or os.getenv("RC_STAGING_ROOT")

    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        [
            Path(tempfile.gettempdir()) / "research_copilot_upload_staging",
            Path.home() / ".research_copilot" / "upload_staging",
        ]
    )

    for candidate in candidates:
        if _can_write_dir(candidate):
            return candidate

    joined = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"No writable staging directory available. Checked: {joined}")


DATA_ROOT = _resolve_data_root()
USERS_ROOT = DATA_ROOT / "users"
STAGING_ROOT = _resolve_staging_root()


def ensure_roots() -> None:
    USERS_ROOT.mkdir(parents=True, exist_ok=True)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)


def sanitize_user_id(user_id: str) -> str:
    safe = "".join(ch for ch in user_id if ch.isalnum() or ch in {"-", "_"})
    if not safe:
        raise ValueError("invalid user id")
    return safe


def user_root(user_id: str) -> Path:
    safe = sanitize_user_id(user_id)
    root = USERS_ROOT / safe
    root.mkdir(parents=True, exist_ok=True)
    return root


def user_pdf_dir(user_id: str) -> Path:
    path = user_root(user_id) / "pdfs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_chroma_dir(user_id: str) -> Path:
    path = user_root(user_id) / "chroma_db"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_user(user_id: str) -> None:
    root = USERS_ROOT / sanitize_user_id(user_id)
    shutil.rmtree(root, ignore_errors=True)


def validate_staged_pdf_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    root = STAGING_ROOT.resolve()

    if root != path and root not in path.parents:
        raise ValueError("Staged file path is outside the allowed staging root")
    if not path.exists() or not path.is_file():
        raise ValueError(f"Staged file does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only staged PDF files are supported: {path.name}")

    return path


def list_user_data_counts(user_id: str) -> dict[str, int]:
    root = USERS_ROOT / sanitize_user_id(user_id)
    if not root.exists():
        return {"pdf_files": 0, "pdf_bytes": 0, "bytes": 0}

    total_bytes = 0
    for p in root.rglob("*"):
        if p.is_file():
            total_bytes += p.stat().st_size

    pdf_dir = root / "pdfs"
    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    pdf_bytes = sum(path.stat().st_size for path in pdf_files if path.is_file())

    return {
        "pdf_files": len(pdf_files),
        "pdf_bytes": pdf_bytes,
        "bytes": total_bytes,
    }
