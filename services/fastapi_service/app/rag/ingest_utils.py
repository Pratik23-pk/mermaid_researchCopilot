# ingest_utils.py

import os
from collections.abc import Iterator

DEFAULT_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "1600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "250"))


def iter_chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[str]:
    """
    Yield chunks by characters with overlap while guaranteeing forward progress.

    The iterator form lets ingestion pipelines flush chunks in batches instead of
    materializing a whole document's chunk list in memory at once.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return

    start = 0
    length = len(text)

    while start < length:
        hard_end = min(start + chunk_size, length)
        end = hard_end

        # Try to break on a natural boundary for better chunk quality.
        if hard_end < length:
            window = text[start:hard_end]
            for sep in ("\n\n", "\n", ". ", " "):
                split_at = window.rfind(sep)
                if split_at > int(chunk_size * 0.4):
                    end = start + split_at + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        if end >= length:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            # Safety fallback to guarantee progress.
            next_start = start + (chunk_size - chunk_overlap)
        start = next_start


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    return list(iter_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
