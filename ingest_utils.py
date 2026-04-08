# ingest_utils.py


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Chunk text by characters with overlap while guaranteeing forward progress.

    This avoids infinite loops and works for large web pages/PDF extractions.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
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
            chunks.append(chunk)

        if end >= length:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            # Safety fallback to guarantee progress.
            next_start = start + (chunk_size - chunk_overlap)
        start = next_start

    return chunks
