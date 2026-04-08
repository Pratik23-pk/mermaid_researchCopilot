import os
import re
from dataclasses import dataclass
from typing import Any

import chromadb

from llm_client import embed_texts

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "research_docs"

DEFAULT_TOP_K = int(os.getenv("RECOMMENDER_TOP_K", "4"))
DEFAULT_MIN_TOP_K = int(os.getenv("RECOMMENDER_MIN_TOP_K", "2"))
DEFAULT_VECTOR_CANDIDATES = int(os.getenv("RECOMMENDER_VECTOR_CANDIDATES", "24"))
DEFAULT_SPARSE_CANDIDATES = int(os.getenv("RECOMMENDER_SPARSE_CANDIDATES", "16"))
DEFAULT_MAX_SPARSE_SCAN = int(os.getenv("RECOMMENDER_MAX_SPARSE_SCAN", "1200"))
DEFAULT_MAX_PER_SOURCE = int(os.getenv("RECOMMENDER_MAX_PER_SOURCE", "2"))
DEFAULT_MAX_CONTEXT_CHARS = int(os.getenv("RECOMMENDER_MAX_CONTEXT_CHARS", "2800"))
DEFAULT_MAX_CHUNK_CHARS = int(os.getenv("RECOMMENDER_MAX_CHUNK_CHARS", "680"))
DEFAULT_MIN_CHUNK_CHARS = int(os.getenv("RECOMMENDER_MIN_CHUNK_CHARS", "220"))

TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
SEPARATOR = "\n\n---\n\n"

SOURCE_WEIGHT = {
    "pdf": 1.05,
    "corpus": 1.0,
    "web": 0.95,
}


@dataclass
class RankedChunk:
    chunk_id: str
    label: str
    document: str
    metadata: dict[str, Any]
    score: float
    vector_score: float
    lexical_score: float


@dataclass
class RecommendationResult:
    context: str
    chunks: list[RankedChunk]
    stats: dict[str, Any]


@dataclass
class _Candidate:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    label: str
    vector_score: float = 0.0
    lexical_score: float = 0.0
    final_score: float = 0.0


def _tokenize(text: str) -> set[str]:
    return set(tok.lower() for tok in TOKEN_RE.findall(text or ""))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_label(meta: dict[str, Any]) -> str:
    return str(meta.get("filename") or meta.get("url") or "unknown")


def _get_collection(chroma_dir: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
    )


def _context_block(label: str, document: str) -> str:
    return f"[{label}]\n{document}"


def _dynamic_top_k(query_tokens: set[str], k: int, min_k: int) -> int:
    token_count = len(query_tokens)
    floor = max(1, min(min_k, k))

    if token_count <= 4:
        return min(k, max(floor, 2))
    if token_count <= 8:
        return min(k, max(floor, 3))
    if token_count <= 14:
        return min(k, max(floor, 4))
    return max(floor, k)


def _truncate_to_boundary(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    text = _normalize_whitespace(text)
    if len(text) <= max_chars:
        return text

    cutoff = max_chars
    window = text[:max_chars]
    for sep in (". ", "! ", "? ", "; ", ": ", ", ", " "):
        idx = window.rfind(sep)
        if idx > int(max_chars * 0.6):
            cutoff = idx + len(sep.rstrip())
            break

    trimmed = window[:cutoff].strip()
    if not trimmed:
        trimmed = window.strip()
    return f"{trimmed} ..."


def _best_window_start(text: str, query_tokens: set[str], max_chars: int) -> int:
    if not query_tokens or len(text) <= max_chars:
        return 0

    text_lower = text.lower()
    positions: list[int] = []
    for token in query_tokens:
        start = 0
        while True:
            idx = text_lower.find(token, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(token)

    if not positions:
        return 0

    max_start = max(0, len(text) - max_chars)
    best_score: tuple[float, float, int] | None = None
    best_start = 0

    for pos in positions[:80]:
        start = min(max(pos - max_chars // 3, 0), max_start)
        window = text_lower[start : start + max_chars]
        hit_total = sum(window.count(token) for token in query_tokens)
        unique_hits = sum(1 for token in query_tokens if token in window)
        score = (float(unique_hits), float(hit_total), -start)
        if best_score is None or score > best_score:
            best_score = score
            best_start = start

    return best_start


def _compress_document(document: str, query_tokens: set[str], max_chars: int) -> str:
    max_chars = max(80, max_chars)
    normalized = _normalize_whitespace(document)
    if len(normalized) <= max_chars:
        return normalized

    start = _best_window_start(normalized, query_tokens, max_chars=max_chars)
    end = min(len(normalized), start + max_chars)
    excerpt = normalized[start:end].strip()

    if start == 0:
        return _truncate_to_boundary(excerpt, max_chars=max_chars)

    excerpt = _truncate_to_boundary(excerpt, max_chars=max_chars - 4)
    excerpt = excerpt.lstrip(".,;: ")
    return f"... {excerpt}"


def _vector_candidates(collection, query_emb: list[float], n_results: int) -> list[_Candidate]:
    if n_results <= 0:
        return []

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates: list[_Candidate] = []
    for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
        meta = meta or {}
        dist = float(distance) if distance is not None else 1.0
        vector_score = 1.0 / (1.0 + max(dist, 0.0))
        candidates.append(
            _Candidate(
                chunk_id=str(chunk_id),
                document=str(doc or ""),
                metadata=meta,
                label=_safe_label(meta),
                vector_score=vector_score,
            )
        )

    return candidates


def _lexical_score(query_tokens: set[str], doc: str) -> float:
    if not query_tokens:
        return 0.0

    doc_tokens = _tokenize(doc)
    if not doc_tokens:
        return 0.0

    overlap = query_tokens & doc_tokens
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(doc_tokens)
    return 0.75 * coverage + 0.25 * precision


def _sparse_candidates(
    collection,
    query_tokens: set[str],
    top_n: int,
    max_scan: int,
) -> list[_Candidate]:
    if top_n <= 0 or not query_tokens:
        return []

    total = collection.count()
    if total == 0:
        return []

    scan_limit = min(total, max(max_scan, top_n * 180))
    page_size = 200
    scored: list[tuple[float, _Candidate]] = []

    offset = 0
    while offset < scan_limit:
        batch = min(page_size, scan_limit - offset)
        data = collection.get(
            limit=batch,
            offset=offset,
            include=["documents", "metadatas"],
        )

        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        for chunk_id, doc, meta in zip(ids, docs, metas):
            doc = str(doc or "")
            meta = meta or {}
            lexical = _lexical_score(query_tokens, doc)
            if lexical <= 0:
                continue

            scored.append(
                (
                    lexical,
                    _Candidate(
                        chunk_id=str(chunk_id),
                        document=doc,
                        metadata=meta,
                        label=_safe_label(meta),
                        lexical_score=lexical,
                    ),
                )
            )

        offset += batch

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_n]]


def _chunk_position_score(meta: dict[str, Any]) -> float:
    chunk = meta.get("chunk")
    if isinstance(chunk, int):
        return 1.0 / (1.0 + 0.15 * chunk)
    return 0.8


def _length_quality_score(doc: str) -> float:
    n = len(doc)
    if n < 120:
        return 0.4
    if n > 2800:
        return 0.7
    return 1.0


def _candidate_final_score(candidate: _Candidate) -> float:
    source = str(candidate.metadata.get("source") or "")
    source_weight = SOURCE_WEIGHT.get(source, 1.0)

    chunk_score = _chunk_position_score(candidate.metadata)
    length_score = _length_quality_score(candidate.document)

    score = (
        0.58 * candidate.vector_score
        + 0.28 * candidate.lexical_score
        + 0.08 * chunk_score
        + 0.06 * length_score
    )

    return score * source_weight


def _merge_candidates(
    vector_items: list[_Candidate],
    sparse_items: list[_Candidate],
) -> list[_Candidate]:
    merged: dict[str, _Candidate] = {}

    for item in vector_items:
        merged[item.chunk_id] = item

    for item in sparse_items:
        if item.chunk_id in merged:
            merged_item = merged[item.chunk_id]
            merged_item.lexical_score = max(merged_item.lexical_score, item.lexical_score)
        else:
            merged[item.chunk_id] = item

    combined = list(merged.values())
    for item in combined:
        item.final_score = _candidate_final_score(item)

    combined.sort(key=lambda c: c.final_score, reverse=True)
    return combined


def _select_diverse(candidates: list[_Candidate], k: int, max_per_source: int) -> list[_Candidate]:
    if k <= 0:
        return []

    selected: list[_Candidate] = []
    source_count: dict[str, int] = {}

    for cand in candidates:
        count = source_count.get(cand.label, 0)
        if count >= max_per_source:
            continue

        near_duplicate = False
        cand_tokens = _tokenize(cand.document)
        for picked in selected:
            picked_tokens = _tokenize(picked.document)
            if not cand_tokens or not picked_tokens:
                continue
            overlap = len(cand_tokens & picked_tokens) / len(cand_tokens | picked_tokens)
            if overlap > 0.85:
                near_duplicate = True
                break

        if near_duplicate:
            continue

        selected.append(cand)
        source_count[cand.label] = count + 1

        if len(selected) >= k:
            break

    if len(selected) < k:
        seen_ids = {item.chunk_id for item in selected}
        for cand in candidates:
            if cand.chunk_id in seen_ids:
                continue
            selected.append(cand)
            if len(selected) >= k:
                break

    return selected


def _assemble_context_chunks(
    candidates: list[_Candidate],
    query_tokens: set[str],
    *,
    requested_k: int,
    min_k: int,
    max_context_chars: int,
    max_chunk_chars: int,
) -> tuple[list[RankedChunk], int, int]:
    if not candidates:
        return [], 0, 0

    chosen: list[RankedChunk] = []
    total_chars = 0
    effective_k = _dynamic_top_k(query_tokens, k=requested_k, min_k=min_k)
    floor = max(1, min(min_k, effective_k))

    for cand in candidates:
        if len(chosen) >= effective_k:
            break

        separator_cost = len(SEPARATOR) if chosen else 0
        remaining_budget = max_context_chars - total_chars - separator_cost
        if remaining_budget <= 0 and len(chosen) >= floor:
            break

        remaining_slots = max(1, effective_k - len(chosen))
        ideal_doc_budget = max(
            DEFAULT_MIN_CHUNK_CHARS,
            min(max_chunk_chars, remaining_budget // remaining_slots if remaining_budget > 0 else 0),
        )

        block_doc = _compress_document(cand.document, query_tokens, max_chars=ideal_doc_budget)
        block = _context_block(cand.label, block_doc)

        if len(block) > remaining_budget:
            fallback_budget = remaining_budget - len(cand.label) - 3
            if fallback_budget > 80:
                block_doc = _compress_document(
                    cand.document,
                    query_tokens,
                    max_chars=max(80, fallback_budget),
                )
                block = _context_block(cand.label, block_doc)

        if len(block) > remaining_budget:
            if len(chosen) >= floor:
                continue
            block_doc = _compress_document(
                cand.document,
                query_tokens,
                max_chars=max(80, max_context_chars - len(cand.label) - 3),
            )
            block = _context_block(cand.label, block_doc)
            if len(block) + separator_cost > max_context_chars:
                continue

        total_chars += separator_cost + len(block)
        chosen.append(
            RankedChunk(
                chunk_id=cand.chunk_id,
                label=cand.label,
                document=block_doc,
                metadata=cand.metadata,
                score=round(cand.final_score, 4),
                vector_score=round(cand.vector_score, 4),
                lexical_score=round(cand.lexical_score, 4),
            )
        )

    if not chosen:
        first = candidates[0]
        doc_budget = max(80, min(max_chunk_chars, max_context_chars - len(first.label) - 3))
        block_doc = _compress_document(first.document, query_tokens, max_chars=doc_budget)
        chosen.append(
            RankedChunk(
                chunk_id=first.chunk_id,
                label=first.label,
                document=block_doc,
                metadata=first.metadata,
                score=round(first.final_score, 4),
                vector_score=round(first.vector_score, 4),
                lexical_score=round(first.lexical_score, 4),
            )
        )
        total_chars = len(_context_block(first.label, block_doc))

    return chosen, total_chars, effective_k


def retrieve_recommended_context(
    query: str,
    k: int = DEFAULT_TOP_K,
    min_k: int = DEFAULT_MIN_TOP_K,
    vector_candidates: int = DEFAULT_VECTOR_CANDIDATES,
    sparse_candidates: int = DEFAULT_SPARSE_CANDIDATES,
    max_sparse_scan: int = DEFAULT_MAX_SPARSE_SCAN,
    max_per_source: int = DEFAULT_MAX_PER_SOURCE,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> RecommendationResult:
    collection = _get_collection(chroma_dir=chroma_dir, collection_name=collection_name)
    total_docs = collection.count()
    if total_docs == 0:
        return RecommendationResult(
            context="",
            chunks=[],
            stats={
                "total_docs": 0,
                "vector_candidates": 0,
                "sparse_candidates": 0,
                "selected": 0,
            },
        )

    query_emb = embed_texts([query])[0]
    query_tokens = _tokenize(query)

    vector_n = min(max(vector_candidates, k * 8), total_docs)

    vector_items = _vector_candidates(collection, query_emb, vector_n)
    sparse_items = _sparse_candidates(
        collection,
        query_tokens,
        top_n=min(max(sparse_candidates, k * 6), total_docs),
        max_scan=max_sparse_scan,
    )

    merged = _merge_candidates(vector_items, sparse_items)
    diverse = _select_diverse(merged, k=k, max_per_source=max_per_source)
    ranked_chunks, context_chars, effective_k = _assemble_context_chunks(
        diverse,
        query_tokens,
        requested_k=k,
        min_k=min_k,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
    )

    blocks = [_context_block(chunk.label, chunk.document) for chunk in ranked_chunks]

    return RecommendationResult(
        context=SEPARATOR.join(blocks),
        chunks=ranked_chunks,
        stats={
            "total_docs": total_docs,
            "vector_candidates": len(vector_items),
            "sparse_candidates": len(sparse_items),
            "selected_before_budget": len(diverse),
            "selected": len(ranked_chunks),
            "top_k_requested": k,
            "top_k_effective": effective_k,
            "min_top_k": min_k,
            "max_per_source": max_per_source,
            "context_budget_chars": max_context_chars,
            "context_chars": context_chars,
        },
    )
