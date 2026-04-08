import os
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Explicitly load .env from project root
load_dotenv(ENV_PATH)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HOST_FRAGMENT = "openrouter.ai"


def _normalize_base_url(raw_url: str) -> str:
    url = raw_url.rstrip("/")
    parsed = urlparse(url)

    if OPENROUTER_HOST_FRAGMENT in parsed.netloc:
        path = parsed.path.rstrip("/")

        if path in {"", "/", "/api"}:
            path = "/api/v1"
        elif path == "/api/api/v1":
            path = "/api/v1"
        elif path.startswith("/api/v"):
            # Already versioned.
            pass
        elif "/api/v" in path:
            # Already contains a version path segment.
            pass
        else:
            path = f"{path}/api/v1"

        parsed = parsed._replace(path=path, params="", query="", fragment="")
        return urlunparse(parsed).rstrip("/")

    return url


LLM_API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = _normalize_base_url(os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL)

CHAT_MODEL = os.getenv("CHAT_MODEL") or "deepseek/deepseek-v3.2"
_raw_embed_model = os.getenv("EMBED_MODEL") or "openai/text-embedding-3-small"
if OPENROUTER_HOST_FRAGMENT in BASE_URL and "/" not in _raw_embed_model:
    EMBEDDING_MODEL = f"openai/{_raw_embed_model}"
else:
    EMBEDDING_MODEL = _raw_embed_model

APP_URL = os.getenv("APP_URL", "http://localhost")
APP_TITLE = os.getenv("APP_TITLE", "Mermaid")

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


def _common_headers() -> dict[str, str]:
    if not LLM_API_KEY:
        raise RuntimeError(
            "LLM_API_KEY is not set. "
            "Make sure .env (in project root) contains:\n"
            "LLM_API_KEY=sk-...\n"
        )

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    # OpenRouter supports these optional headers for attribution/routing quality.
    if OPENROUTER_HOST_FRAGMENT in BASE_URL:
        headers["HTTP-Referer"] = APP_URL
        headers["X-Title"] = APP_TITLE

    return headers


def _try_parse_json_response(resp: requests.Response, endpoint: str) -> dict[str, Any]:
    try:
        return resp.json()
    except ValueError as exc:
        snippet = (resp.text or "")[:800]
        ctype = resp.headers.get("content-type")
        raise RuntimeError(
            f"Expected JSON response from {endpoint} but got content-type={ctype}. "
            f"Body starts with: {snippet}"
        ) from exc


def _post_json(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    headers = _common_headers()

    last_error: Exception | None = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt < API_MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 4))
                continue
            raise RuntimeError(f"Request failed after retries: {exc}") from exc

        if resp.status_code == 200:
            return _try_parse_json_response(resp, endpoint)

        # Retry common transient failures.
        if resp.status_code in {429, 500, 502, 503, 504} and attempt < API_MAX_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 4))
            continue

        msg = resp.text[:1000]
        raise RuntimeError(
            f"API call failed ({resp.status_code}) at {endpoint}. "
            f"Response: {msg}"
        )

    if last_error is not None:
        raise RuntimeError(f"API call failed after retries: {last_error}")

    raise RuntimeError(f"API call failed after retries for endpoint {endpoint}")


def llm_chat(messages: list[dict], model: str | None = None) -> str:
    model = model or CHAT_MODEL

    payload = {
        "model": model,
        "messages": messages,
    }

    data = _post_json("/chat/completions", payload)

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Chat response parsing failed. Response: {data}") from exc


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []

    model = model or EMBEDDING_MODEL
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        payload = {
            "model": model,
            "input": batch,
        }

        data = _post_json("/embeddings", payload)
        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError(f"Embedding response missing list 'data': {data}")

        # Preserve returned order using the optional index field when present.
        items_sorted = sorted(items, key=lambda x: x.get("index", 0))

        for item in items_sorted:
            emb = item.get("embedding")
            if emb is None:
                raise RuntimeError(f"Invalid embedding item in response: {item}")
            embeddings.append(emb)

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
        )

    return embeddings
