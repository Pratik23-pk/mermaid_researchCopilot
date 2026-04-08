import os
import shutil

import chromadb


def _is_recoverable_chroma_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "no such table: tenants",
        "database disk image is malformed",
        "schema",
        "readonly database",
        "attempt to write a readonly database",
        "unable to open database file",
        "query error",
    )
    return any(marker in msg for marker in markers)


def _reset_chroma_dir(chroma_dir: str) -> None:
    if os.path.isdir(chroma_dir):
        shutil.rmtree(chroma_dir, ignore_errors=True)
    os.makedirs(chroma_dir, exist_ok=True)


def get_collection(chroma_dir: str, collection_name: str):
    os.makedirs(chroma_dir, exist_ok=True)

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        return client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )
    except Exception as exc:
        if not _is_recoverable_chroma_error(exc):
            raise

    # Recover from corrupted/incompatible local Chroma sqlite state.
    _reset_chroma_dir(chroma_dir)
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
    )
