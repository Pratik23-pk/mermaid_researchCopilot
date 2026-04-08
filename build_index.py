import argparse
import os

import chromadb

from llm_client import embed_texts

CORPUS_DIR = "corpus"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "research_docs"


def load_corpus() -> tuple[list[str], list[str], list[dict]]:
    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict] = []

    for fname in sorted(os.listdir(CORPUS_DIR)):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(CORPUS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        doc_id = f"corpus::{fname}"
        ids.append(doc_id)
        docs.append(text)
        metadatas.append({"filename": fname, "source": "corpus"})

    return ids, docs, metadatas


def _upsert_documents(collection, ids: list[str], docs: list[str], metadatas: list[dict]):
    try:
        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embed_texts(docs),
        )
        return
    except AttributeError:
        # Older client fallback.
        collection.delete(ids=ids)
        collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embed_texts(docs),
        )


def main(reset_collection: bool = False):
    if not os.path.exists(CORPUS_DIR):
        print(f"{CORPUS_DIR}/ folder not found. Create it and add .txt files first.")
        return

    ids, docs, metadatas = load_corpus()
    if not docs:
        print("No .txt files with content found in corpus/. Add some text files first.")
        return

    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset_collection:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    print(f"Embedding and indexing {len(docs)} corpus documents...")
    _upsert_documents(collection, ids, docs, metadatas)

    total = collection.count()
    print(
        f"Indexed {len(docs)} corpus docs into Chroma at {CHROMA_DIR}. "
        f"Collection total documents: {total}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index corpus text files into Chroma.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before indexing.",
    )
    args = parser.parse_args()

    main(reset_collection=args.reset)
