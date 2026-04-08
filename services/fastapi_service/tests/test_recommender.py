import unittest
from unittest.mock import patch

from app.rag import recommender


class FakeCollection:
    def __init__(self, docs: list[str], metas: list[dict]) -> None:
        self.docs = docs
        self.metas = metas
        self.ids = [f"id-{idx}" for idx in range(len(docs))]

    def count(self) -> int:
        return len(self.docs)

    def query(self, query_embeddings, n_results, include):  # noqa: ANN001
        limit = min(n_results, len(self.docs))
        ids = self.ids[:limit]
        docs = self.docs[:limit]
        metas = self.metas[:limit]
        distances = [round(idx * 0.07, 4) for idx in range(limit)]
        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances],
        }

    def get(self, limit, offset, include):  # noqa: ANN001
        end = min(offset + limit, len(self.docs))
        return {
            "ids": self.ids[offset:end],
            "documents": self.docs[offset:end],
            "metadatas": self.metas[offset:end],
        }


class RecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            (
                "Overview of mermaid retrieval pipelines. Mermaid ranks evidence before "
                "sending compact context to the language model for grounded answers."
            )
            * 3,
            (
                "Mermaid uses hybrid retrieval, lexical scoring, and reranking to keep "
                "context small while preserving answer quality for research questions."
            )
            * 3,
            (
                "This note covers vector search, chunk filtering, and context compression "
                "for efficient prompting in long-document question answering."
            )
            * 3,
            "General research workflow note without much relation to the question." * 8,
            "Another generic note about unrelated systems and integrations." * 8,
            "Short appendix about Mermaid evidence trails and source selection." * 5,
        ]
        self.metas = [
            {"filename": "doc-a.pdf", "source": "pdf", "chunk": 0},
            {"filename": "doc-a.pdf", "source": "pdf", "chunk": 1},
            {"filename": "doc-b.pdf", "source": "pdf", "chunk": 0},
            {"filename": "doc-c.pdf", "source": "pdf", "chunk": 0},
            {"filename": "doc-d.pdf", "source": "pdf", "chunk": 0},
            {"filename": "doc-e.pdf", "source": "pdf", "chunk": 0},
        ]

    def test_compress_document_focuses_on_query_terms(self) -> None:
        long_doc = (
            "intro filler " * 40
            + "mermaid retrieval keeps context lean and focused for grounded answers. "
            + "tail filler " * 40
        )

        excerpt = recommender._compress_document(long_doc, {"mermaid", "retrieval"}, max_chars=140)

        self.assertIn("mermaid", excerpt.lower())
        self.assertIn("retrieval", excerpt.lower())
        self.assertLessEqual(len(excerpt), 145)

    def test_retrieve_recommended_context_uses_dynamic_top_k_and_budget(self) -> None:
        fake_collection = FakeCollection(self.docs, self.metas)

        with (
            patch.object(recommender, "_get_collection", return_value=fake_collection),
            patch.object(recommender, "embed_texts", side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts]),
        ):
            result = recommender.retrieve_recommended_context(
                "mermaid retrieval",
                k=6,
                min_k=2,
                max_context_chars=900,
                max_chunk_chars=280,
            )

        self.assertEqual(result.stats["top_k_effective"], 2)
        self.assertEqual(result.stats["selected"], 2)
        self.assertLessEqual(result.stats["context_chars"], 900)
        self.assertIn("mermaid", result.context.lower())

    def test_retrieve_recommended_context_preserves_multiple_chunks_with_tight_budget(self) -> None:
        fake_collection = FakeCollection(self.docs, self.metas)

        with (
            patch.object(recommender, "_get_collection", return_value=fake_collection),
            patch.object(recommender, "embed_texts", side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts]),
        ):
            result = recommender.retrieve_recommended_context(
                "mermaid evidence",
                k=5,
                min_k=2,
                max_context_chars=520,
                max_chunk_chars=180,
            )

        self.assertGreaterEqual(result.stats["selected"], 2)
        self.assertLessEqual(result.stats["context_chars"], 520)

    def test_empty_collection_returns_empty_context(self) -> None:
        fake_collection = FakeCollection([], [])

        with patch.object(recommender, "_get_collection", return_value=fake_collection):
            result = recommender.retrieve_recommended_context("anything")

        self.assertEqual(result.context, "")
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.stats["selected"], 0)


if __name__ == "__main__":
    unittest.main()
