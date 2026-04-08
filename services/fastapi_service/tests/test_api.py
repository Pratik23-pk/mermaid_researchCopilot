import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.rag.recommender import RankedChunk, RecommendationResult


class FastAPIEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("app.main.ask_question")
    def test_chat_endpoint_returns_answer_and_retrieval(self, ask_question_mock) -> None:
        ask_question_mock.return_value = (
            "Mermaid answer",
            RecommendationResult(
                context="[doc-a.pdf]\nFocused excerpt",
                chunks=[
                    RankedChunk(
                        chunk_id="chunk-1",
                        label="doc-a.pdf",
                        document="Focused excerpt",
                        metadata={"source": "pdf"},
                        score=0.91,
                        vector_score=0.88,
                        lexical_score=0.66,
                    )
                ],
                stats={"selected": 1, "context_chars": 42},
            ),
        )

        response = self.client.post(
            "/chat",
            json={"question": "What does Mermaid do?"},
            headers={"X-User-Id": "42"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "Mermaid answer")
        self.assertEqual(payload["retrieval"][0]["label"], "doc-a.pdf")


if __name__ == "__main__":
    unittest.main()
