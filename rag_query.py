from llm_client import llm_chat
from recommender import retrieve_recommended_context

SYSTEM_PROMPT = (
    "You are Mermaid, a research assistant. "
    "Use the provided context documents to answer the question. "
    "If the context is insufficient, say you are unsure instead of guessing."
)


def answer_with_rag(question: str) -> tuple[str, list[dict]]:
    rec = retrieve_recommended_context(question, k=4)

    if rec.context:
        context_prompt = (
            f"Context documents:\n{rec.context}\n\n"
            f"User question: {question}\n\n"
            "Answer using only the information from the context when possible. "
            "If something is not in the context, say you are unsure."
        )
    else:
        context_prompt = (
            "No context documents are currently indexed or retrievable.\n\n"
            f"User question: {question}\n\n"
            "State that you are unsure and ask the user to ingest documents."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_prompt},
    ]

    answer = llm_chat(messages)

    retrieval_info = [
        {
            "label": c.label,
            "score": c.score,
            "vector": c.vector_score,
            "lexical": c.lexical_score,
        }
        for c in rec.chunks
    ]

    return answer, retrieval_info


def main():
    print("Mermaid – RAG + Recommender mode")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        answer, retrieval_info = answer_with_rag(q)
        print("\nAnswer:\n")
        print(answer)

        if retrieval_info:
            print("\nRetrieved Context:")
            for i, info in enumerate(retrieval_info, start=1):
                print(
                    f"{i}. {info['label']} "
                    f"(score={info['score']}, vector={info['vector']}, lexical={info['lexical']})"
                )


if __name__ == "__main__":
    main()
