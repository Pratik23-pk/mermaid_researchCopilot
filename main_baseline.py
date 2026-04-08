from llm_client import llm_chat

SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Explain concepts clearly with short paragraphs and bullet points when helpful. "
    "If you are unsure or lack information, say you are unsure."
)

def main():
    print("Mermaid – Baseline (DeepSeek v3.2 via OpenRouter, no RAG yet)")
    print("Type 'exit' to quit.")

    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        answer = llm_chat(messages)
        print("\nAnswer:\n")
        print(answer)

if __name__ == "__main__":
    main()
