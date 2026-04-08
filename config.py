from dotenv import load_dotenv
import os

# Load .env values into environment variables
load_dotenv()

# Chat (DeepSeek v3.2 via OpenRouter)
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or ""
CHAT_MODEL = os.getenv("CHAT_MODEL") or "deepseek/deepseek-v3.2"

# Embeddings (can reuse same key/provider or a different one)
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or LLM_API_KEY
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL") or LLM_BASE_URL
EMBED_MODEL = os.getenv("EMBED_MODEL") or "text-embedding-3-small"

if __name__ == "__main__":
    print("LLM_API_KEY set:", bool(LLM_API_KEY))
    print("LLM_BASE_URL:", LLM_BASE_URL)
    print("CHAT_MODEL:", CHAT_MODEL)
    print("EMBED_API_KEY set:", bool(EMBED_API_KEY))
    print("EMBED_BASE_URL:", EMBED_BASE_URL)
    print("EMBED_MODEL:", EMBED_MODEL)
