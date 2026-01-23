import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

from search import search, USE_OPENAI  # use the same toggle
from ingest import ingest_documents


# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

# OpenAI model for completion
LLM_MODEL = "gpt-3.5-turbo"

# Initialize OpenAI client
load_dotenv()
if USE_OPENAI:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI API key missing. Set USE_OPENAI=False or provide a key.")

# Mock LLM function
def mock_llm(context: List[str], query: str) -> str:
    # Return a deterministic fake answer for testing without OpenAI.
    combined_context = " ".join([c[:100] for c in context])
    return (
        f"[MOCK ANSWER]\n"
        f"Query: {query}\n"
        f"Using context: {combined_context}...\n"
        f"This is a placeholder response. Replace USE_OPENAI=True to call the real LLM."
    )

# RAG function
# Retrieve relevant chunks using search, then generate a grounded answer
# using either the real LLM or mock.
def rag_answer(query: str, top_k: int = 3) -> str:
    
    # Retrieve top-k chunks from FAISS index
    results = search(query, top_k=top_k)
    context = [r["text"] for r in results]

    # Call LLM (mock or real)
    if USE_OPENAI:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content
    else:
        return mock_llm(context, query)

# CLI test - optional
if __name__ == "__main__":
    # Ensure documents are ingested and FAISS index is built
    chunks = ingest_documents(DATA_DIR)

    # Example query
    query = "What is retrieval-augmented generation?"
    answer = rag_answer(query)

    print("\nRAG Answer:\n")
    print(answer)
