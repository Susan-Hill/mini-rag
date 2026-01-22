import os
import pickle
from typing import List, Dict
import numpy as np
import faiss
import hashlib
from dotenv import load_dotenv
from openai import OpenAI

from ingest import ingest_documents

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.pkl")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # fixed size for this model
TOP_K = 3

# Toggle embeddings mode
USE_OPENAI = False      # False = mock embeddings, True = real OpenAI

# Initialize OPENAI client
load_dotenv()
if USE_OPENAI:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI API key missing. Set USE_OPENAI=False or provide a key.")

# Mock embedding function
def mock_embedding(text: str, dim=EMBEDDING_DIM) -> np.ndarray:
    
    # Generate deterministic fake embedding from text for testing.
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    rng = np.random.default_rng(h % (2**32))
    return rng.random(dim).astype("float32")

# Embedding function
def embed_text(text: str) -> np.ndarray:
    if USE_OPENAI:
        # Convert text into a vector embedding using OpenAI.
        response = client.embeddings.create(
            model=EMBEDDING_MODEL, 
            input=text
            )
        return np.array(response.data[0].embedding, dtype="float32")
    else:
        return mock_embedding(text)

# Build vector index
def build_index(chunks: List[Dict]) -> None:
    
    # Create a FAISS index from text chunks and store it on disk.
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = []

    for chunk in chunks:
        vector = embed_text(chunk["text"])
        index.add(np.array([vector], dtype="float32"))

        metadata.append({
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"]
        })

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(
            {
                "index": index,
                "metadata": metadata
            },
            f
        )

    print(f"Vector index built with {len(metadata)} chunks.")

# Load vector index
def load_index():
    
    # Load FAISS index and metadata from disk.
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Run build_index first.")

    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    return data["index"], data["metadata"]

# Search function
def search(query: str, top_k: int = TOP_K) -> List[Dict]:
    
    # Search the vector index for the most relevant chunks.
    index, metadata = load_index()
    query_vector = embed_text(query)
    query_vector = np.array([query_vector], dtype="float32")

    distances, indices = index.search(query_vector, top_k)

    results = []
    for i in indices[0]:
        results.append(metadata[i])

    return results

# CLI test - optional
if __name__ == "__main__":
    # Ingest documents and build index
    chunks = ingest_documents(DATA_DIR)
    build_index(chunks)

    # Run test query
    query = "What is retrieval augmented generation?"
    results = search(query)

    print("\nTop results:\n")
    for r in results:
        print(f"- {r['text'][:200]}...\n")
