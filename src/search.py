import os
import pickle
from typing import List, Dict

import numpy as np
import faiss
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

# Initialize OPENAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Embedding function
def embed_text(text: str) -> List[float]:
   
   # Convert text into a vector embedding using OpenAI.
       response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

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
    chunks = ingest_documents(DATA_DIR)
    build_index(chunks)

    query = "What is retrieval augmented generation?"
    results = search(query)

    print("\nTop results:\n")
    for r in results:
        print(f"- {r['text'][:200]}...\n")
