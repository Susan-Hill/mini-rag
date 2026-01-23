import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

CHUNK_SIZE = 200        # Number of words per chunk

# Load all .txt files from data directory
def load_documents(data_dir):
    
    documents = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents[filename] = f.read()
    return documents

# Split text into chunks and return list of chunks 
def chunk_text(text, chunk_size=CHUNK_SIZE):
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Ingest documents and produce chunks with metadata
def ingest_documents(data_dir):
    
    documents = load_documents(data_dir)
    all_chunks = []

    for filename, content in documents.items():
        chunks = chunk_text(content)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "source": filename,
                "chunk_index": idx,
                "text": chunk
            })
    return all_chunks

if __name__ == "__main__":
    chunks = ingest_documents(DATA_DIR)
    print(f"Ingested {len(chunks)} chunks from documents in '{DATA_DIR}'\n")

    # Print first 3 chunks for inspection
    for chunk in chunks[:3]:
        print(f"Source: {chunk['source']}, Chunk: {chunk['chunk_index']}")
        print(chunk['text'])
        print("-" * 50)