# Mini RAG System — Retrieval-Augmented Generation

## Overview

This project implements a mini Retrieval-Augmented Generation (RAG) system that demonstrates how to:

- Chunk documents into smaller pieces
- Store and search embeddings with FAISS
- Retrieve relevant context and generate grounded answers
- Use either mock embeddings/LLM (zero-cost) or real OpenAI embeddings/LLM (toggle-based)

It is designed to showcase practical skills in AI engineering and knowledge-grounded systems.

---

## Project Structure

```
mini-rag/
├── data/           # Source text files
├── docs/           # Documentation (currently empty)
├── embeddings/     # Optional directory for storing real embeddings
├── src/
│ ├── ingest.py     # Ingests and chunks documents
│ ├── search.py     # Builds FAISS index, searches for relevant chunks
│ ├── rag.py        # Full RAG pipeline (mock/OpenAI toggle)
├── requirements.txt
├── .gitignore
└── README.md
```

- Empty directories include `.gitkeep` to preserve structure for future expansion.  

---

## Setup

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd mini-rag
```
2. **Create a Python virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. **Install dependencies**

```bash
pip install -r requirements.txt
```
4. **Environment variables (for OpenAI)**

- Create a .env file in the root directory:
```bash
OPENAI_API_KEY=your_key_here
```
- If you don’t have an API key or want zero-cost testing, leave it empty and set ```USE_OPENAI = False``` in search.py and rag.py.

## Usage

1. **Ingest documents**

```bash
python3 src/ingest.py
```

2. **Build FAISS index and search chunks**

```
python3 src/search.py
```
- Supports a mock/OpenAI toggle:

```python
USE_OPENAI = False  # zero-cost mode
USE_OPENAI = True   # use real OpenAI embeddings (requires API key)
```

3. **Run full RAG pipeline**

```
python3 src/rag.py
```
- Retrieves relevant chunks and generates a grounded answer

- Uses mock LLM by default (zero-cost)

- Switch to OpenAI by toggling ```USE_OPENAI = True``` and providing your API key

## Features

- Fully modular: ingestion, search, RAG pipeline

- Mock LLM and embeddings for zero-cost testing

- Ready to switch to OpenAI embeddings and GPT LLM

- FAISS-based vector search for fast retrieval

- Clear directory structure for future expansions

## Notes

- **USE_OPENAI = False** runs entirely locally, zero-cost

- **USE_OPENAI = True** requires OpenAI API key, consumes tokens

- FAISS index is stored locally as ```faiss_index.pkl```

- Empty directories are placeholders for future features

## Future Improvements

- Add more documents to ```data/``` to test search and RAG performance

- Integrate real OpenAI embeddings and GPT completions when API key is available

- Expand RAG system with caching embeddings, long-context handling, or more advanced retrieval strategies