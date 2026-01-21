# Mini-RAG

## v1 Implementation Details

### Architecture Overview
This version implements a basic RAG pipeline:

1. Document ingestion
2. Text chunking
3. Embedding generation
4. Vector storage
5. Similarity search
6. Context-grounded LLM response

### Tech Stack (v1)
- Language: Python
- Embeddings: TBD
- Vector Store: TBD
- LLM: TBD

### Project Structure
```
src/ingest.py   # document loading + chunking
src/search.py   # vector similarity search
src/rag.py      # RAG prompt + response generation
data/           # raw documents
embeddings/     # stored vectors
docs/           # notes and experiments
```

### Current Status
In development

### Known Limitations
- No evaluation metrics yet
- No caching layer
- Single-user workflow
