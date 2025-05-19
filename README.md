# RAG Demo with Qdrant

This is a minimal setup for demonstrating Retrieval-Augmented Generation (RAG) using a Qdrant vector database and FastEmbed for embeddings.

## Requirements

```bash
pip install qdrant-client fastembed langchain
```

## Run Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

See the `__main__` block or import `QdrantDB` to embed texts, store vectors, and perform similarity search.
