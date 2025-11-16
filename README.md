# LangChain RAG

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and HuggingFace embedding models. It loads PDF documents, chunks them, generates and stores embeddings, and answers queries by retrieving the most relevant context and sending it to a language model.

## Overview

The project is built around three components:

1. PDF loading and chunking  
2. Embedding generation and vector storage (ChromaDB)  
3. A RAG pipeline using LangChain Runnables and a chat-based LLM  

Everything is designed to be simple, modular, and easy to extend.

## Document Loading

PDFs are loaded using `DirectoryLoader` and split into chunks via `RecursiveCharacterTextSplitter` to preserve paragraph structure. The function returns a list of LangChain `Document` objects.

## Embeddings and Vector Store

Embeddings are generated with `intfloat/multilingual-e5-small` using `HuggingFaceEmbeddings`.  
CUDA is used automatically if available.

A Chroma collection is created or updated:

```python
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_dir,
)
