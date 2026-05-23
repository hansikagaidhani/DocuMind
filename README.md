---
title: DocuMind
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🧠 DocuMind — Chat with Your Documents

> Upload any PDF or text document. Ask questions in plain English. Get answers grounded in your content — no hallucinations.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/HansikaG/documind)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)](https://langchain.com)

---

## What It Does

DocuMind is a **production-ready RAG (Retrieval-Augmented Generation)** application that lets users upload documents and have a conversation with them. The LLM answers strictly from the document content, eliminating hallucinations.

**Key capabilities:**
- Upload PDF, TXT, or Markdown files
- Semantic search over document chunks using FAISS vector store
- Answers cite the exact pages and context chunks retrieved
- Supports multiple LLM models via HuggingFace Serverless Inference API
- Session-based — supports multiple users simultaneously

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Convert text chunks to dense vectors |
| Vector Store | FAISS (Facebook AI Similarity Search) | Fast nearest-neighbour retrieval |
| Orchestration | LangChain + LCEL | Document loading, chunking, pipeline |
| LLM | HuggingFace Serverless Inference API | Answer generation (Qwen, Mistral, Zephyr) |
| Backend | FastAPI + Uvicorn | REST API with session management |
| Frontend | Streamlit | Interactive chat UI |
| Deployment | HuggingFace Spaces (Docker) | Cloud hosting |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   INGESTION                      │
│                                                  │
│  PDF/TXT  →  PyPDFLoader  →  Text Splitter       │
│              (500 token chunks, 50 overlap)      │
│                    ↓                             │
│         HuggingFace Embeddings (MiniLM)          │
│                    ↓                             │
│            FAISS Vector Store                    │
└─────────────────────────────────────────────────┘
                     ↓  session stored in memory
┌─────────────────────────────────────────────────┐
│                   RETRIEVAL                      │
│                                                  │
│  User Question  →  Embed Question               │
│                    ↓                             │
│         FAISS Top-4 Similarity Search            │
│                    ↓                             │
│  Retrieved Chunks + Question → LLM Prompt        │
│                    ↓                             │
│       HuggingFace Inference API (LLM)            │
│                    ↓                             │
│         Grounded Answer + Page Citations         │
└─────────────────────────────────────────────────┘
```

---

## Why RAG? (For Recruiters)

Standard LLMs hallucinate — they generate plausible-sounding but incorrect answers when they don't know something. RAG fixes this by:

1. **Chunking** — splitting the document into overlapping text segments
2. **Embedding** — converting each chunk into a vector that captures its meaning
3. **Indexing** — storing all vectors in FAISS for millisecond-speed search
4. **Retrieval** — when a question arrives, embed it and find the top-4 most semantically similar chunks
5. **Generation** — pass only those chunks as context to the LLM, instructing it to answer from context only
6. **Grounding** — the LLM cannot go beyond what the document says

This pattern is the backbone of most enterprise AI assistants built today.

---

## Run Locally

```bash
git clone https://github.com/hansikagaidhani/DocuMind.git
cd DocuMind

# Install dependencies
pip install -r requirements.txt

# Set your HuggingFace token
cp .env.example .env
# Edit .env and add your HUGGINGFACEHUB_API_TOKEN

# Terminal 1 — start the backend
uvicorn api:app --reload --port 8000

# Terminal 2 — start the frontend
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + active session count |
| `POST` | `/upload` | Upload & index a document |
| `POST` | `/query` | Ask a question against a session |
| `DELETE` | `/session/{id}` | Clear a session |

Interactive docs available at **http://localhost:8000/docs**

---

## Project Structure

```
DocuMind/
├── app.py              # Streamlit frontend
├── api.py              # FastAPI backend
├── rag_pipeline.py     # RAG logic (embeddings, FAISS, LLM)
├── requirements.txt
├── Dockerfile          # HuggingFace Spaces Docker config
├── start.sh            # Launches both services
└── .streamlit/
    └── config.toml     # Streamlit server config
```

---

## Live Demo

Try it live on HuggingFace Spaces — no setup required:

**[huggingface.co/spaces/HansikaG/documind](https://huggingface.co/spaces/HansikaG/documind)**

> Bring your own HuggingFace API token (free at huggingface.co/settings/tokens) or the Space uses its own if configured.
