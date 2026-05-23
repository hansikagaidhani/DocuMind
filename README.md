---
title: DocuMind
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🧠 DocuMind

Upload any document. Ask anything. Get answers **grounded in your content**.

🚀 **Live Demo:** [huggingface.co/spaces/HansikaG/documind](https://huggingface.co/spaces/HansikaG/documind)

## Stack

| Layer | Technology |
|---|---|
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Orchestration | LangChain (LCEL) |
| LLM | HuggingFace Serverless Inference API (Mistral-7B, Zephyr, Falcon) |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Deployment | HuggingFace Spaces (Docker) |

## Architecture

```
User uploads PDF
      │
      ▼
PyPDFLoader → RecursiveCharacterTextSplitter (500 tokens, 50 overlap)
      │
      ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Store  ◄──── persisted in session
      │
User asks question
      │
      ▼
Retriever (top-4 similar chunks) → PromptTemplate → LLM → Answer
```

## Run Locally

```bash
git clone https://github.com/hansikagaidhani/DocuMind.git
cd DocuMind

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env and add your HuggingFace token

# Terminal 1: Start FastAPI backend
uvicorn api:app --reload --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Push this repo to the Space
4. Add `HUGGINGFACEHUB_API_TOKEN` in **Settings → Variables and Secrets**

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/upload` | Upload & index a document |
| POST | `/query` | Ask a question |
| DELETE | `/session/{id}` | Clear a session |

## How RAG Works (Interview Cheat Sheet)

1. **Ingestion**: Document split into overlapping chunks
2. **Embedding**: Each chunk → dense vector via sentence-transformer
3. **Indexing**: Vectors stored in FAISS for fast ANN search
4. **Retrieval**: User question → embed → top-k nearest chunks
5. **Generation**: Retrieved chunks injected into LLM prompt as context
6. **Grounding**: LLM answers ONLY from provided context, reducing hallucination
