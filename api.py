import os
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import build_vectorstore, RAGPipeline

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload documents and ask questions grounded in their content.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (use Redis for production)
_sessions: dict[str, RAGPipeline] = {}


class QueryRequest(BaseModel):
    session_id: str
    question: str
    hf_token: str = ""


class QueryResponse(BaseModel):
    answer: str
    source_chunks: list[str]
    page_refs: list[int]


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    hf_token: str = Form(""),
    model_id: str = Form("mistralai/Mistral-7B-Instruct-v0.2"),
):
    token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if not token:
        raise HTTPException(status_code=400, detail="HuggingFace API token is required.")

    allowed = {".pdf", ".txt", ".md"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        vectorstore, num_pages, num_chunks = build_vectorstore(tmp_path)
        session_id = str(uuid.uuid4())
        _sessions[session_id] = RAGPipeline(vectorstore, token, model_id)
        return {
            "session_id": session_id,
            "filename": file.filename,
            "pages": num_pages,
            "chunks": num_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
def query_document(req: QueryRequest):
    pipeline = _sessions.get(req.session_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Session not found. Upload a document first.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = pipeline.query(req.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": session_id}
    raise HTTPException(status_code=404, detail="Session not found.")
