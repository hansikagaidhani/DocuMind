import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use ONLY the retrieved context below to answer the question. "
    'If the answer is not in the context, say "I don\'t know." '
    "Keep your answer concise and factual."
)


def load_document(file_path: str) -> list:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


def build_vectorstore(file_path: str) -> tuple:
    docs = load_document(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(docs), len(chunks)


class RAGPipeline:
    def __init__(self, vectorstore: FAISS, hf_token: str, model_id: str = DEFAULT_LLM_MODEL):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        self.client = InferenceClient(model=model_id, token=hf_token)

    def query(self, question: str) -> dict:
        source_docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in source_docs)

        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"},
            {"role": "user", "content": question},
        ]

        response = self.client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()

        page_refs = sorted(
            {doc.metadata.get("page", i) for i, doc in enumerate(source_docs)}
        )
        return {
            "answer": answer,
            "source_chunks": [doc.page_content[:200] for doc in source_docs],
            "page_refs": page_refs,
        }
