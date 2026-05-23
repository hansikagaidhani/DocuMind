import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DocuMind – Chat with Your Documents",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 DocuMind")
    st.caption("Powered by LangChain · FAISS · HuggingFace")
    st.divider()

    hf_token = st.text_input(
        "🔑 HuggingFace API Token",
        type="password",
        value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
        help="Get a free token at huggingface.co/settings/tokens",
    )

    model_id = st.selectbox(
        "🤖 LLM Model",
        options=[
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
        ],
        help="Inference runs on HuggingFace Serverless API (free tier).",
    )

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF or text file")
    st.markdown("2. Document is chunked & embedded")
    st.markdown("3. Your question retrieves top-4 chunks")
    st.markdown("4. LLM answers using only that context")
    st.divider()

    if st.button("🗑️ Clear Session", use_container_width=True):
        if st.session_state.get("session_id"):
            try:
                requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=5)
            except Exception:
                pass
        for key in ["session_id", "doc_info", "messages"]:
            st.session_state.pop(key, None)
        st.rerun()

# ── Session state init ─────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main header ────────────────────────────────────────────────────────────────
st.title("🧠 DocuMind")
st.caption("Upload any document. Ask anything. Get answers grounded in your content.")

# ── Upload section ─────────────────────────────────────────────────────────────
with st.container(border=True):
    col_upload, col_btn = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
        )
    with col_btn:
        process_clicked = st.button(
            "⚙️ Process",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    if process_clicked and uploaded_file:
        if not hf_token:
            st.warning("Enter your HuggingFace API Token in the sidebar first.")
        else:
            with st.spinner("Chunking · Embedding · Indexing…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")},
                        data={"hf_token": hf_token, "model_id": model_id},
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.doc_info = data
                        st.session_state.messages = []
                        st.success(
                            f"✅ **{data['filename']}** ready — "
                            f"{data['pages']} pages · {data['chunks']} chunks indexed"
                        )
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend. Is the API server running?")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Active doc banner ──────────────────────────────────────────────────────────
if st.session_state.doc_info:
    d = st.session_state.doc_info
    st.info(
        f"📂 **Active:** {d['filename']} &nbsp;|&nbsp; "
        f"{d['pages']} pages · {d['chunks']} chunks · model: `{model_id}`",
        icon="ℹ️",
    )

st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("page_refs"):
            st.caption(f"📖 Referenced pages: {msg['page_refs']}")
        if msg.get("source_chunks"):
            with st.expander("🔍 Retrieved context chunks"):
                for i, chunk in enumerate(msg["source_chunks"], 1):
                    st.markdown(f"**Chunk {i}:** {chunk}…")

# ── Chat input ─────────────────────────────────────────────────────────────────
placeholder = (
    "Ask a question about your document…"
    if st.session_state.session_id
    else "Upload and process a document to start asking questions"
)

if prompt := st.chat_input(placeholder, disabled=not st.session_state.session_id):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context · Generating answer…"):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={
                        "session_id": st.session_state.session_id,
                        "question": prompt,
                    },
                    timeout=90,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    page_refs = data.get("page_refs", [])
                    source_chunks = data.get("source_chunks", [])

                    st.markdown(answer)
                    if page_refs:
                        st.caption(f"📖 Referenced pages: {page_refs}")
                    if source_chunks:
                        with st.expander("🔍 Retrieved context chunks"):
                            for i, chunk in enumerate(source_chunks, 1):
                                st.markdown(f"**Chunk {i}:** {chunk}…")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "page_refs": page_refs,
                        "source_chunks": source_chunks,
                    })
                else:
                    err = resp.json().get("detail", resp.text)
                    st.error(f"Query failed: {err}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend API.")
            except Exception as e:
                st.error(f"Error: {e}")
