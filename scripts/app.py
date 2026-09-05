import os
import sys
import tempfile
import time
from typing import List, Dict, Any

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Ensure scripts directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from retriever import build_retriever, clean_text, METRICS_FILE
from generation import create_model, get_available_models
from rag_chain import ConversationalRAGEngine, format_docs_with_citations


# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="DocuMind RAG - PDF Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .source-card {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    .badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1D4ED8;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of dicts: {"role", "content", "sources", "standalone_query"}

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "document_metrics" not in st.session_state:
    st.session_state.document_metrics = None

if "active_file_id" not in st.session_state:
    st.session_state.active_file_id = None

if "temp_doc_path" not in st.session_state:
    st.session_state.temp_doc_path = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_messages_for_langchain(history: List[Dict[str, Any]]) -> List[Any]:
    """Converts internal UI message history into LangChain BaseMessage objects."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def reset_chat():
    """Clears conversational history."""
    st.session_state.chat_history = []
    st.rerun()


def reset_all():
    """Clears entire application state including loaded document."""
    st.session_state.chat_history = []
    st.session_state.rag_engine = None
    st.session_state.retriever = None
    st.session_state.document_metrics = None
    st.session_state.active_file_id = None
    if st.session_state.temp_doc_path and os.path.exists(st.session_state.temp_doc_path):
        try:
            os.remove(st.session_state.temp_doc_path)
        except Exception:
            pass
    st.session_state.temp_doc_path = None
    st.rerun()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.title("⚙️ RAG Settings")
    st.markdown("---")

    # 1. Document Upload
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="Upload any PDF to build dense (FAISS) and sparse (BM25) search indexes.",
    )

    st.markdown("---")

    # 2. LLM Model Options (Hugging Face)
    st.subheader("🤖 Hugging Face Model")
    available_models = get_available_models()
    model_options = available_models + ["Enter Custom Model ID..."]

    model_selection = st.selectbox(
        "Select Model",
        options=model_options,
        index=0,
        help="Choose a pre-configured Hugging Face model or type a custom repository ID.",
    )

    if model_selection == "Enter Custom Model ID...":
        selected_model_id = st.text_input(
            "Hugging Face Repo ID",
            value="openai/gpt-oss-120b",
            placeholder="e.g. openai/gpt-oss-120b",
        ).strip()
    else:
        selected_model_id = model_selection

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    with col2:
        max_tokens = st.slider("Max Tokens", min_value=256, max_value=2048, value=1024, step=128)

    st.markdown("---")

    # 3. Retrieval & HyDE Settings
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider(
        "Top-K Chunks to Retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of most relevant context chunks to provide to the model.",
    )

    use_hyde = st.toggle(
        "Enable HyDE (Hypothetical Embeddings)",
        value=True,
        help="Expands the user question into a hypothetical answer before retrieval to improve semantic matches.",
    )

    use_compression = st.toggle(
        "Enable Contextual Compression",
        value=True,
        help="Extracts verbatim relevant spans from retrieved chunks using an extractor LLM, pruning irrelevant context.",
    )

    force_rebuild = st.checkbox(
        "Force Re-index",
        value=False,
        help="Ignore cached FAISS index and regenerate embeddings.",
    )

    st.markdown("---")

    # 4. Session Controls
    st.subheader("🔄 Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            reset_chat()
    with c2:
        if st.button("🔄 Reset All", use_container_width=True):
            reset_all()


# ============================================================
# DOCUMENT PROCESSING PIPELINE
# ============================================================

if uploaded_file:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    # Check if a new file has been uploaded or settings changed
    needs_processing = (
        st.session_state.active_file_id != file_id
        or st.session_state.rag_engine is None
        or force_rebuild
    )

    if needs_processing:
        with st.status("🔄 Indexing document with Hybrid RAG...", expanded=True) as status:
            temp_dir = os.path.join(ROOT_DIR, "data", "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)

            status.write("📥 Saving uploaded file temporarily...")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.temp_doc_path = temp_file_path

            status.write("⚡ Loading, chunking, and creating FAISS + BM25 indexes...")
            try:
                retriever, metrics = build_retriever(
                    document_path=temp_file_path,
                    force_rebuild=force_rebuild,
                    top_k=top_k,
                    use_hyde=use_hyde,
                )

                status.write("🧠 Initializing language model...")
                model = create_model(
                    model_name=selected_model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=True,
                )

                # Initialize RAG Engine
                st.session_state.retriever = retriever
                st.session_state.rag_engine = ConversationalRAGEngine(
                    retriever=retriever,
                    model=model,
                    use_compression=use_compression,
                )
                st.session_state.document_metrics = metrics
                st.session_state.active_file_id = file_id
                st.session_state.chat_history = []

                status.update(
                    label=f"✅ Document '{uploaded_file.name}' indexed successfully!",
                    state="complete",
                    expanded=False,
                )
                st.toast("Document indexed and ready for chat!", icon="🎉")

            except Exception as e:
                status.update(label=f"❌ Error indexing document: {e}", state="error")
                st.error(f"Failed to process document: {e}")

    # Synchronize retriever and engine parameters if changed dynamically in sidebar
    elif st.session_state.retriever is not None and st.session_state.rag_engine is not None:
        st.session_state.retriever.top_k = top_k
        st.session_state.retriever.use_hyde = use_hyde
        st.session_state.rag_engine.use_compression = use_compression
        # Update model in rag_engine if parameters or model choice changed
        try:
            st.session_state.rag_engine.model = create_model(
                model_name=selected_model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True,
            )
            st.session_state.rag_engine.extractor_model = st.session_state.rag_engine.model
        except Exception as e:
            st.warning(f"Could not re-initialize model '{selected_model_id}': {e}")


# ============================================================
# SIDEBAR METRICS DISPLAY
# ============================================================

if st.session_state.document_metrics:
    m = st.session_state.document_metrics
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Document Statistics")
        st.markdown(
            f"""
            <div class="metric-box">
                <b>Document:</b> {m.get('document_name', 'N/A')}<br>
                <b>Size:</b> {m.get('document_size_mb', 0)} MB<br>
                <b>Pages:</b> {m.get('pages', 0)} | <b>Chunks:</b> {m.get('chunks', 0)}<br>
                <b>Loading Speed:</b> {m.get('loading_mb_per_sec', 0)} MB/s<br>
                <b>Total Index Time:</b> {m.get('total_time_sec', 0)}s
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MAIN USER INTERFACE
# ============================================================

st.markdown('<div class="main-title">📚 DocuMind RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Advanced Conversational Document Q&A powered by Hybrid HyDE Retrieval & LangChain</div>',
    unsafe_allow_html=True,
)

# If no document is uploaded yet, display friendly onboarding instructions
if not uploaded_file or not st.session_state.rag_engine:
    st.info("👈 **Get Started**: Please upload a PDF document in the sidebar to begin your chat session.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Hybrid Retrieval")
        st.markdown("Combines **dense semantic vectors (FAISS)** with **sparse keyword matching (BM25)** for precise document recall.")
    with col2:
        st.markdown("### 💡 HyDE Query Expansion")
        st.markdown("Generates hypothetical passages from your query to bridge vocabulary mismatch and boost semantic search accuracy.")
    with col3:
        st.markdown("### 📌 Grounded Citations")
        st.markdown("Answers are strictly grounded in document context with interactive expandable source & page references.")

else:
    # ------------------------------------------------------------
    # CHAT MESSAGE HISTORY
    # ------------------------------------------------------------
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        standalone_q = msg.get("standalone_query", "")

        with st.chat_message(role):
            st.markdown(content)

            # Display source expander for assistant messages if sources exist
            if role == "assistant" and sources:
                with st.expander(f"🔍 View {len(sources)} Retrieved Sources & Context Chunks"):
                    if standalone_q:
                        st.caption(f"**Contextualized Search Query:** `{standalone_q}`")

                        for idx, doc in enumerate(sources, start=1):
                            page_num = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
                            source_name = doc.metadata.get("source_file", "PDF")
                            snippet = clean_text(doc.page_content)
                            comp_badge = '<span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); margin-left: 4px;">⚡ Verbatim Extracted</span>' if doc.metadata.get("compressed") else ''

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <span class="badge">Source {idx}</span>{comp_badge}
                                    <b>{source_name} (Page {page_num})</b>
                                    <p style="margin-top: 6px; white-space: pre-wrap;">{snippet}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # ------------------------------------------------------------
    # CHAT INPUT & STREAMING GENERATION
    # ------------------------------------------------------------
    user_query = st.chat_input("Ask any question about your uploaded document...")

    if user_query:
        # 1. Display User Message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Convert conversation history for LangChain
        langchain_history = convert_messages_for_langchain(st.session_state.chat_history[:-1])

        # 3. Stream Assistant Response
        with st.chat_message("assistant"):
            try:
                stream_generator, retrieved_docs, standalone_query = (
                    st.session_state.rag_engine.stream_response(
                        user_input=user_query,
                        chat_history=langchain_history,
                    )
                )

                # Stream response into UI
                full_response = st.write_stream(stream_generator)

                # Show Retrieved Sources
                if retrieved_docs:
                    with st.expander(f"🔍 View {len(retrieved_docs)} Retrieved Sources & Context Chunks"):
                        if standalone_query:
                            st.caption(f"**Contextualized Search Query:** `{standalone_query}`")

                        for idx, doc in enumerate(retrieved_docs, start=1):
                            page_num = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
                            source_name = doc.metadata.get("source_file", "PDF")
                            snippet = clean_text(doc.page_content)
                            comp_badge = '<span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); margin-left: 4px;">⚡ Verbatim Extracted</span>' if doc.metadata.get("compressed") else ''

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <span class="badge">Source {idx}</span>{comp_badge}
                                    <b>{source_name} (Page {page_num})</b>
                                    <p style="margin-top: 6px; white-space: pre-wrap;">{snippet}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # 4. Save Assistant Message to State
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": retrieved_docs,
                    "standalone_query": standalone_query,
                })

            except Exception as e:
                err_str = str(e)
                error_msg = f"⚠️ **Error generating answer:** {err_str}"
                st.error(error_msg)
                if "model_not_supported" in err_str or "not supported by any provider" in err_str:
                    st.warning("💡 **Suggestion:** The selected Hugging Face model requires enabled third-party providers. In the left sidebar, switch **LLM Provider** to **⚡ Groq (Recommended)** and use **LLaMA 3.3 70B** for reliable responses.")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "standalone_query": "",
                })
