"""RAG Tabanli Soru-Cevap Sistemi — Streamlit Arayuzu.

Kullanim:
    streamlit run app.py
"""

import sys
import time
import shutil
from pathlib import Path
from typing import List, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Sayfa Konfigurasyonu ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Soru-Cevap Sistemi",
    page_icon="assets/favicon.ico" if (PROJECT_ROOT / "assets/favicon.ico").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0a0e1a;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(139,92,246,0.06) 100%);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 16px;
    padding: 2.2rem 3rem;
    margin-bottom: 1.8rem;
    text-align: center;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f1f5f9;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 0.875rem;
    color: #64748b;
    margin: 0;
    letter-spacing: 0.01em;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080c18 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

.sidebar-brand {
    padding: 1.5rem 0 1rem;
    text-align: center;
}
.sidebar-brand .brand-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: 0.01em;
}
.sidebar-brand .brand-sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 0.2rem;
}

/* ── Status indicators ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    padding: 0.25rem 0;
}
.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-green { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.5); }
.dot-red   { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.5); }
.dot-yellow { background: #f59e0b; box-shadow: 0 0 6px rgba(245,158,11,0.5); }

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #3b82f6;
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Chat bubbles ── */
.chat-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.75rem 0;
    align-items: flex-end;
    gap: 0.6rem;
}
.chat-user .bubble {
    background: #3b82f6;
    color: #fff;
    border-radius: 14px 14px 3px 14px;
    padding: 0.75rem 1.1rem;
    max-width: 68%;
    font-size: 0.9rem;
    line-height: 1.55;
}
.chat-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 0.75rem 0;
    align-items: flex-end;
    gap: 0.6rem;
}
.chat-assistant .bubble {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #cbd5e1;
    border-radius: 14px 14px 14px 3px;
    padding: 0.75rem 1.1rem;
    max-width: 72%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    flex-shrink: 0;
    letter-spacing: 0.02em;
}
.avatar-user { background: #3b82f6; color: #fff; }
.avatar-ai   { background: rgba(139,92,246,0.25); color: #a78bfa; border: 1px solid rgba(139,92,246,0.3); }

/* ── Cache badge ── */
.cache-badge {
    display: inline-block;
    background: rgba(245,158,11,0.15);
    color: #f59e0b;
    border: 1px solid rgba(245,158,11,0.3);
    font-size: 0.68rem;
    font-weight: 500;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    margin-left: 0.5rem;
    vertical-align: middle;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Elapsed badge ── */
.elapsed-info {
    font-size: 0.7rem;
    color: #334155;
    margin-top: 0.35rem;
}

/* ── Source card ── */
.source-card {
    background: rgba(59,130,246,0.04);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-top: 0.4rem;
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.5;
}
.source-card .source-name {
    color: #93c5fd;
    font-weight: 500;
    margin-bottom: 0.2rem;
}

/* ── File list item ── */
.file-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.35rem;
    font-size: 0.82rem;
    color: #94a3b8;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.file-item .file-name { color: #cbd5e1; font-weight: 500; }
.file-item .file-size { color: #334155; font-size: 0.75rem; }

/* ── Buttons ── */
.stButton > button {
    background: #3b82f6 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.45rem 1rem !important;
    transition: background 0.15s, transform 0.1s !important;
}
.stButton > button:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
}

/* ── Form inputs ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(59,130,246,0.5) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #64748b !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f1f5f9 !important;
}

/* ── Selectbox / radio ── */
.stRadio > div {
    gap: 0.5rem;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

/* ── File uploader ── */
.stFileUploader > div {
    background: rgba(59,130,246,0.03) !important;
    border: 1.5px dashed rgba(59,130,246,0.25) !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    color: #64748b !important;
}

/* ── Dividers ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── General text ── */
p, label, li { color: #94a3b8 !important; }
h1, h2, h3, h4 { color: #f1f5f9 !important; }
.stMarkdown h3 { font-size: 1rem !important; font-weight: 600 !important; color: #e2e8f0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

/* ── Code blocks ── */
.stCode, code { font-size: 0.78rem !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 0;
    color: #334155;
}
.empty-state .empty-icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
    opacity: 0.4;
}
.empty-state .empty-text {
    font-size: 0.875rem;
    color: #475569;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ─── Guvenli import ───────────────────────────────────────────────────────────
def _safe_import():
    try:
        from src.config import config, logger
        from src.embeddings import create_vector_store, save_vector_store, load_vector_store
        from src.document_loader import load_documents
        from src.text_splitter import split_documents
        from src.retriever import semantic_search, hybrid_search
        from src.generator import create_qa_chain, ask_question
        from src.cache import QueryCache
        return True, None
    except Exception as e:
        return False, str(e)


ok, err = _safe_import()
if not ok:
    st.error(f"Moduller yuklenemedi: {err}")
    st.stop()

from src.config import config, logger
from src.embeddings import create_vector_store, save_vector_store, load_vector_store
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.retriever import semantic_search, hybrid_search
from src.generator import create_qa_chain, ask_question
from src.cache import QueryCache


# ─── Session State ────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "question_count": 0,
        "total_tokens": 0,
        "index_loaded": False,
        "vector_store": None,
        "qa_chain": None,
        "search_type": "semantic",
        "last_search_results": None,
        "last_search_query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── Kaynak yukleme ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_vector_store_cached():
    return load_vector_store()


@st.cache_resource(show_spinner=False)
def _load_qa_chain_cached(_vs):
    return create_qa_chain(_vs)


def _ensure_index():
    if st.session_state.vector_store is None:
        vs = _load_vector_store_cached()
        if vs:
            st.session_state.vector_store = vs
            st.session_state.qa_chain = _load_qa_chain_cached(vs)
            st.session_state.index_loaded = True
    return st.session_state.vector_store is not None


def _invalidate_cache():
    st.cache_resource.clear()
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.index_loaded = False


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-name">RAG Assistant</div>
        <div class="brand-sub">LangChain · Gemini · FAISS</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Durum gostergeleri
    index_exists = False
    if config.index_dir.exists():
        index_exists = any(config.index_dir.iterdir())

    if index_exists:
        st.markdown('<div class="status-row"><div class="dot dot-green"></div><span style="color:#94a3b8;font-size:0.82rem;">Index Ready</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-row"><div class="dot dot-red"></div><span style="color:#94a3b8;font-size:0.82rem;">Index Not Found</span></div>', unsafe_allow_html=True)

    try:
        config.validate()
        st.markdown('<div class="status-row"><div class="dot dot-green"></div><span style="color:#94a3b8;font-size:0.82rem;">API Connected</span></div>', unsafe_allow_html=True)
    except ValueError:
        st.markdown('<div class="status-row"><div class="dot dot-red"></div><span style="color:#94a3b8;font-size:0.82rem;">API Key Missing</span></div>', unsafe_allow_html=True)

    st.divider()

    # Arama tipi
    st.markdown('<span style="font-size:0.78rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.06em;">Search Mode</span>', unsafe_allow_html=True)
    search_type = st.radio(
        label="search_mode",
        options=["semantic", "hybrid"],
        format_func=lambda x: "Semantic" if x == "semantic" else "Hybrid  (BM25 + Semantic)",
        index=0 if st.session_state.search_type == "semantic" else 1,
        label_visibility="collapsed",
    )
    st.session_state.search_type = search_type

    st.divider()

    # Oturum metrikleri
    st.markdown('<span style="font-size:0.78rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.06em;">Session</span>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.question_count}</div><div class="metric-label">Queries</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_tokens:,}</div><div class="metric-label">Tokens</div></div>', unsafe_allow_html=True)

    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.session_state.total_tokens = 0
        st.rerun()


# ─── Ana baslik ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">RAG Question Answering System</div>
    <div class="hero-sub">LangChain · Google Gemini · FAISS — Retrieval-Augmented Generation</div>
</div>
""", unsafe_allow_html=True)


# ─── Sekmeler ─────────────────────────────────────────────────────────────────
tab_chat, tab_docs, tab_search, tab_settings = st.tabs([
    "Chat", "Documents", "Search Explorer", "Settings"
])


# ══════════════════════════════════════════════════════════════════════════════
# SEKME 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not _ensure_index():
        st.warning("Index not found. Please upload documents in the Documents tab and index them.", icon=None)

    # Mesaj gecmisi
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-text">No messages yet.<br>Type your question below to get started.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    <div class="bubble">{msg["content"]}</div>
                    <div class="chat-avatar avatar-user">YOU</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                cache_badge = '<span class="cache-badge">Cached</span>' if msg.get("from_cache") else ""
                elapsed_str = f'<div class="elapsed-info">{msg.get("elapsed","")}</div>' if msg.get("elapsed") else ""
                st.markdown(f"""
                <div class="chat-assistant">
                    <div class="chat-avatar avatar-ai">AI</div>
                    <div>
                        <div class="bubble">
                            {msg["content"]}{cache_badge}
                            {elapsed_str}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if msg.get("sources"):
                    with st.expander(f"{len(msg['sources'])} source document(s)"):
                        for i, src in enumerate(msg["sources"], 1):
                            name = src.get("source", "Unknown")
                            preview = src.get("preview", "")
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-name">[{i}]  {name}</div>
                                {preview}
                            </div>
                            """, unsafe_allow_html=True)

    st.divider()

    # Soru girisi
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                label="question",
                placeholder="Ask a question about your documents...",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        question = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": question})

        if not st.session_state.index_loaded:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Please index your documents first.",
                "sources": [], "from_cache": False, "elapsed": "",
            })
            st.rerun()

        cache = QueryCache()
        cached = cache.get(question)

        if cached:
            sources = []
            for s in cached.get("source_documents", []):
                src_name = s.metadata.get("source_file", s.metadata.get("source", "Unknown"))
                preview = s.page_content[:120].replace("\n", " ").strip() + "..."
                sources.append({"source": src_name, "preview": preview})
            st.session_state.messages.append({
                "role": "assistant",
                "content": cached["answer"],
                "sources": sources,
                "from_cache": True,
                "elapsed": "",
            })
            st.session_state.question_count += 1
            st.rerun()

        with st.spinner("Searching documents and generating answer..."):
            try:
                t0 = time.time()
                result = ask_question(st.session_state.qa_chain, question, show_sources=False, show_tokens=False)
                elapsed = time.time() - t0

                sources = []
                seen = set()
                for doc in result.get("source_documents", []):
                    src = doc.metadata.get("source_file", doc.metadata.get("source", "Unknown"))
                    if src in seen:
                        continue
                    seen.add(src)
                    preview = doc.page_content[:120].replace("\n", " ").strip() + "..."
                    sources.append({"source": src, "preview": preview})

                cache.set(question, result)
                st.session_state.question_count += 1
                st.session_state.total_tokens += result.get("total_tokens", 0)

                tokens = result.get("total_tokens", 0)
                elapsed_label = f"{elapsed:.1f}s · {tokens:,} tokens" if tokens else f"{elapsed:.1f}s"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "from_cache": False,
                    "elapsed": elapsed_label,
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"An error occurred: {str(e)}",
                    "sources": [], "from_cache": False, "elapsed": "",
                })

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SEKME 2 — DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### Upload Documents")
        st.markdown('<p style="font-size:0.82rem;color:#475569;">Supported formats: PDF, TXT, DOCX, MD. Multiple files can be uploaded at once.</p>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            label="Select files",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.markdown(f'<p style="font-size:0.82rem;color:#64748b;margin-bottom:0.5rem;">{len(uploaded_files)} file(s) selected</p>', unsafe_allow_html=True)
            for f in uploaded_files:
                size_kb = len(f.getvalue()) / 1024
                st.markdown(f'<div class="file-item"><span class="file-name">{f.name}</span><span class="file-size">{size_kb:.1f} KB</span></div>', unsafe_allow_html=True)

            if st.button("Save and Index", use_container_width=True):
                config.documents_dir.mkdir(parents=True, exist_ok=True)
                prog = st.progress(0, text="Saving files...")
                for i, f in enumerate(uploaded_files):
                    (config.documents_dir / f.name).write_bytes(f.getvalue())
                    prog.progress((i + 1) / len(uploaded_files), text=f"Saved: {f.name}")

                st.success(f"{len(uploaded_files)} file(s) saved.")

                with st.spinner("Indexing documents..."):
                    try:
                        docs = load_documents(str(config.documents_dir))
                        chunks = split_documents(docs)
                        vs = create_vector_store(chunks)
                        save_vector_store(vs)
                        _invalidate_cache()
                        st.session_state.vector_store = vs
                        st.session_state.qa_chain = create_qa_chain(vs)
                        st.session_state.index_loaded = True
                        st.success(f"Indexing complete. {len(docs)} document(s) → {len(chunks)} chunk(s).")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Indexing error: {str(e)}")

    with col_right:
        st.markdown("### Document Library")
        docs_dir = config.documents_dir
        if docs_dir.exists():
            all_files = sorted(
                [f for f in docs_dir.iterdir() if f.is_file()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if all_files:
                for f in all_files[:20]:
                    size_kb = f.stat().st_size / 1024
                    st.markdown(f'<div class="file-item"><span class="file-name">{f.name}</span><span class="file-size">{size_kb:.1f} KB</span></div>', unsafe_allow_html=True)
                if len(all_files) > 20:
                    st.caption(f"... and {len(all_files) - 20} more")
            else:
                st.markdown('<div class="empty-state"><div class="empty-text">No documents uploaded yet.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state"><div class="empty-text">Document directory not found.</div></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### Index Management")

        idx_files = list(config.index_dir.iterdir()) if config.index_dir.exists() else []
        if idx_files:
            idx_size = sum(f.stat().st_size for f in config.index_dir.rglob("*") if f.is_file())
            st.markdown(f'<div class="metric-card" style="margin-bottom:1rem;text-align:left;padding:0.75rem 1rem;"><span style="font-size:0.78rem;color:#22c55e;font-weight:600;">ACTIVE</span><span style="font-size:0.78rem;color:#334155;margin-left:1rem;">{idx_size/1024:.1f} KB</span></div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Rebuild Index", use_container_width=True):
                    with st.spinner("Rebuilding..."):
                        try:
                            if config.index_dir.exists():
                                shutil.rmtree(config.index_dir)
                            config.index_dir.mkdir(parents=True, exist_ok=True)
                            docs = load_documents(str(config.documents_dir))
                            chunks = split_documents(docs)
                            vs = create_vector_store(chunks)
                            save_vector_store(vs)
                            _invalidate_cache()
                            st.session_state.vector_store = vs
                            st.session_state.qa_chain = create_qa_chain(vs)
                            st.session_state.index_loaded = True
                            st.success("Index rebuilt.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            with c2:
                if st.button("Delete Index", use_container_width=True):
                    if config.index_dir.exists():
                        shutil.rmtree(config.index_dir)
                        config.index_dir.mkdir(parents=True, exist_ok=True)
                        _invalidate_cache()
                        st.warning("Index deleted.")
                        st.rerun()
        else:
            st.markdown('<p style="font-size:0.82rem;color:#475569;">No index found.</p>', unsafe_allow_html=True)
            if st.button("Index Existing Documents", use_container_width=True):
                with st.spinner("Indexing..."):
                    try:
                        docs = load_documents(str(config.documents_dir))
                        chunks = split_documents(docs)
                        vs = create_vector_store(chunks)
                        save_vector_store(vs)
                        _invalidate_cache()
                        st.session_state.vector_store = vs
                        st.session_state.qa_chain = create_qa_chain(vs)
                        st.session_state.index_loaded = True
                        st.success(f"{len(docs)} document(s) indexed.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# SEKME 3 — SEARCH EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("### Search Explorer")
    st.markdown('<p style="font-size:0.82rem;color:#475569;">Run a raw semantic search against the vector index to inspect retrieved chunks and similarity scores.</p>', unsafe_allow_html=True)

    search_q = st.text_input(
        "Query",
        placeholder="Enter a search query...",
        value=st.session_state.last_search_query,
    )

    col_s1, col_s2 = st.columns([1, 5])
    with col_s1:
        top_k_search = st.number_input("Top-K", min_value=1, max_value=20, value=config.top_k, step=1)

    if st.button("Search", use_container_width=False) and search_q.strip():
        if not _ensure_index():
            st.error("Index not found. Please index your documents first.")
        else:
            with st.spinner("Searching..."):
                try:
                    sem_results = semantic_search(
                        st.session_state.vector_store,
                        search_q,
                        top_k=top_k_search,
                    )
                    st.session_state.last_search_results = {
                        "semantic": sem_results,
                        "query": search_q,
                    }
                    st.session_state.last_search_query = search_q
                except Exception as e:
                    st.error(f"Search error: {str(e)}")

    if st.session_state.last_search_results:
        data = st.session_state.last_search_results
        sem = data.get("semantic", [])

        st.markdown(f'<p style="font-size:0.82rem;color:#475569;margin:1rem 0 0.5rem;">Query: <code style="color:#93c5fd;">{data["query"]}</code> — {len(sem)} result(s)</p>', unsafe_allow_html=True)
        st.divider()

        if sem:
            for i, (doc, score) in enumerate(sem, 1):
                src = doc.metadata.get("source_file", doc.metadata.get("source", "Unknown"))
                preview = doc.page_content[:300].replace("\n", " ").strip()
                normalized_sim = max(0.0, 1.0 - min(score, 2.0) / 2.0)

                with st.expander(f"Result {i}  ·  {src}  ·  {normalized_sim:.0%} similarity"):
                    st.progress(normalized_sim)
                    st.markdown(f'<div class="source-card">{preview}...</div>', unsafe_allow_html=True)
                    st.caption(f"FAISS distance: {score:.4f}  ·  Length: {len(doc.page_content)} chars")
        else:
            st.markdown('<div class="empty-state"><div class="empty-text">No results found.</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SEKME 4 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    col_cfg, col_cache = st.columns([1, 1], gap="large")

    with col_cfg:
        st.markdown("### Configuration")

        cfg_data = {
            "LLM Model": config.llm_model,
            "Embedding Model": config.embedding_model,
            "Temperature": str(config.temperature),
            "Chunk Size": str(config.chunk_size),
            "Chunk Overlap": str(config.chunk_overlap),
            "Top-K": str(config.top_k),
            "Vector DB": config.vector_db.upper(),
            "Log Level": config.log_level,
        }

        for k, v in cfg_data.items():
            col_k, col_v = st.columns([2, 3])
            with col_k:
                st.markdown(f'<span style="font-size:0.82rem;color:#475569;">{k}</span>', unsafe_allow_html=True)
            with col_v:
                st.markdown(f'<span style="font-size:0.82rem;color:#cbd5e1;font-weight:500;">{v}</span>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### Directories")
        dirs = {
            "Documents": config.documents_dir,
            "Index": config.index_dir,
            "Cache": config.cache_dir,
            "Logs": config.logs_dir,
        }
        for name, path in dirs.items():
            exists_label = "OK" if path.exists() else "MISSING"
            exists_color = "#22c55e" if path.exists() else "#ef4444"
            st.markdown(
                f'<div class="file-item"><span class="file-name">{name}</span>'
                f'<span style="font-size:0.72rem;color:{exists_color};font-weight:600;">{exists_label}</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("### API Status")
        try:
            config.validate()
            key_preview = config.gemini_api_key[:6] + "..." + config.gemini_api_key[-4:] if len(config.gemini_api_key) > 10 else "***"
            st.markdown(f'<div class="status-row"><div class="dot dot-green"></div><span style="color:#94a3b8;font-size:0.82rem;">Active &nbsp;·&nbsp; <code style="color:#64748b;">{key_preview}</code></span></div>', unsafe_allow_html=True)
        except ValueError as e:
            st.markdown('<div class="status-row"><div class="dot dot-red"></div><span style="color:#ef4444;font-size:0.82rem;">API key missing or invalid</span></div>', unsafe_allow_html=True)

    with col_cache:
        st.markdown("### Cache")

        cache_files = list(config.cache_dir.glob("*.json")) if config.cache_dir.exists() else []
        total_cache_size = sum(f.stat().st_size for f in cache_files) / 1024

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(cache_files)}</div><div class="metric-label">Entries</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_cache_size:.1f}</div><div class="metric-label">KB Used</div></div>', unsafe_allow_html=True)

        st.markdown(f'<p style="font-size:0.75rem;color:#334155;margin-top:0.5rem;">TTL: {config.cache_ttl_hours}h</p>', unsafe_allow_html=True)
        st.divider()

        if st.button("Clear Cache", use_container_width=True):
            try:
                cache = QueryCache()
                cache.clear()
                st.success("Cache cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

        st.divider()
        st.markdown("### Application Log")
        log_file = config.logs_dir / "app.log"
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            last_lines = lines[-25:]
            st.code("".join(last_lines), language="text")
            st.caption(f"Last {len(last_lines)} of {len(lines)} lines")
        else:
            st.markdown('<p style="font-size:0.82rem;color:#334155;">No log file yet.</p>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### Quick Start")
        st.markdown("""
        <ol style="font-size:0.82rem;color:#475569;line-height:2;padding-left:1.2rem;">
            <li>Upload documents in the <strong style="color:#94a3b8;">Documents</strong> tab</li>
            <li>Click <strong style="color:#94a3b8;">Save and Index</strong></li>
            <li>Go to the <strong style="color:#94a3b8;">Chat</strong> tab and ask a question</li>
            <li>Use <strong style="color:#94a3b8;">Search Explorer</strong> to inspect raw results</li>
        </ol>
        """, unsafe_allow_html=True)
