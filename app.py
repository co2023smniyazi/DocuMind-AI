"""
Streamlit User Interface (Cloud-Ready & Claude AI Integrated)
Local & Cloud LLM-Based College Document Question Answering Assistant Using RAG
Project Management and Tracking Using Jira
"""

import os
import time
import streamlit as st
import pandas as pd
from rag_service import RAGService
from document_loader import load_pdf_with_metadata
from llm_service import ClaudeLLMService, OllamaLLMService, MockLLMService

# Configure page layout and style
st.set_page_config(
    page_title="DocuMind AI - College Document Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .answer-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .source-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 8px 12px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.95rem;
    }
    .status-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for runtime provider selection
if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = os.environ.get("LLM_PROVIDER", "claude")
if "user_anthropic_key" not in st.session_state:
    st.session_state["user_anthropic_key"] = ""
if "claude_model" not in st.session_state:
    st.session_state["claude_model"] = os.environ.get("CLAUDE_MODEL", "claude-3-5-haiku-20241022")


def get_configured_rag():
    # If user provided a key in the UI session, pass it to the service
    provider = st.session_state["llm_provider"]
    api_key = st.session_state["user_anthropic_key"] or os.environ.get("ANTHROPIC_API_KEY")

    rag = RAGService(llm_provider=provider)
    if provider == "claude" and isinstance(rag.llm_service, ClaudeLLMService):
        if api_key:
            rag.llm_service.api_key = api_key
        rag.llm_service.model_name = st.session_state["claude_model"]

    # Verify vector store population
    stats = rag.get_stats()
    if stats["total_chunks"] == 0 and os.path.exists("./data"):
        rag.index_directory("./data")

    return rag


rag_service = get_configured_rag()
health = rag_service.check_health()
stats = rag_service.get_stats()


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/books.png", width=64)
    st.title("Cloud Settings")

    st.markdown("### 🤖 LLM Provider Selection")
    provider_options = ["Anthropic Claude (Cloud)", "Ollama (Local Fallback)", "Mock (Testing / Offline)"]
    current_idx = 0 if st.session_state["llm_provider"] == "claude" else (1 if st.session_state["llm_provider"] == "ollama" else 2)
    selected_provider_label = st.selectbox("Active Provider:", provider_options, index=current_idx)

    new_provider = "claude" if "Claude" in selected_provider_label else ("ollama" if "Ollama" in selected_provider_label else "mock")
    if new_provider != st.session_state["llm_provider"]:
        st.session_state["llm_provider"] = new_provider
        st.rerun()

    if st.session_state["llm_provider"] == "claude":
        st.markdown("**Claude Model:**")
        model_options = [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307"
        ]
        chosen_model = st.selectbox("Select Model:", model_options, index=0)
        st.session_state["claude_model"] = chosen_model

        # Check API key from env/secrets
        env_has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or (hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets))
        if not env_has_key:
            st.caption("🔑 Enter Anthropic API Key (Session only):")
            input_key = st.text_input("API Key:", type="password", value=st.session_state["user_anthropic_key"])
            if input_key != st.session_state["user_anthropic_key"]:
                st.session_state["user_anthropic_key"] = input_key
                st.rerun()
        else:
            st.caption("✅ `ANTHROPIC_API_KEY` detected from Cloud Secrets.")

    st.divider()

    st.markdown("### ⚙️ Cloud Health Diagnostics")
    if health["llm_online"]:
        st.success(f"🟢 {health['llm_service']} Online")
    else:
        st.warning(f"🟡 {health['llm_service']} Needs Configuration")

    st.caption(f"**Embedding:** `{health['embedding_service']}`")
    st.caption(f"**Vector Store:** `{health['vector_store_provider']}`")
    st.caption(f"**Indexed Chunks:** `{stats['total_chunks']}` across `{len(stats['unique_documents'])}` docs")

    st.divider()

    st.markdown("### 📚 Indexed Documents")
    if stats["unique_documents"]:
        for doc in stats["unique_documents"]:
            st.markdown(f"📄 `{doc}`")
    else:
        st.warning("No documents indexed.")

    col_idx1, col_idx2 = st.columns(2)
    with col_idx1:
        if st.button("🔄 Re-index All", use_container_width=True):
            with st.spinner("Re-indexing PDF documents in data/ ..."):
                res = rag_service.index_directory("./data", force_reindex=True)
                st.success(f"Indexed {res.get('indexed_chunks', 0)} chunks!")
                st.rerun()

    with col_idx2:
        if st.button("🗑️ Reset DB", use_container_width=True):
            rag_service.reset_index()
            st.warning("Vector database reset.")
            st.rerun()

    st.divider()

    st.markdown("### 📤 Ingest New PDF")
    uploaded_file = st.file_uploader("Upload a College PDF", type=["pdf"])
    if uploaded_file is not None:
        save_path = os.path.join("./data", uploaded_file.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Processing {uploaded_file.name}..."):
                res = rag_service.index_pdf(save_path)
                if res.get("status") == "success":
                    st.success(f"Added {uploaded_file.name} ({res.get('indexed_chunks', 0)} chunks)")
                    st.rerun()
                elif res.get("status") == "warning":
                    st.warning(f"⚠️ {res.get('message')}")
                else:
                    st.error(f"❌ {res.get('message')}")

    st.divider()
    st.caption("🚀 **DocuMind AI** | Cloud RAG Assistant")


# ----------------- MAIN CONTENT -----------------
st.markdown('<div class="main-header">🎓 College Document Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cloud-Native Retrieval-Augmented Generation (RAG) powered by <b>Anthropic Claude</b> & ChromaDB / Qdrant | Tracked via Jira Scrum</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Ask Assistant",
    "☁️ Cloud Architecture & Diagnostics",
    "📊 Jira Scrum Project",
    "📑 Document Knowledge Base",
    "📈 Evaluation & Metrics"
])

# ----------------- TAB 1: ASK ASSISTANT -----------------
with tab1:
    st.markdown("#### Ask questions about college regulations, exams, placement, or campus conduct:")

    # Quick test question presets
    presets = [
        "Select a sample question...",
        "What is the minimum attendance required to appear for the semester examination?",
        "How many classes can I miss?",
        "How many credits are required for graduation?",
        "What is the procedure for examination revaluation?",
        "What are the rules for backlog examination?",
        "What is the eligibility criterion for campus placement?",
        "What documents are required for semester registration?",
        "What is the fine for overdue library books?",
        "What is the hostel fee?"  # Unanswerable test question (anti-hallucination test)
    ]

    selected_preset = st.selectbox("💡 Quick Test Presets:", presets, index=0)

    # Input query
    query_input = st.text_input(
        "Enter your question:",
        value=selected_preset if selected_preset != presets[0] else "",
        placeholder="e.g., What is the minimum attendance required?"
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        ask_clicked = st.button("🔍 Ask Question", type="primary", use_container_width=True)

    if ask_clicked and query_input:
        start_time = time.time()
        with st.spinner(f"Retrieving context and querying {health['llm_service']}..."):
            result = rag_service.answer_question(query_input)
            elapsed_time = round(time.time() - start_time, 2)

        st.markdown("### 📝 Answer")
        if result.get("is_error"):
            st.error(f"❌ {result['answer']}")
            if "Anthropic API key is not configured" in result["answer"]:
                st.info("💡 **Setup Instructions:** Configure `ANTHROPIC_API_KEY` in Streamlit Cloud Secrets (share.streamlit.io), enter it in the sidebar, or switch to the Local/Mock provider.")
        elif result["is_refusal"]:
            st.warning(f"⚠️ **{result['answer']}**")
            st.caption("ℹ️ The requested information was not found in the indexed college documents. The system adheres to strict zero-hallucination policies.")
        else:
            st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

            if result["sources"]:
                st.markdown("#### 📌 Verified Source Document(s):")
                for src in result["sources"]:
                    st.markdown(f'<div class="source-box">📄 <b>Source:</b> {src}</div>', unsafe_allow_html=True)

        st.caption(f"⏱️ Generated in {elapsed_time}s via `{health['llm_service']}` ({health['llm_model']}) & `{health['embedding_service']}`")

        # Retrieved Context Accordion
        with st.expander("🔍 View Retrieved Context & Similarity Details"):
            st.markdown(f"**Retrieved {len(result['retrieved_chunks'])} document chunk(s):**")
            for idx, chunk in enumerate(result["retrieved_chunks"], 1):
                st.markdown(f"**Chunk #{idx} | Source:** `{chunk.get('source')}` | **Page:** `{chunk.get('page')}` | **Similarity Score:** `{chunk.get('similarity')}`")
                st.info(chunk.get("text"))


# ----------------- TAB 2: CLOUD ARCHITECTURE & DIAGNOSTICS -----------------
with tab2:
    st.markdown("### ☁️ Cloud Platforms & Service Integrations")
    st.markdown("Status and configuration of the 6 integrated cloud components:")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown("""
        <div class="status-card">
            <h4>🌐 Hosting Platform</h4>
            <p><b>Platform:</b> Streamlit Community Cloud</p>
            <p><b>Status:</b> 🟢 Ready for Deploy</p>
            <p><b>Tier:</b> 100% Free Forever</p>
            <p><b>URL:</b> Public Cloud Webhook</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        claude_stat = "🟢 Connected" if health["llm_online"] else "🟡 Awaiting API Key"
        st.markdown(f"""
        <div class="status-card">
            <h4>🧠 Cloud LLM</h4>
            <p><b>Provider:</b> Anthropic Claude</p>
            <p><b>Status:</b> {claude_stat}</p>
            <p><b>Model:</b> {health['llm_model']}</p>
            <p><b>Auth:</b> Streamlit Secrets / Env</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c3:
        st.markdown(f"""
        <div class="status-card">
            <h4>🗄️ Vector Database</h4>
            <p><b>Provider:</b> {stats.get('provider')}</p>
            <p><b>Status:</b> 🟢 Persistent</p>
            <p><b>Storage:</b> Git-Backed Persistent</p>
            <p><b>Chunks:</b> {stats['total_chunks']} Indexed</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📋 Integration Matrix")
    integration_matrix = [
        {"Platform": "A. Streamlit Cloud", "Role": "Production Web Hosting", "Status": "Configured & Deployable", "Cost": "₹0 (Free Forever)", "Notes": "Auto-deploys from GitHub repository on git push"},
        {"Platform": "B. Anthropic Claude", "Role": "Cloud LLM Answer Synthesis", "Status": "Integrated via Python SDK", "Cost": "Pay-as-you-go API", "Notes": "Configured with strict anti-hallucination prompt"},
        {"Platform": "C. Cloud Vector Store", "Role": "Semantic Index Persistence", "Status": "ChromaDB Persistent / Qdrant", "Cost": "₹0 (Free Tier)", "Notes": "Repository-backed files persist across restarts"},
        {"Platform": "D. Cloud Document Storage", "Role": "PDF Persistence", "Status": "GitHub Repo & Session State", "Cost": "₹0 (Free)", "Notes": "Pre-indexed college documents stored in repo"},
        {"Platform": "E. GitHub", "Role": "Source Control & CI/CD", "Status": "Repository Connected", "Cost": "₹0 (Free)", "Notes": "Remote: co2023smniyazi/DocuMind-AI"},
        {"Platform": "F. Jira Cloud", "Role": "Agile Scrum Management", "Status": "Backlog CSV & Guide Ready", "Cost": "₹0 (Free Tier)", "Notes": "Project Key: RAG, 6 Epics, 4 Sprints, 22 Items"}
    ]
    st.dataframe(pd.DataFrame(integration_matrix), use_container_width=True)

    st.markdown("#### 🔒 Security & Secrets Configuration")
    st.markdown("""
    To configure secrets in Streamlit Community Cloud:
    1. Navigate to your app dashboard on [share.streamlit.io](https://share.streamlit.io).
    2. Click **App Settings > Secrets**.
    3. Add your secrets in TOML format:
    ```toml
    ANTHROPIC_API_KEY = "sk-ant-api03-..."
    CLAUDE_MODEL = "claude-3-5-haiku-20241022"
    LLM_PROVIDER = "claude"
    EMBEDDING_PROVIDER = "cloud"
    ```
    """)


# ----------------- TAB 3: JIRA SCRUM PROJECT -----------------
with tab3:
    st.markdown("### 📋 Jira Project Tracking (Project Key: RAG)")
    st.markdown("This project is managed and tracked according to the Scrum framework across 4 one-week sprints.")

    # Jira Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Project Name", "College RAG")
    col2.metric("Project Key", "RAG")
    col3.metric("Total Epics", "6 Epics")
    col4.metric("Backlog Items", "22 Items")
    col5.metric("Total Story Points", "90 SP")

    st.markdown("#### 🚀 Jira Workflow State")
    st.code("TO DO  ➡️  IN PROGRESS  ➡️  CODE REVIEW  ➡️  TESTING  ➡️  DONE", language="text")

    # Epics overview
    epics_data = [
        {"Epic ID": "EPIC-1", "Name": "Requirements and Dataset", "Status": "DONE", "Story Points": 11},
        {"Epic ID": "EPIC-2", "Name": "LLM & Cloud Setup", "Status": "DONE", "Story Points": 10},
        {"Epic ID": "EPIC-3", "Name": "RAG Pipeline & Claude AI", "Status": "DONE", "Story Points": 42},
        {"Epic ID": "EPIC-4", "Name": "User Interface & Cloud Hosting", "Status": "DONE", "Story Points": 13},
        {"Epic ID": "EPIC-5", "Name": "Testing and Evaluation", "Status": "DONE", "Story Points": 15},
        {"Epic ID": "EPIC-6", "Name": "Documentation and Demonstration", "Status": "IN PROGRESS", "Story Points": 5},
    ]
    st.dataframe(pd.DataFrame(epics_data), use_container_width=True)

    st.markdown("#### 📅 Sprint Breakdown")
    sprint_tabs = st.tabs(["Sprint 1: Foundation", "Sprint 2: RAG Pipeline", "Sprint 3: Cloud & Claude", "Sprint 4: Evaluation & Demo"])

    with sprint_tabs[0]:
        st.markdown("""
        - `RAG-1` Define functional requirements (Story - 3 SP) ✅ DONE
        - `RAG-2` Collect college documents (Task - 3 SP) ✅ DONE
        - `RAG-3` Prepare 50 sample questions (Task - 5 SP) ✅ DONE
        - `RAG-4` Setup LLM environment (Task - 2 SP) ✅ DONE
        - `RAG-5` Run and test baseline LLM (Task - 3 SP) ✅ DONE
        """)

    with sprint_tabs[1]:
        st.markdown("""
        - `RAG-6` Develop PDF text extractor (Story - 3 SP) ✅ DONE
        - `RAG-7` Develop text chunking module (Story - 3 SP) ✅ DONE
        - `RAG-8` Generate vector embeddings (Story - 5 SP) ✅ DONE
        - `RAG-9` Create vector database (Story - 5 SP) ✅ DONE
        - `RAG-10` Develop semantic retrieval (Story - 5 SP) ✅ DONE
        """)

    with sprint_tabs[2]:
        st.markdown("""
        - `RAG-11` Integrate retrieved context with LLM (Story - 5 SP) ✅ DONE
        - `RAG-12` Develop RAG prompt with anti-hallucination (Story - 3 SP) ✅ DONE
        - `RAG-13` Implement source citation (Story - 3 SP) ✅ DONE
        - `RAG-14` Implement unknown-answer handling (Story - 3 SP) ✅ DONE
        - `RAG-15` Create Streamlit UI (Story - 5 SP) ✅ DONE
        - `RAG-16` Display retrieved sources (Story - 3 SP) ✅ DONE
        - `RAG-21` Cloud deployment configuration for Streamlit Cloud (Story - 5 SP) ✅ DONE
        - `RAG-22` Anthropic Claude API integration (Story - 5 SP) ✅ DONE
        """)

    with sprint_tabs[3]:
        st.markdown("""
        - `RAG-17` Perform functional testing (Task - 5 SP) ✅ DONE
        - `RAG-18` Evaluate LLM responses across 50 questions (Task - 5 SP) ✅ DONE
        - `RAG-19` Fix identified defects (Bug/Task - 5 SP) ✅ DONE
        - `RAG-20` Prepare documentation and cloud demonstration (Task - 5 SP) 🔄 IN PROGRESS
        """)


# ----------------- TAB 4: DOCUMENT KNOWLEDGE BASE -----------------
with tab4:
    st.markdown("### 📑 College Document Knowledge Base")
    st.markdown("Inspect all college documents currently indexed in ChromaDB:")

    if os.path.exists("./data"):
        doc_files = [f for f in sorted(os.listdir("./data")) if f.endswith(".pdf")]
        for doc_file in doc_files:
            file_path = os.path.join("./data", doc_file)
            file_size_kb = round(os.path.getsize(file_path) / 1024, 1)

            with st.expander(f"📄 {doc_file} ({file_size_kb} KB)"):
                pages, summary = load_pdf_with_metadata(file_path)
                st.markdown(f"**Total Pages:** {summary['total_pages']} | **Total Characters:** {summary['total_chars']}")
                if summary.get("is_scanned"):
                    st.warning("⚠️ Scanned Image Document (no digital text layer).")
                for p in pages:
                    st.markdown(f"**Page {p['page']} Text Preview:**")
                    st.text(p['text'])


# ----------------- TAB 5: EVALUATION & METRICS -----------------
with tab5:
    st.markdown("### 📈 System Evaluation & Metrics (STEP 21-23, 44-47)")
    st.markdown("Comprehensive evaluation across **50 benchmark questions** categorized into Direct, Paraphrased, and Unanswerable queries.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Category A (Direct)", "20 Questions", "Retrieval Testing")
    col2.metric("Category B (Paraphrased)", "20 Questions", "Semantic Testing")
    col3.metric("Category C (Unanswerable)", "10 Questions", "Safe Refusal Testing")

    st.markdown("#### 🎯 Evaluation Rubric")
    st.markdown("""
    | Metric | Meaning | Target Score |
    |---|---|---|
    | **Correctness** | Is the answer factually supported by the document? | 5 / 5 |
    | **Relevance** | Does it answer the specific question asked? | 5 / 5 |
    | **Groundedness** | Is it strictly based on retrieved context (no hallucinations)? | 5 / 5 |
    | **Clarity** | Is it understandable and grammatically well-formed? | 5 / 5 |
    | **Completeness** | Is sufficient information provided? | 5 / 5 |
    """)

    if os.path.exists("./evaluation_report.md"):
        st.markdown("#### 📄 Latest Benchmark Report")
        with open("./evaluation_report.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Run `python evaluate.py` to generate the latest benchmark report.")
