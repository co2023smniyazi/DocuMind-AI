# DocuMind AI: Cloud-Native College Document Q&A Assistant Using RAG & Claude AI
### Project Management and Tracking Using Jira Scrum Framework

[![DocuMind AI CI Test Suite](https://github.com/co2023smniyazi/DocuMind-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/co2023smniyazi/DocuMind-AI/actions/workflows/ci.yml)
[![Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Anthropic Claude](https://img.shields.io/badge/LLM-Anthropic%20Claude-8A2BE2.svg)](https://www.anthropic.com/)
[![Jira Scrum](https://img.shields.io/badge/Jira-Scrum%20RAG-0052CC.svg)](https://www.atlassian.com/software/jira)

---

## 1. Project Overview & Cloud Architecture

**DocuMind AI** is a cloud-native Retrieval-Augmented Generation (RAG) assistant designed to allow college students and faculty to query official academic policies, regulations, examination guidelines, student handbooks, and placement rules using natural language.

### Cloud-Native Redesign:
- **Cloud Hosting Platform:** **Streamlit Community Cloud** (`share.streamlit.io`) — 100% free cloud deployment hosted directly from the GitHub repository without requiring local computer hosting.
- **Cloud LLM Provider:** **Anthropic Claude** (`claude-3-5-sonnet-20241022` and `claude-3-5-haiku-20241022`) via the official `anthropic` Python SDK.
- **Local Fallback Engine:** Local Ollama (`llama3.2:3b` and `nomic-embed-text`) remains switchable for offline development.
- **Persistent Document Storage:** Pre-indexed college PDF documents and vector catalogs are version-controlled within the Git repository, guaranteeing zero data loss across container reboots and redeployments.
- **Cloud Vector Database Adapter:** Pluggable vector store architecture supporting repository-backed **ChromaDB** and **Qdrant Cloud Free Tier** (1GB permanent free cluster).
- **Zero-Hallucination Policy:** If an answer cannot be deduced from the retrieved documents, the assistant strictly returns:
  > *"Information not found in the available documents."*

```mermaid
flowchart TD
    subgraph CLOUD_STORAGE["1. Cloud Document & Vector Storage"]
        A["College PDF Documents\n(academic_regulations.pdf,\nexamination_guidelines.pdf, etc.)"] --> B["PyPDF Document Loader\n(document_loader.py)"]
        B --> C["Text Chunking Module\n(chunker.py: 1500 chars, 200 overlap)"]
        C --> D["Cloud-Native Feature Embedder\n(embeddings.py: 768-dim Unit Vectors)"]
        D --> E[("Repository-Backed Persistent ChromaDB\nor Qdrant Cloud Free Tier")]
    end

    subgraph CLOUD_RAG["2. Cloud RAG Inference Pipeline"]
        F["Student Enters Query\n(Streamlit Community Cloud / app.py)"] --> G["Query Vector Embedding"]
        G --> H["Semantic Vector Search\n(retriever.py: Top-4 Chunks)"]
        E -.->|Similarity Match| H
        H --> I["Augmented Prompt Assembly\n(Context + Source Tags + Anti-Hallucination)"]
        I --> J["Anthropic Claude API\n(claude-3-5-haiku or claude-3-5-sonnet)"]
        J --> K["Verified Answer & Source Citations\n(Streamlit Cloud UI)"]
    end
```

---

## 2. Six Platform Integrations Matrix

| Platform | Role | Integration Type | Free-Tier Limits & Costs | Status |
|---|---|---|---|---|
| **A. Streamlit Cloud** | Production Web Hosting | Web Application Platform | 100% Free Forever (Public GitHub repos, 1 GB RAM container) | **PASS** (Configured & deployable) |
| **B. Anthropic Claude API** | Cloud LLM Answer Synthesis | Official `anthropic` Python SDK | Pay-as-you-go API ($0.25/$1.25 per MTok for Haiku). Requires API key. | **PASS** (Integrated with error guards) |
| **C. Cloud Vector Store** | Semantic Index Persistence | ChromaDB Persistent / Qdrant Cloud | ₹0 (Repository-backed ChromaDB) / 1GB Free Cluster (Qdrant Cloud) | **PASS** (Verified duplicate-safe) |
| **D. Cloud Document Storage** | PDF Persistence | Git Repository & Session State | ₹0 (Free up to GitHub 2GB repo size) | **PASS** (5 college PDFs committed) |
| **E. GitHub** | Source Control & CI/CD | Git Remote + GitHub Actions | Free GitHub Actions runners (2000 mins/month) | **PASS** (Repo: `DocuMind-AI`) |
| **F. Jira Cloud** | Scrum Project Management | Atlassian Cloud Scrum | ₹0 (Free up to 10 users) | **PASS** (Backlog CSV & Guide ready) |

---

## 3. Security and Secret Management

> [!CAUTION]
> **Never commit API keys or passwords to GitHub!**

All credentials are read dynamically using the `get_secret(key)` utility, checking:
1. Streamlit Community Cloud Secrets (`st.secrets["ANTHROPIC_API_KEY"]`)
2. Local `.env` files (ignored by `.gitignore`)
3. OS environment variables (`os.environ["ANTHROPIC_API_KEY"]`)

### How to Configure Secrets in Streamlit Community Cloud:
1. Open your application dashboard at [share.streamlit.io](https://share.streamlit.io).
2. Click **Settings > Secrets**.
3. Paste the following configuration:
```toml
# Streamlit Cloud Secrets Configuration
ANTHROPIC_API_KEY = "sk-ant-api03-your-real-key-here"
CLAUDE_MODEL = "claude-3-5-haiku-20241022"
LLM_PROVIDER = "claude"
EMBEDDING_PROVIDER = "cloud"
VECTOR_STORE_PROVIDER = "chroma"
```

---

## 4. Directory Structure

```
D:\New folder\cc project 3\
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow (automated tests on push)
│
├── .streamlit/
│   ├── config.toml             # Production Streamlit Cloud server & theme configuration
│   └── secrets.toml.example    # Template for Streamlit Cloud secrets
│
├── .gitignore                  # Security guard preventing commit of keys, envs, caches
├── .env.example                # Local environment secrets template
│
├── app.py                      # Multi-tab Streamlit Cloud UI with diagnostics & provider switch
├── document_loader.py          # PyPDF text extractor with scanned PDF & error guards (RAG-6)
├── chunker.py                  # Text chunking with overlap & boundary parameter guards (RAG-7)
├── embeddings.py               # Cloud-native embedder & Ollama nomic-embed-text (RAG-8)
├── vector_store.py             # Repository-backed ChromaDB & Qdrant Cloud adapter (RAG-9)
├── retriever.py                # Semantic retriever with top-k scoring (RAG-10)
├── llm_service.py              # Anthropic Claude & Ollama unified service (RAG-11, 22)
├── rag_service.py              # Cloud RAG orchestrator with citation & safe refusal (RAG-13, 14, 16)
├── evaluate.py                 # Benchmark runner supporting Claude, Ollama, and Mock modes
│
├── data/                       # Official College Policy PDF Documents (Persistent)
│   ├── academic_regulations.pdf
│   ├── examination_guidelines.pdf
│   ├── student_handbook.pdf
│   ├── placement_guidelines.pdf
│   └── department_rules.pdf
│
├── chroma_db/                  # Pre-computed persistent ChromaDB vector storage
│
├── tests/
│   ├── test_loader.py          # PyPDF loader unit tests
│   ├── test_chunker.py         # Chunking boundary unit tests
│   ├── test_mocked_rag.py      # Offline RAG integration tests (fast CI)
│   ├── test_claude.py          # Anthropic Claude SDK & error handling unit tests
│   ├── test_live_rag.py        # Live Ollama integration tests
│   └── test_dataset.json       # 50 sample questions (Category A, B, C)
│
├── jira_management/
│   ├── jira_backlog.csv        # 22 Jira items (RAG-1 to RAG-22) ready for 1-click import
│   ├── JIRA_PROJECT_GUIDE.md   # Step-by-step Scrum setup guide for Jira Free
│   └── sprints_and_board.md    # 4-sprint breakdown & defect resolution tracker
│
├── requirements.txt            # Python dependencies
└── README.md                   # Complete system documentation
```

---

## 5. Deployment Instructions

### Step 1: Push Code to Your GitHub Repository
```bash
git add .
git commit -m "feat: complete cloud-only deployment and Anthropic Claude AI integration"
git push origin main
```

### Step 2: Deploy to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your repository: `co2023smniyazi/DocuMind-AI`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **Advanced settings > Secrets** and paste your `ANTHROPIC_API_KEY`.
7. Click **Deploy!** Your app will be live at `https://documind-ai.streamlit.app` in 2 minutes!

---

## 6. Testing Guide

### Run Offline Unit Tests (20 Tests — No API Keys Needed)
```bash
python -m pytest tests/test_loader.py tests/test_chunker.py tests/test_mocked_rag.py tests/test_claude.py -v
```

### Run Live Ollama Integration Tests (When Ollama is Running)
```bash
python -m pytest tests/test_live_rag.py -v
```

### Run 50-Question Benchmark
```bash
# Evaluate with Mock provider (fast, offline)
python evaluate.py --sample 10 --provider mock

# Evaluate with Claude (requires ANTHROPIC_API_KEY)
python evaluate.py --provider claude
```

---

## 7. Limitations and Cost Considerations

1. **Claude API Billing:** Claude is a commercial API. While `claude-3-5-haiku` is inexpensive (~$0.0003 per question), usage requires an active credit balance in the Anthropic Console. If no key is set, the app cleanly notifies the user without crashing.
2. **Scanned Image PDFs:** PDFs consisting purely of image scans without digital font layers cannot be extracted by PyPDF. An OCR pipeline (e.g., Tesseract) is recommended for scanned paper archives.
3. **Streamlit Cloud RAM Limit:** Free Streamlit Community Cloud containers provide 1 GB of memory. The self-contained cloud embedder was deliberately engineered to consume <10 MB of RAM, leaving maximum headroom.
