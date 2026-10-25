# DocuMind AI: Cloud-Native College Document Q&A Assistant Using RAG
### Free-Tier Native Architecture (₹0 Budget) | Tracked Using Jira Scrum Framework

[![DocuMind AI CI Test Suite](https://github.com/co2023smniyazi/DocuMind-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/co2023smniyazi/DocuMind-AI/actions/workflows/ci.yml)
[![Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastEmbed ONNX](https://img.shields.io/badge/Embeddings-FastEmbed%20ONNX-orange.svg)](https://qdrant.github.io/fastembed/)
[![Google Gemini](https://img.shields.io/badge/Default%20LLM-Google%20Gemini%202.0%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Qdrant Cloud](https://img.shields.io/badge/Vector%20DB-Qdrant%20Cloud%201GB%20Free-red.svg)](https://cloud.qdrant.io/)
[![Jira Scrum](https://img.shields.io/badge/Jira-Scrum%20RAG-0052CC.svg)](https://www.atlassian.com/software/jira)

---

## 1. Project Overview & Architecture

**DocuMind AI** is a production-ready, cloud-native Retrieval-Augmented Generation (RAG) assistant designed for college students and faculty to query academic regulations, examination policies, code of conduct, and placement criteria using natural language.

### Core Architectural Pillars (Strict ₹0 Budget Policy):
- **Web UI Hosting:** **Streamlit Community Cloud** (`share.streamlit.io`) — Free community tier hosting deployed directly from GitHub (1 GB container RAM limit).
- **Default Cloud LLM:** **Google Gemini API** (`gemini-2.0-flash` / `gemini-1.5-flash`) via the modern `google-genai` SDK. Free tier provides 15 RPM, 1,000,000 TPM, and 1,500 RPD at ₹0 cost.
- **Secondary Cloud LLM:** **Groq Cloud** (`llama-3.3-70b-versatile`) via the official `groq` SDK for ultra-fast fallback.
- **Semantic Vector Embedder:** **FastEmbed ONNX** (`BAAI/bge-small-en-v1.5`, 384 dimensions) running locally inside the container via ONNX Runtime (~67 MB weights, ~150 MB RAM). Zero API calls, zero cost, real semantic similarity.
- **Cloud Vector Database:** **Qdrant Cloud Free Tier** (1 GB managed cluster) via `qdrant-client` 1.20+. Persistent payload storage for chunk text and metadata.
- **Local / Test Vector DB:** Persistent **ChromaDB** and in-memory stores for offline testing and local development.
- **In-Memory Upload Ingestion:** Uploaded PDFs are parsed in-memory and stored directly inside vector database payloads. No user PDFs are written to container disk (preventing loss on reboot). Bundled sample college PDFs remain in `data/` as seed data.
- **Optional LLM (OFF by Default):** **Anthropic Claude** (`claude-3-5-haiku-20241022`) is supported via the official `anthropic` SDK, but strictly kept disabled by default (`ENABLE_CLAUDE=false`) to ensure ₹0 expenditure.
- **Zero Hallucination Policy:** When retrieved context does not answer the question, the assistant strictly outputs:
  > *"Information not found in the available documents."*

```mermaid
flowchart TD
    subgraph INGESTION["1. In-Memory Document Ingestion & Chunking"]
        A["Official College PDFs\n(data/ or User Upload)"] --> B["PyPDF In-Memory Loader\n(load_pdf_from_bytes)"]
        B --> C["Recursive Character Chunker\n(chunk_size=1500, overlap=200)"]
        C --> D["FastEmbed ONNX Embedder\n(BAAI/bge-small-en-v1.5, 384-dim)"]
        D --> E[("Qdrant Cloud Free Tier 1GB\n(or ChromaDB Local Fallback)")]
    end

    subgraph INFERENCE["2. Cloud RAG Inference Pipeline"]
        F["Student Enters Query\n(Streamlit Web UI / app.py)"] --> G["Query Vector Embedding\n(FastEmbed 384-dim)"]
        G --> H["Semantic Vector Retrieval\n(Top-K Chunks + Payload Metadata)"]
        E -.->|Vector Match| H
        H --> I["Anti-Hallucination Prompt Assembly\n(Context + Source Citations)"]
        I --> J{"Configured Cloud LLM"}
        J -->|Default (Free)| K["Google Gemini 2.0 Flash\n(15 RPM, 1M TPM, 1,500 RPD)"]
        J -->|Secondary (Free)| L["Groq Cloud\n(llama-3.3-70b-versatile)"]
        J -->|Optional (Paid)| M["Anthropic Claude\n(OFF by default)"]
        K --> N["Verified Answer & Page Citations\n(Streamlit UI)"]
        L --> N
        M --> N
    end
```

---

## 2. Cloud Platforms & Verified Free Tier Matrix

All limits have been verified against official platform documentation as of October 2026. For complete limit specifications and data privacy policies, see [docs/FREE_TIER_LIMITS.md](file:///docs/FREE_TIER_LIMITS.md).

| Platform | Role | SDK / Integration | Free-Tier Limits & Quotas | Verified Doc URL |
|---|---|---|---|---|
| **Streamlit Community Cloud** | Production Web Hosting | Web Platform | ₹0 (1 GB RAM / app, sleeps after 7 days inactivity) | [Streamlit Limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resource-limits) |
| **Google Gemini API** | Default Cloud LLM | `google-genai` | ₹0 (15 RPM, 1M TPM, 1,500 RPD; data logged for training on free tier) | [Gemini Pricing](https://ai.google.dev/pricing) |
| **Groq Cloud** | Secondary Cloud LLM | `groq` | ₹0 (30 RPM, 6k-30k TPM) | [Groq Rate Limits](https://console.groq.com/docs/rate-limits) |
| **FastEmbed (ONNX)** | Semantic Embeddings | `fastembed` | ₹0 (In-container, 384 dims, ~67 MB weights, ~150 MB RAM) | [FastEmbed Docs](https://qdrant.github.io/fastembed/) |
| **Qdrant Cloud** | Cloud Vector DB | `qdrant-client` | ₹0 (1 GB permanent free cluster, suspends after 7 days inactivity) | [Qdrant Pricing](https://qdrant.tech/pricing/) |
| **ChromaDB** | Local / Offline Vector DB | `chromadb` | ₹0 (Local disk storage for development & CI fallback) | Local |
| **GitHub Actions** | Source Control & CI/CD | GitHub Actions | ₹0 (2,000 runner minutes/month) | [GitHub Billing](https://docs.github.com/en/billing) |
| **Anthropic Claude** | Optional Cloud LLM | `anthropic` | Paid API (Disabled by default: `ENABLE_CLAUDE=false`) | [Anthropic Pricing](https://www.anthropic.com/pricing) |
| **Jira Cloud** | Agile Scrum Management | Atlassian Scrum | ₹0 (Free up to 10 users; 6 Epics, 4 Sprints, 22 Items) | [Jira Pricing](https://www.atlassian.com/software/jira/pricing) |

---

## 3. Directory Structure

```
D:\New folder\cc project 3\
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow (automated test suite)
│
├── .streamlit/
│   ├── config.toml             # Streamlit Cloud server & UI styling configuration
│   └── secrets.toml.example    # Verified template for Streamlit Cloud secrets
│
├── docs/
│   └── FREE_TIER_LIMITS.md     # Verified quotas, doc URLs, privacy terms, suspension rules
│
├── pages/
│   └── Health_Check.py         # Automated diagnostics for Ingestion, Embeddings, DB, LLM
│
├── data/                       # Bundled Seed College PDF Documents (Read-Only)
│   ├── academic_regulations.pdf
│   ├── examination_guidelines.pdf
│   ├── student_handbook.pdf
│   ├── placement_guidelines.pdf
│   └── department_rules.pdf
│
├── tests/
│   ├── test_chunker.py          # Text chunking boundary & overlap tests
│   ├── test_claude.py           # Claude SDK, rate limit, & OFF-by-default tests
│   ├── test_cloud_rag.py        # FastEmbed, Qdrant in-memory, Gemini/Groq tests
│   ├── test_live_rag.py         # Live Ollama integration tests (skips if offline)
│   ├── test_loader.py           # PyPDF loader, corruption, and bytes parsing tests
│   ├── test_mocked_rag.py       # Deterministic end-to-end RAG unit tests
│   ├── test_rag.py              # Regression test suite
│   └── test_dataset.json        # 50 benchmark questions (Categories A, B, C)
│
├── jira_management/
│   ├── jira_backlog.csv         # 22 Jira items (RAG-1 to RAG-22) ready for import
│   ├── JIRA_PROJECT_GUIDE.md    # Complete Scrum project setup guide
│   └── sprints_and_board.md     # 4-sprint breakdown & board states
│
├── app.py                      # Multi-tab Streamlit UI with In-Memory Ingestion
├── document_loader.py          # PyPDF text loader with bytes support (load_pdf_from_bytes)
├── chunker.py                  # Chunking with boundary parameters & overlap
├── embeddings.py               # FastEmbed ONNX, Gemini, Ollama, & Deterministic Embedders
├── vector_store.py             # Qdrant Cloud Free Tier & ChromaDB adapters
├── retriever.py                # Semantic top-k retrieval with similarity scoring
├── llm_service.py              # Gemini, Groq, Claude, Ollama, & Mock LLM providers
├── rag_service.py              # Complete RAG pipeline with safe refusal & citation
├── evaluate.py                 # 50-question benchmark runner
├── requirements.txt            # Pinned production Python dependencies
└── README.md                   # System documentation
```

---

## 4. Deployment to Streamlit Community Cloud

### Step 1: Push Code to Branch `free-cloud-deploy`
```bash
git add .
git commit -m "feat: implement free cloud deployment with FastEmbed, Gemini, and Qdrant"
git push origin free-cloud-deploy
```

### Step 2: Deploy on Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository: `co2023smniyazi/DocuMind-AI`.
4. Select branch: `free-cloud-deploy`.
5. Main file path: `app.py`.
6. Click **Advanced settings > Secrets** and paste your secrets:
```toml
# Primary Free Cloud LLM (Google Gemini)
GEMINI_API_KEY = "AIzaSy..."
GEMINI_MODEL = "gemini-2.0-flash"
LLM_PROVIDER = "gemini"

# Embeddings (Zero API calls, in-container ONNX)
EMBEDDING_PROVIDER = "fastembed"

# Optional Qdrant Cloud Cluster
# QDRANT_URL = "https://your-cluster-id.region.gcp.cloud.qdrant.io:6333"
# QDRANT_API_KEY = "your-qdrant-api-key"
# VECTOR_STORE_PROVIDER = "qdrant"

# Optional Claude (OFF by default to enforce ₹0 budget)
ENABLE_CLAUDE = "false"
```
7. Click **Deploy!** Your cloud assistant will be live within 2-3 minutes.

---

## 5. Live Health Diagnostics

DocuMind AI includes an automated diagnostics dashboard located at `pages/Health_Check.py` (accessible from the sidebar in the Streamlit app). It runs live probes on:
1. **PDF Ingestion:** Generates and verifies an in-memory synthetic PDF.
2. **Embeddings:** Runs `FastEmbedEmbedder` to verify 384-dimensional vector output.
3. **Vector Database:** Performs an automated probe lifecycle (Upsert ➡️ Similarity Search ➡️ Clean-up deletion).
4. **Cloud LLM:** Performs a rate-limit-aware live probe or credential health check.
5. **Secrets Audit:** Displays masked statuses for all configured environment variables.

---

## 6. Running the Automated Test Suite

All 34 automated unit and integration tests pass cleanly:

```bash
# Run the complete test suite
python -m pytest -v

# Run offline unit tests only (instantaneous, no Ollama required)
python -m pytest tests/test_loader.py tests/test_chunker.py tests/test_mocked_rag.py tests/test_claude.py tests/test_cloud_rag.py -v
```

---

## 7. Jira Scrum Tracking

Project Key: **`RAG`**  
Total Story Points: **90 SP** across **4 Sprints** and **6 Epics** (22 Jira items).  
Workflow: `TO DO` ➡️ `IN PROGRESS` ➡️ `CODE REVIEW` ➡️ `TESTING` ➡️ `DONE`

To import the backlog into your free Jira Cloud instance:
1. Go to Jira Cloud Settings ➡️ System ➡️ External System Import.
2. Select **CSV Import** and upload `jira_management/jira_backlog.csv`.
3. Follow the field mappings detailed in `jira_management/JIRA_PROJECT_GUIDE.md`.
