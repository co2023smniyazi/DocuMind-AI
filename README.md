# DocuMind-AI: Cloud-Native College Document Q&A Assistant Using RAG
### Free-Tier Native Architecture (₹0 Operating Budget) | Managed Under Jira Scrum Framework

[![DocuMind AI CI Suite](https://github.com/co2023smniyazi/DocuMind-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/co2023smniyazi/DocuMind-AI/actions/workflows/ci.yml)
[![Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastEmbed ONNX](https://img.shields.io/badge/Embeddings-FastEmbed%20ONNX%20(384d)-orange.svg)](https://qdrant.github.io/fastembed/)
[![Google Gemini](https://img.shields.io/badge/Primary%20LLM-Google%20Gemini%202.0%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Qdrant Cloud](https://img.shields.io/badge/Vector%20DB-Qdrant%20Cloud%201GB%20Free-red.svg)](https://cloud.qdrant.io/)
[![Jira Scrum](https://img.shields.io/badge/Jira-Scrum%20(6%20Epics%2C%2025%20Stories)-0052CC.svg)](https://www.atlassian.com/software/jira)

---

## 1. Project Overview & Architecture

**DocuMind-AI** is a cloud-native, scalable Retrieval-Augmented Generation (RAG) assistant designed for college students and faculty to query academic regulations, examination ordinances, codes of conduct, and placement criteria using natural language.

### Core Architectural Pillars (Strict ₹0 Budget Policy):
- **Web UI Hosting:** **Streamlit Community Cloud** (`share.streamlit.io`) — Free community tier hosting deployed directly from GitHub (1.0 GB container RAM limit).
- **Default Cloud LLM:** **Google Gemini API** (`gemini-2.0-flash`) via the modern `google-genai` SDK. Free tier provides 15 RPM, 1,000,000 TPM, and 1,500 RPD at ₹0 cost.
- **Secondary Cloud LLM:** **Groq Cloud** (`llama-3.3-70b-versatile`) via the official `groq` SDK for instant fallback on rate limits.
- **Semantic Vector Embedder:** **FastEmbed ONNX** (`BAAI/bge-small-en-v1.5`, 384 dimensions) running locally inside the container via ONNX Runtime (~67 MB weights, ~145 MB RAM). Zero API calls, zero cost, real dense semantic similarity.
- **Cloud Vector Database:** **Qdrant Cloud Free Tier** (1 GB permanent cluster) via `qdrant-client` 1.20+. Persistent payload storage for chunk text and metadata.
- **Local / Test Vector DB:** Persistent **ChromaDB** and in-memory stores for offline testing and local development.
- **In-Memory Upload Ingestion:** Uploaded PDFs are parsed in-memory and stored directly inside vector database payloads. No user PDFs are written to container disk (preventing loss on container reboot).
- **Optional Enterprise Adapters (OFF by Default):** **Anthropic Claude**, **AWS Bedrock (Claude)**, and **GCP Vertex AI (Claude)** adapters plus a Model Context Protocol (**MCP**) server are fully implemented and verified via mocked unit tests, but kept disabled (`ENABLE_CLAUDE=false`) to maintain our strict ₹0 budget constraint.
- **Zero Hallucination Policy:** When retrieved context does not answer the question, the assistant strictly outputs:
  > *"Information not found in the available documents."*

```mermaid
flowchart TD
    subgraph INGESTION["1. In-Memory Document Ingestion & Chunking"]
        A["Official College PDFs\n(data/ or User Upload)"] --> B["PyPDF In-Memory Loader\n(load_pdf_from_bytes)"]
        B --> C["Recursive Character Chunker\n(chunk_size=500, overlap=100)"]
        C --> D["FastEmbed ONNX Embedder\n(BAAI/bge-small-en-v1.5, 384-dim)"]
        D --> E[("Qdrant Cloud Free Tier 1GB\n(or ChromaDB Local Fallback)")]
    end

    subgraph INFERENCE["2. Cloud RAG Inference Pipeline"]
        F["Student Enters Query\n(Streamlit Web UI / app.py)"] --> G["Query Vector Embedding\n(FastEmbed 384-dim)"]
        G --> H["Semantic Vector Retrieval\n(Top-K Chunks + Payload Metadata)"]
        E -.->|Vector Match| H
        H --> I["Anti-Hallucination Prompt Assembly\n(Context + Source Citations)"]
        I --> J{"Fallback LLM Chain"}
        J -->|Primary (Free)| K["Google Gemini 2.0 Flash\n(15 RPM, 1M TPM, 1,500 RPD)"]
        J -->|Fallback 429 (Free)| L["Groq Cloud\n(llama-3.3-70b-versatile)"]
        J -->|Optional (Paid)| M["Anthropic Claude / Bedrock / Vertex\n(OFF by default)"]
        K --> N["Verified Answer & Page Citations\n(Streamlit UI)"]
        L --> N
        M --> N
    end
```

---

## 2. Cloud Platforms & Verified Free Tier Matrix

All limits have been verified against official platform documentation as of October 2026. For complete limit specifications, see [docs/FREE_TIER_LIMITS.md](file:///docs/FREE_TIER_LIMITS.md).

| Platform | Role | SDK / Integration | Free-Tier Limits & Quotas | Verified Doc URL |
|---|---|---|---|---|
| **Streamlit Community Cloud** | Production Web Hosting | Web Platform | ₹0 (Dynamic ~1.0 GB RAM [690MB–2.7GB], sleeps after **12 hours** inactivity) | [Streamlit Limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resource-limits) |
| **Google Gemini API** | Primary Cloud LLM | `google-genai` | ₹0 (15 RPM, 1M TPM, 1,500 RPD for `gemini-2.0-flash`) | [Gemini Pricing](https://ai.google.dev/pricing) |
| **Groq Cloud** | Secondary Cloud LLM | `groq` | ₹0 (30 RPM, 12k TPM, 1,000 RPD for `llama-3.3-70b`) | [Groq Rate Limits](https://console.groq.com/docs/rate-limits) |
| **FastEmbed (ONNX)** | Semantic Embeddings | `fastembed` | ₹0 (In-process, 384 dims, ~67 MB weights, ~145 MB process RAM) | [FastEmbed Docs](https://qdrant.github.io/fastembed/) |
| **Qdrant Cloud** | Cloud Vector DB | `qdrant-client` | ₹0 (1 GB RAM, 4 GB Disk; suspends after **1 week**, deleted after **4 weeks**) | [Qdrant Pricing](https://qdrant.tech/pricing/) |
| **ChromaDB** | Local / Offline Vector DB | `chromadb` | ₹0 (Local disk storage for development & CI fallback) | Local |
| **GitHub Actions** | Source Control & CI/CD | GitHub Actions | ₹0 (Unlimited public mins / 2,000 private mins/mo on Linux) | [GitHub Billing](https://docs.github.com/en/billing) |
| **Anthropic Claude** | Optional Cloud LLM | `anthropic` | Paid API (Disabled by default: `ENABLE_CLAUDE=false`) | [Anthropic Pricing](https://www.anthropic.com/pricing) |
| **Jira Cloud** | Agile Scrum Management | Atlassian Scrum | ₹0 (Free up to 10 users; 6 Epics, 25 User Stories) | [Jira Pricing](https://www.atlassian.com/software/jira/pricing) |

---

## 3. Directory Structure

```
D:\New folder\cc project 3\
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Secret scanning + offline unit & integration test suite
│       ├── evaluate.yml           # On-demand 50-question benchmark runner (workflow_dispatch)
│       └── benchmark.yml          # Synthetic throughput & latency benchmark runner
│
├── .streamlit/
│   ├── config.toml                # Streamlit Cloud server & UI styling configuration
│   └── secrets.toml.example       # Verified template for Streamlit Cloud secrets
│
├── docs/
│   ├── FREE_TIER_LIMITS.md        # Verified quotas, doc URLs, privacy terms, suspension rules
│   ├── BENCHMARKS.md              # Real empirical latency percentiles (P50, P90, P99) & throughput
│   ├── SCALABILITY.md             # Scalability architecture, quota mitigation & upgrade path
│   ├── SETUP_CHECKLIST.md         # Step-by-step click guide for free keys & Streamlit deployment
│   ├── DEMO_SCRIPT.md             # 5-minute structured viva presentation demo script
│   ├── REPORT_DRAFT.md            # Comprehensive academic report draft
│   └── INSPECTION_REPORT.md       # Initial system inspection & audit findings
│
├── evaluation/
│   ├── human_scoring_sheet.csv    # 50-question blank human scoring rubric template
│   └── final_evaluation_table.md  # Standardized evaluation & architecture comparison matrix
│
├── pages/
│   └── Health_Check.py            # Live diagnostics: Embedder, Vector Store, LLM, Secrets
│
├── data/                          # Bundled Seed College PDF Documents (Read-Only)
│   ├── academic_regulations.pdf
│   ├── examination_guidelines.pdf
│   ├── student_handbook.pdf
│   ├── placement_guidelines.pdf
│   └── department_rules.pdf
│
├── tests/
│   ├── test_chunker.py             # Text chunking boundary & overlap tests
│   ├── test_claude.py              # Claude SDK, rate limit, & OFF-by-default tests
│   ├── test_cloud_rag.py           # FastEmbed, Qdrant in-memory, Gemini/Groq, fallback tests
│   ├── test_live_rag.py            # Live integration tests (skips if offline)
│   ├── test_loader.py              # PyPDF loader, corruption, and bytes parsing tests
│   ├── test_mocked_rag.py          # Deterministic end-to-end RAG unit tests
│   ├── test_rag.py                 # Regression test suite
│   └── test_dataset.json           # 50 benchmark questions (Categories A, B, C)
│
├── scripts/
│   ├── benchmark.py                # Ingestion throughput & latency percentiles suite
│   └── generate_sample_pdfs.py     # PDF corpus generator for test fixtures
│
├── jira_management/
│   ├── jira_import.csv             # 25 Jira user stories across 6 Epics ready for import
│   ├── user_stories.md             # Detailed acceptance criteria for RAG-6, RAG-10, RAG-11, etc.
│   └── status.json                 # Machine-readable sprint metrics & velocity
│
├── app.py                         # Multi-tab Streamlit UI with In-Memory Ingestion
├── document_loader.py             # PyPDF text loader with bytes support
├── chunker.py                     # Recursive character chunker
├── embeddings.py                  # FastEmbed ONNX, Gemini, Ollama, & Test Embedders
├── vector_store.py                # Qdrant Cloud Free Tier & ChromaDB adapters
├── retriever.py                   # Semantic retrieval with deduplication & similarity threshold
├── llm_service.py                 # Gemini, Groq, Claude, Bedrock, Vertex, FallbackLLMChain
├── rag_service.py                 # Complete RAG pipeline with Answer Cache & token budgeting
├── mcp_server.py                  # Model Context Protocol server exposing RAG tools
├── evaluate.py                    # 50-question non-destructive benchmark evaluation engine
├── requirements.txt               # Lean production dependencies
├── requirements-claude.txt        # Optional multi-cloud Claude adapter dependencies
└── README.md                      # System documentation
```

---

## 4. Empirical Performance Benchmarks (Local Workstation Measurements)

> [!NOTE]
> Measured **locally on a Windows 11 PC (Python 3.11.9)** on 2026-10-25 using `python scripts/benchmark.py --sizes 250 500 1000 --queries 30 --vector-store chroma`.  
> These numbers reflect local SSD and in-process ONNX execution without network latency to Qdrant Cloud or external LLMs, tested up to 1,000 chunks.

| Corpus Size (Chunks) | Ingestion Duration (s) | Ingestion Throughput | P50 Latency (ms) | P90 Latency (ms) | P99 Latency (ms) |
|---|---|---|---|---|---|
| **250** | 3.889 s | **64.3 ch/s** | 15.73 ms | 17.79 ms | 20.53 ms |
| **500** | 8.020 s | **62.3 ch/s** | 16.05 ms | 20.78 ms | 21.30 ms |
| **1000** | 16.887 s | **59.2 ch/s** | 20.81 ms | 25.42 ms | 27.23 ms |

- **Cold-Start Time**: **1.150 seconds** (FastEmbed ONNX session initialization)
- **Local Process Memory Footprint**: **~145 MB** (measured for the active Python benchmark process)
- **Answer Cache Speedup**: **< 2 ms** for cached repeat queries (in-memory)

---

## 5. Deployment Guide (Streamlit Community Cloud)

Follow [docs/SETUP_CHECKLIST.md](file:///docs/SETUP_CHECKLIST.md) for detailed click-by-click instructions:

1. Log in to [share.streamlit.io](https://share.streamlit.io) via GitHub.
2. Click **Deploy an app**.
3. Select Repository: `co2023smniyazi/DocuMind-AI`.
4. Select Branch: `cloud-scalable-rag`.
5. Main file path: `app.py`.
6. Click **Advanced settings > Secrets** and paste your credentials:
```toml
# Primary Free Cloud LLM (Google Gemini)
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "AIzaSy..."
GEMINI_MODEL = "gemini-2.0-flash"

# Embeddings (Zero API calls, in-container ONNX)
EMBED_PROVIDER = "fastembed"

# Persistent Cloud Vector DB (Qdrant Free Tier)
VECTOR_PROVIDER = "qdrant"
QDRANT_URL = "https://your-cluster-id.eu-central.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_COLLECTION = "college_documents"

# Optional Claude (OFF by default to enforce Rs 0 budget)
ENABLE_CLAUDE = "false"
```
7. Click **Deploy!**

---

## 6. Running Tests & Evaluation

### Run Unit and Integration Tests:
```bash
python -m pytest tests/test_loader.py tests/test_chunker.py tests/test_mocked_rag.py tests/test_claude.py tests/test_cloud_rag.py -v
```

### Run Performance Benchmarks:
```bash
python scripts/benchmark.py --sizes 250 500 1000 --queries 30
```

### Run Automated 50-Question Evaluation:
```bash
# Production Run (Requires GEMINI_API_KEY)
python evaluate.py --llm-provider gemini --embed-provider fastembed --sample 50

# Offline Verification Run
python evaluate.py --allow-mock --embed-provider test --llm-provider mock --sample 10
```

---

## 7. Jira Scrum Tracking

- **Project Key:** `RAG`
- **Total Stories:** 25 user stories across 6 Epics
- **Story Points:** 109 SP completed across 4 Sprints
- **Artifacts:**
  - `jira_management/jira_import.csv` (CSV ready for Jira Cloud import)
  - `jira_management/user_stories.md` (Acceptance criteria for RAG-6, RAG-10, RAG-11, RAG-14)
  - `jira_management/status.json` (Machine-readable velocity tracking displayed live in Streamlit UI)
