# DocuMind AI — Phase 0 Codebase Inspection Report

**Project:** DocuMind AI — Cloud-Based College Document Question Answering Assistant Using RAG  
**Branch:** `cloud-scalable-rag`  
**Audit Date:** October 25, 2026  
**Auditor:** Senior Cloud / RAG Systems Engineer  
**Budget Constraint:** ₹0 (Zero INR / Zero USD). Strictly zero paid billing or active credit card requirements.

---

## 1. Executive Summary

A comprehensive architectural and functional inspection was conducted across all Python modules, tests, CI configurations, Jira management assets, and historical benchmark artifacts in the repository.

### Dataset Integrity Check:
- **`tests/test_dataset.json` Question Count:** **Exactly 50 questions**
  - **Category A (Direct Questions):** 20 questions (IDs 1–20)
  - **Category B (Paraphrased Questions):** 20 questions (IDs 21–40)
  - **Category C (Unanswerable / Anti-Hallucination):** 10 questions (IDs 41–50)
- **Historical Benchmark Validity:** **INVALID**. Historical results in `evaluation_report.md` only evaluated 4 questions, produced identical refusal strings (*"Information not found in the available documents."*) across all questions with an artificial 0.05s latency. Documented in `docs/OLD_RESULTS_NOT_VALID.md`.

### Real Test Execution Output:
Running `python -m pytest -v`:
- **Collected:** 34 tests
- **Passed:** 21 tests
- **Skipped:** 6 tests (`test_live_rag.py` and `test_rag.py` skipped gracefully due to absent local Ollama daemon)
- **Failed:** 7 tests (5 Claude mock tests expecting `is_enabled` keyword argument in `ClaudeLLMService`, 1 Qdrant in-memory stats test expecting unique documents, 1 in-memory upload test expecting `**kwargs` support in `QdrantCloudVectorStore`).

---

## 2. Component-by-Component Architectural Audit

| Module / Asset | Role | Current Status | Issues & Gaps Identified |
|---|---|---|---|
| **`app.py`** | Web Interface | **Partial** | Misleading claims ("100% Free Forever", "Pay-as-you-go") present in parts. Hardcoded Jira status metrics rather than reading dynamic state. LLM availability checks must be optimized to avoid unnecessary API handshakes. |
| **`embeddings.py`** | Vector Embeddings | **Partial** | `FastEmbedEmbedder` (384-dim ONNX `BAAI/bge-small-en-v1.5`) successfully integrated. Old `CloudSemanticEmbedder` (hashed bag-of-words) needs deprecation to a test-only fixture (`TestOnlyHashEmbedder`) inaccessible in production. |
| **`vector_store.py`** | Vector Storage | **Partial** | `QdrantCloudVectorStore` implemented with `qdrant-client` 1.20+ (`query_points`). Needs `**kwargs` support in constructor, unique document scrolling in `get_stats()`, and scalar quantization / payload indexing for 1 GB free cluster scalability. |
| **`document_loader.py`** | PDF Ingestion | **Met** | `load_pdf_from_bytes()` implemented for in-memory upload handling. Handles encrypted, scanned, and corrupted PDFs without disk writes. |
| **`chunker.py`** | Text Chunking | **Met** | Recursive chunking with 1500 character window and 200 character overlap. Strict boundary parameter guards in place. |
| **`retriever.py`** | Retrieval Engine | **Partial** | Dense vector search implemented. Needs similarity score threshold tuning, MMR / near-duplicate deduplication, and context token budgeting. |
| **`llm_service.py`** | LLM Synthesis | **Partial** | `GeminiLLMService` (default free) and `GroqLLMService` (secondary free) implemented. `ClaudeLLMService` needs constructor parameter alignment (`is_enabled`). Bedrock, Vertex AI, and MCP server adapters are missing. |
| **`rag_service.py`** | RAG Pipeline | **Met** | In-memory `index_pdf_bytes()` implemented. Safe refusal phrase (`"Information not found in the available documents."`) strictly enforced. |
| **`evaluate.py`** | Benchmark Runner | **Partial** | Script structure exists, but needs enforcement to reject mock embedders unless `--allow-mock` is passed, timestamped artifact generation without overwriting historical files, and rate-limit backoff. |
| **`jira_management/`** | Scrum Tracking | **Partial** | `jira_backlog.csv` contains 22 items. Missing `status.json` (for live app consumption), `jira_import.csv` with full assignment specs, and `user_stories.md`. |
| **`.github/workflows/`** | CI / Automation | **Partial** | `ci.yml` exists. Missing secret scanning action and `workflow_dispatch` evaluation / benchmark workflows. |

---

## 3. Requirement Coverage Matrix

### 3.1. Assignment Functional Requirements (College RAG)
- [x] PyPDF document loading with page metadata (RAG-6)
- [x] Text chunking with overlap (1500 / 200) (RAG-7)
- [x] Semantic vector embeddings (FastEmbed 384-dim ONNX) (RAG-8)
- [x] Persistent cloud vector database (Qdrant Cloud Free Tier 1GB) (RAG-9)
- [x] Semantic retrieval of relevant chunks (RAG-10)
- [x] Anti-hallucination prompt & LLM integration (RAG-11, 12)
- [x] Source citations with document title and page number (RAG-13, 16)
- [x] Exact unknown-answer refusal handling (`"Information not found in the available documents."`) (RAG-14)
- [x] Streamlit multi-tab user interface (RAG-15)

### 3.2. Cloud-Native & Free-Tier Constraints
- [x] 100% Cloud-Only runtime (zero localhost or Ollama dependencies in cloud mode)
- [x] ₹0 Budget strictly enforced (zero credit card or paid trial dependencies)
- [x] Claude integration strictly optional and OFF by default (`ENABLE_CLAUDE=false`)
- [x] No raw PDFs saved to container disk (processed in-memory, stored in Qdrant payloads)
- [x] Bundled sample PDFs in `data/` preserved as read-only seed data
- [x] Masked secret management (Streamlit Secrets / GitHub Actions Secrets)

### 3.3. Advanced Scalable RAG Requirements (To Be Implemented)
- [ ] Multi-Cloud Claude Adapters (Amazon Bedrock, Google Vertex AI, Anthropic API, MCP Server) — Mocked & Documented only
- [ ] Stateless Streamlit caching (`st.cache_resource`) and connection pooling
- [ ] Idempotent, resumable ingestion with content hashing and batched upserts
- [ ] Dense retrieval with payload indexes, MMR near-duplicate removal, and token budgeting
- [ ] Structured observability (latency logging, cache hit/miss counters, provider fallback counters)
- [ ] Standalone benchmark suite (`scripts/benchmark.py`) for p50/p95 latency and throughput
- [ ] Full Jira artifacts (`jira_import.csv`, `JIRA_SETUP_GUIDE.md`, `user_stories.md`, `status.json`)

---

## 4. Phase-by-Phase Remediation Roadmap

1. **Phase 1: Core Foundation & Bug Fixes**
   - Align `ClaudeLLMService` parameters (`is_enabled`, `model_name`).
   - Fix `QdrantCloudVectorStore` `**kwargs` and `get_stats()` unique document tracking.
   - Achieve **100% pass rate** on all unit and integration tests. Commit.
2. **Phase 2: Scalable Cloud Architecture & Multi-Cloud Claude Adapters**
   - Implement Bedrock, Vertex AI, Anthropic Claude adapters (all OFF by default, mocked).
   - Implement Model Context Protocol (MCP) server for document search and Q&A.
   - Implement idempotent ingestion with SHA256 content hashes, MMR retrieval, and token budgeting.
3. **Phase 3: Observability, Diagnostics & Benchmarks**
   - Implement structured latency tracking and metrics in `rag_service.py`.
   - Update `pages/Health_Check.py` with full subsystem status, cache stats, and latency metrics.
   - Build `scripts/benchmark.py` for p50/p95 throughput measurements and `docs/BENCHMARKS.md`.
4. **Phase 4: Evaluation Workflow & Jira Scrum Assets**
   - Upgrade `evaluate.py` with strict mock rejection, timestamped outputs, and rate-limit handling.
   - Create `.github/workflows/evaluate.yml` and `.github/workflows/benchmark.yml` (`workflow_dispatch`).
   - Generate `jira_management/status.json`, `jira_management/jira_import.csv`, and `jira_management/user_stories.md`.
5. **Phase 5: Documentation & Final Verification**
   - Deliver `docs/REPORT_DRAFT.md`, `docs/DEMO_SCRIPT.md`, `docs/SCALABILITY.md`, `docs/SETUP_CHECKLIST.md`.
   - Update `README.md` and verify zero secrets in the repository.
   - Produce final honest report.
