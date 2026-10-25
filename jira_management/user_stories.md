# DocuMind AI — Detailed Scrum User Stories

This document specifies the official Agile user stories, acceptance criteria, and technical validation steps for the core RAG pipeline stories (RAG-6, RAG-10, RAG-11, and RAG-14) as required by the final-year project assignment.

---

## 1. User Story: RAG-6 (PDF Document Loading & Text Extraction)

- **Issue Key:** `RAG-6`
- **Issue Type:** Story
- **Epic:** `EPIC-3: RAG Pipeline & Claude AI`
- **Sprint:** `Sprint 2: RAG Pipeline`
- **Story Points:** 3 SP
- **Priority:** High

### User Story Statement:
> **As a** college assistant system developer,  
> **I want to** extract text and metadata from official college PDF documents,  
> **So that** policy contents can be chunked, embedded, and indexed for semantic search without manual transcription.

### Acceptance Criteria:
1. **Format Compatibility:** Accepts valid PDF files up to 15 MB and 50 pages without crashing.
2. **Page-by-Page Extraction:** Retains source document filename and discrete 1-based page numbers for every extracted page.
3. **In-Memory Streaming:** Supports loading from raw bytes (`load_pdf_from_bytes`) so user uploads never touch container disk in cloud environments.
4. **Resilience & Error Handling:**
   - Empty or corrupted PDFs must produce a descriptive error dictionary (`error: ...`) rather than an unhandled runtime exception.
   - Scanned image PDFs without a digital font layer must be flagged as `is_scanned: True` with an instructional warning.
   - Password-encrypted PDFs must be flagged with `is_encrypted: True`.

---

## 2. User Story: RAG-10 (Semantic Vector Retrieval)

- **Issue Key:** `RAG-10`
- **Issue Type:** Story
- **Epic:** `EPIC-3: RAG Pipeline & Claude AI`
- **Sprint:** `Sprint 2: RAG Pipeline`
- **Story Points:** 5 SP
- **Priority:** High

### User Story Statement:
> **As a** college student or faculty member,  
> **I want to** submit a natural language question and retrieve the most relevant sections of college documents,  
> **So that** the LLM is supplied with grounded evidence to answer my query.

### Acceptance Criteria:
1. **Semantic Vector Search:** Converts the input question into a 384-dimensional dense vector using FastEmbed ONNX (`BAAI/bge-small-en-v1.5`) and queries the vector database.
2. **Top-K Retrieval:** Returns the top-$k$ (default 4) most semantically similar chunks ranked by cosine similarity.
3. **Context Diversity (Deduplication):** Implements near-duplicate chunk suppression to prevent redundant consecutive chunks from crowding the prompt.
4. **Metadata Preservation:** Every returned chunk must contain `source`, `page`, `chunk_index`, and `similarity` score.
5. **Threshold Guard:** Supports a configurable minimum similarity threshold (`min_similarity`) to discard irrelevant noise.

---

## 3. User Story: RAG-11 (LLM Context Integration & Answer Synthesis)

- **Issue Key:** `RAG-11`
- **Issue Type:** Story
- **Epic:** `EPIC-3: RAG Pipeline & Claude AI`
- **Sprint:** `Sprint 3: Cloud & Claude`
- **Story Points:** 5 SP
- **Priority:** Highest

### User Story Statement:
> **As a** student,  
> **I want** the assistant to synthesize an understandable, grammatically clear answer from the retrieved document excerpts,  
> **So that** I can quickly understand complex academic regulations without reading entire PDFs.

### Acceptance Criteria:
1. **Context Groundedness:** Prompts the configured LLM (Google Gemini 2.0 Flash / Groq Llama-3.3-70B) with the retrieved text blocks clearly formatted with source and page markers.
2. **Token Budget Enforcement:** Context injection is capped at 12,000 characters (~3,000 tokens) to prevent exceeding free-tier tokens-per-minute (TPM) limits.
3. **Zero Fabrications:** System instructions strictly forbid the LLM from inventing attendance percentages, fees, grading scales, or placement eligibility.
4. **Fallback & Retry:** Includes an exponential backoff retry loop and circuit breaker to switch seamlessly between Gemini and Groq on HTTP 429 rate limits.

---

## 4. User Story: RAG-14 (Safe Refusal / Unknown-Answer Handling)

- **Issue Key:** `RAG-14`
- **Issue Type:** Story
- **Epic:** `EPIC-3: RAG Pipeline & Claude AI`
- **Sprint:** `Sprint 3: Cloud & Claude`
- **Story Points:** 3 SP
- **Priority:** Highest

### User Story Statement:
> **As a** college administrator,  
> **I want** the assistant to refuse to answer when the requested information is absent from official college documents,  
> **So that** students are never misled by hallucinated policies, non-existent fees, or false deadlines.

### Acceptance Criteria:
1. **Exact Refusal String:** When information is absent or retrieval yields no matching chunks, the system must strictly output:
   > `"Information not found in the available documents."`
2. **Unanswerable Benchmark Verification:** Must achieve a **100% Safe Refusal Rate** on the 10 unanswerable test questions in Category C (e.g., hostel fees, swimming pool charges, cafeteria timings).
3. **Empty Source Suppression:** When a refusal occurs, no misleading document source citations must be rendered in the UI.
