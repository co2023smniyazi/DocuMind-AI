# Jira Board and Sprint Execution Tracker

## 1. Live Scrum Board Status Example (Sprint 3 in Progress)

```
+--------------------+--------------------+--------------------+--------------------+--------------------+
|       TO DO        |    IN PROGRESS     |    CODE REVIEW     |      TESTING       |        DONE        |
+--------------------+--------------------+--------------------+--------------------+--------------------+
| RAG-14             | RAG-12             | RAG-11             | RAG-10             | RAG-6              |
| Unknown-answer     | RAG Prompt         | LLM Integration    | Semantic Retrieval | PDF Extraction     |
| handling           | Engineering        |                    |                    |                    |
|                    |                    |                    |                    | RAG-7              |
| RAG-16             | RAG-15             | RAG-13             | RAG-9              | Text Chunking      |
| Source display     | Streamlit UI       | Source Citation    | Vector Store       |                    |
|                    |                    |                    |                    | RAG-8              |
|                    |                    |                    |                    | Embeddings Gen     |
|                    |                    |                    |                    |                    |
|                    |                    |                    |                    | RAG-1 to RAG-5     |
|                    |                    |                    |                    | Sprint 1 Complete  |
+--------------------+--------------------+--------------------+--------------------+--------------------+
```

---

## 2. Sprint Burndown & Velocity Projection

| Sprint | Committed Story Points | Completed Story Points | Velocity Achieved | Status |
|---|---|---|---|---|
| **Sprint 1 (Foundation)** | 16 SP | 16 SP | 100% | Completed |
| **Sprint 2 (RAG Pipeline)** | 21 SP | 21 SP | 100% | Completed |
| **Sprint 3 (LLM App & UI)** | 22 SP | 22 SP | 100% | Completed |
| **Sprint 4 (Testing & Demo)** | 20 SP | 15 SP | 75% | In Progress (RAG-20 final demo) |

---

## 3. Bug and Defect Tracking in Jira (RAG-19)

| Bug ID | Title | Severity | Root Cause | Resolution | Status |
|---|---|---|---|---|---|
| **BUG-1** | Telemetry Exception on ChromaDB Client Init | Low | Chroma library default telemetry mismatch in local python 3.11 environment. | Set `ANONYMIZED_TELEMETRY=False` in Settings and environment variable. | RESOLVED |
| **BUG-2** | Paraphrased Questions Over-refusing on Related Topics | Medium | Strict zero-shot refusal prompt caused model to refuse attendance percentage inquiries when phrased as "how many classes can I miss?". | Updated prompt instructions to synthesize policy rules for related inquiries while maintaining safe refusal for unknown topics. | RESOLVED |
| **BUG-3** | Edge Case Boundary Loss in Chunking | Medium | Strict chunk boundary cuts dropped contextual overlap on single-sentence paragraphs. | Adjusted chunk overlap to 200 chars and set `top_k=4` in retriever. | RESOLVED |

