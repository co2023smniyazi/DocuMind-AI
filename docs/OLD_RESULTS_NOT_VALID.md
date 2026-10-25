# Notice: Invalidation of Historical Evaluation Results

**Audit Date:** October 25, 2026  
**File Reference:** `evaluation_report.md` & `evaluation_results.csv`  
**Status:** **INVALID / MOCK RUN (NOT FOR SUBMISSION)**

---

## 1. Findings from Historical Evaluation Inspection

An inspection of the pre-existing `evaluation_report.md` and `evaluation_results.csv` revealed that the recorded metrics do **not** represent a genuine, completed benchmark run against the 50 college document questions:

1. **Truncated Evaluation Set (4 of 50 Questions):**
   - The test suite defines 50 questions in `tests/test_dataset.json` (20 Category A, 20 Category B, 10 Category C).
   - The historical evaluation only executed 4 questions (Q1, Q2, Q21, Q41).
2. **Universal Safe Refusal / Zero Real Generation:**
   - In `evaluation_results.csv`, every single question—including direct factual questions (Q1: *"What is the minimum attendance required..."*, Q2: *"What is the condonation fee..."*)—produced the exact refusal string:
     > `"Information not found in the available documents."`
   - Real answers were never synthesized because the retrieval failed or was mocked, causing the anti-hallucination guard to trigger unconditionally.
3. **Unrealistic Latency (0.04s - 0.08s):**
   - Average recorded latency was **0.05 seconds** (50 ms).
   - Real inference using local Ollama (`llama3.2:3b`) or Cloud LLMs (`gemini-2.0-flash` / `claude-3-5-haiku`) requires at least 800ms to 3.5s per response. A 50ms latency proves that no genuine LLM text generation was performed during that run.

---

## 2. Integrity Preservation Policy

Per project hard rules:
- `evaluation_report.md` and `evaluation_results.csv` have been retained in their original state and will **not** be deleted or overwritten.
- A new, fully transparent automated evaluation workflow (`.github/workflows/evaluate.yml`) will execute against all 50 questions using genuine cloud inference, saving timestamped result artifacts independently.
