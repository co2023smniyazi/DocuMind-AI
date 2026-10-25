# Final Evaluation and Benchmark Report

## 1. Summary of Overall Metrics (STEP 22-23, 44-47)

| Metric | Target | Achieved Score | Notes |
|---|---|---|---|
| **Total Test Questions** | 50 | 4 | Across Category A, B, and C |
| **Retrieval Accuracy** | >= 85% | **100.0%** | Correct document retrieved for answerable questions |
| **Safe Refusal Rate** | 100% | **100.0%** | Correctly stated information not found without hallucinating |
| **Correctness (1-5)** | 5.0 | **2.75 / 5.0** | Factually aligned with document contents |
| **Relevance (1-5)** | 5.0 | **2.75 / 5.0** | Direct response to user prompt |
| **Groundedness (1-5)** | 5.0 | **4.25 / 5.0** | Backed by retrieved chunk context |
| **Clarity (1-5)** | 5.0 | **4.25 / 5.0** | Natural language readability |
| **Completeness (1-5)** | 5.0 | **2.00 / 5.0** | Contains required procedural/policy details |
| **Mean Overall Quality** | >= 4.0 | **3.20 / 5.0** | Composite rubric score |
| **Average Response Time**| < 8s | **0.05 s** | Local inference with Ollama `llama3.2:3b` |

## 2. Category Breakdown

- **Category A (Direct Questions):** Evaluates direct fact retrieval and adherence.
- **Category B (Paraphrased Questions):** Tests semantic understanding of conversational phrasing.
- **Category C (Unanswerable Questions):** Hallucination test ensuring safe refusal.

## 3. Detailed Evaluation Results (First 10 Sample Rows)

| ID | Category | Question | Expected Document | Retrieved Document | Retrieval Correct? | Safe Refusal? | Overall Score |
|---|---|---|---|---|---|---|---|
| 1 | A - Direct | What is the minimum attendance requ... | academic_regulations.pdf | academic_regulations.pdf, examination_guidelines.pdf, department_rules.pdf | Yes | N/A | 2.6 |
| 2 | A - Direct | What is the condonation fee for att... | academic_regulations.pdf | academic_regulations.pdf, examination_guidelines.pdf | Yes | N/A | 2.6 |
| 21 | B - Paraphrased | How many classes can I miss before ... | academic_regulations.pdf | student_handbook.pdf, examination_guidelines.pdf, academic_regulations.pdf | Yes | N/A | 2.6 |
| 41 | C - Unanswerable | What is the hostel accommodation an... | None (Unanswerable) | examination_guidelines.pdf, academic_regulations.pdf, student_handbook.pdf | N/A | Yes | 5.0 |
