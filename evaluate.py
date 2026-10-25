"""
Automated Evaluation and Benchmarking Module (STEP 22-23, 44-47, RAG-17, RAG-18)
Evaluates:
1. Retrieval Accuracy (Category A and B)
2. Safe Refusal Rate (Category C Hallucination Test)
3. 5-Metric Answer Quality Scoring (Correctness, Relevance, Groundedness, Clarity, Completeness)
4. Exports Final Evaluation Table to CSV and Markdown.
"""

import os
import json
import argparse
import time
import pandas as pd
from rag_service import RAGService
from llm_service import UNKNOWN_ANSWER_PHRASE


def score_answer_heuristics(
    question: str,
    answer: str,
    expected_doc: str,
    retrieved_docs: list,
    expected_keywords: list,
    is_refusal: bool,
    unanswerable: bool
) -> dict:
    """
    Evaluates answer quality on 1 to 5 scale across 5 metrics specified in Step 23/45.
    """
    if unanswerable:
        if is_refusal:
            return {
                "correctness": 5,
                "relevance": 5,
                "groundedness": 5,
                "clarity": 5,
                "completeness": 5,
                "overall": 5.0
            }
        else:
            # Hallucinated on an unanswerable question
            return {
                "correctness": 1,
                "relevance": 2,
                "groundedness": 1,
                "clarity": 3,
                "completeness": 1,
                "overall": 1.6
            }

    # For answerable questions (Category A & B)
    if is_refusal:
        return {
            "correctness": 2,
            "relevance": 2,
            "groundedness": 4,
            "clarity": 4,
            "completeness": 1,
            "overall": 2.6
        }

    # Keyword coverage
    answer_lower = answer.lower()
    matched_kw = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    kw_ratio = matched_kw / max(len(expected_keywords), 1)

    # Document retrieval accuracy
    doc_retrieved = expected_doc in retrieved_docs if expected_doc else False

    correctness = 5 if (doc_retrieved and kw_ratio >= 0.6) else (4 if doc_retrieved else 2)
    relevance = 5 if (len(answer.split()) >= 10 and kw_ratio >= 0.5) else 4
    groundedness = 5 if doc_retrieved else 3
    clarity = 5 if ("." in answer and len(answer) > 20) else 4
    completeness = 5 if kw_ratio >= 0.7 else (4 if kw_ratio >= 0.4 else 3)

    overall = round((correctness + relevance + groundedness + clarity + completeness) / 5.0, 2)

    return {
        "correctness": correctness,
        "relevance": relevance,
        "groundedness": groundedness,
        "clarity": clarity,
        "completeness": completeness,
        "overall": overall
    }


def run_evaluation(sample_size: int = None, dataset_path: str = "./tests/test_dataset.json", provider: str = None):
    print("=" * 70)
    print("COLLEGE RAG ASSISTANT - COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if sample_size and sample_size < len(dataset):
        # Pick proportional sample from each category
        cat_a = [q for q in dataset if q["category"].startswith("A")][:max(1, sample_size // 2)]
        cat_b = [q for q in dataset if q["category"].startswith("B")][:max(1, sample_size // 4)]
        cat_c = [q for q in dataset if q["category"].startswith("C")][:max(1, sample_size // 4)]
        test_questions = cat_a + cat_b + cat_c
    else:
        test_questions = dataset

    print(f"Loaded {len(test_questions)} questions for evaluation.")
    rag = RAGService(llm_provider=provider)

    # Verify index
    stats = rag.get_stats()
    if stats["total_chunks"] == 0:
        print("Vector database is empty. Indexing ./data now...")
        rag.index_directory("./data")

    records = []
    correct_retrievals = 0
    total_answerable = 0
    correct_refusals = 0
    total_unanswerable = 0

    print("\nRunning evaluation on test cases...")
    for idx, item in enumerate(test_questions, 1):
        q_id = item["id"]
        category = item["category"]
        question = item["question"]
        expected_doc = item["expected_document"]
        expected_keywords = item["expected_keywords"]
        unanswerable = item["unanswerable"]

        print(f"[{idx}/{len(test_questions)}] Q{q_id} ({category}): {question[:45]}...", end="", flush=True)

        start_t = time.time()
        result = rag.answer_question(question)
        latency = round(time.time() - start_t, 2)

        answer = result["answer"]
        is_refusal = result["is_refusal"]
        retrieved_docs = [c.get("source") for c in result["retrieved_chunks"]]

        # Retrieval check
        if not unanswerable:
            total_answerable += 1
            retrieval_correct = expected_doc in retrieved_docs
            if retrieval_correct:
                correct_retrievals += 1
        else:
            total_unanswerable += 1
            retrieval_correct = None  # N/A for unanswerable

        # Refusal check for Category C
        if unanswerable:
            refusal_correct = is_refusal
            if refusal_correct:
                correct_refusals += 1
        else:
            refusal_correct = None

        # Score answer
        scores = score_answer_heuristics(
            question=question,
            answer=answer,
            expected_doc=expected_doc,
            retrieved_docs=retrieved_docs,
            expected_keywords=expected_keywords,
            is_refusal=is_refusal,
            unanswerable=unanswerable
        )

        print(f" -> Latency: {latency}s | Overall Score: {scores['overall']}/5")

        records.append({
            "Question ID": q_id,
            "Category": category,
            "Question": question,
            "Expected Document": expected_doc or "None (Unanswerable)",
            "Retrieved Document(s)": ", ".join(set(retrieved_docs)) if retrieved_docs else "None",
            "Retrieval Correct?": "Yes" if retrieval_correct is True else ("No" if retrieval_correct is False else "N/A"),
            "Is Refusal?": "Yes" if is_refusal else "No",
            "Safe Refusal Correct?": "Yes" if refusal_correct is True else ("No" if refusal_correct is False else "N/A"),
            "Correctness (1-5)": scores["correctness"],
            "Relevance (1-5)": scores["relevance"],
            "Groundedness (1-5)": scores["groundedness"],
            "Clarity (1-5)": scores["clarity"],
            "Completeness (1-5)": scores["completeness"],
            "Overall Score": scores["overall"],
            "Latency (s)": latency,
            "Generated Answer": answer
        })

    df = pd.DataFrame(records)

    # Calculate overall metrics
    retrieval_accuracy = (correct_retrievals / total_answerable * 100) if total_answerable > 0 else 0.0
    safe_refusal_rate = (correct_refusals / total_unanswerable * 100) if total_unanswerable > 0 else 0.0

    avg_correctness = df["Correctness (1-5)"].mean()
    avg_relevance = df["Relevance (1-5)"].mean()
    avg_groundedness = df["Groundedness (1-5)"].mean()
    avg_clarity = df["Clarity (1-5)"].mean()
    avg_completeness = df["Completeness (1-5)"].mean()
    avg_overall = df["Overall Score"].mean()
    avg_latency = df["Latency (s)"].mean()

    # Save to CSV
    csv_path = "./evaluation_results.csv"
    df.to_csv(csv_path, index=False)

    # Generate Markdown Report
    report_md = f"""# Final Evaluation and Benchmark Report

## 1. Summary of Overall Metrics (STEP 22-23, 44-47)

| Metric | Target | Achieved Score | Notes |
|---|---|---|---|
| **Total Test Questions** | 50 | {len(test_questions)} | Across Category A, B, and C |
| **Retrieval Accuracy** | >= 85% | **{retrieval_accuracy:.1f}%** | Correct document retrieved for answerable questions |
| **Safe Refusal Rate** | 100% | **{safe_refusal_rate:.1f}%** | Correctly stated information not found without hallucinating |
| **Correctness (1-5)** | 5.0 | **{avg_correctness:.2f} / 5.0** | Factually aligned with document contents |
| **Relevance (1-5)** | 5.0 | **{avg_relevance:.2f} / 5.0** | Direct response to user prompt |
| **Groundedness (1-5)** | 5.0 | **{avg_groundedness:.2f} / 5.0** | Backed by retrieved chunk context |
| **Clarity (1-5)** | 5.0 | **{avg_clarity:.2f} / 5.0** | Natural language readability |
| **Completeness (1-5)** | 5.0 | **{avg_completeness:.2f} / 5.0** | Contains required procedural/policy details |
| **Mean Overall Quality** | >= 4.0 | **{avg_overall:.2f} / 5.0** | Composite rubric score |
| **Average Response Time**| < 8s | **{avg_latency:.2f} s** | Local inference with Ollama `llama3.2:3b` |

## 2. Category Breakdown

- **Category A (Direct Questions):** Evaluates direct fact retrieval and adherence.
- **Category B (Paraphrased Questions):** Tests semantic understanding of conversational phrasing.
- **Category C (Unanswerable Questions):** Hallucination test ensuring safe refusal.

## 3. Detailed Evaluation Results (First 10 Sample Rows)

| ID | Category | Question | Expected Document | Retrieved Document | Retrieval Correct? | Safe Refusal? | Overall Score |
|---|---|---|---|---|---|---|---|
"""
    for _, row in df.head(10).iterrows():
        report_md += f"| {row['Question ID']} | {row['Category']} | {row['Question'][:35]}... | {row['Expected Document']} | {row['Retrieved Document(s)']} | {row['Retrieval Correct?']} | {row['Safe Refusal Correct?']} | {row['Overall Score']} |\n"

    report_path = "./evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Retrieval Accuracy : {retrieval_accuracy:.1f}% ({correct_retrievals}/{total_answerable})")
    print(f"Safe Refusal Rate  : {safe_refusal_rate:.1f}% ({correct_refusals}/{total_unanswerable})")
    print(f"Mean Overall Score : {avg_overall:.2f} / 5.0")
    print(f"Average Latency    : {avg_latency:.2f} s")
    print(f"Detailed CSV saved : {os.path.abspath(csv_path)}")
    print(f"Markdown Report    : {os.path.abspath(report_path)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate College RAG Assistant")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate a sample subset of N questions")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: claude, ollama, mock, auto")
    args = parser.parse_args()
    run_evaluation(sample_size=args.sample, provider=args.provider)

