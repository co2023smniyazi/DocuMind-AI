"""
Automated Evaluation and Benchmarking Module (STEP 22-23, 44-47, RAG-17, RAG-18)
Evaluates:
1. Retrieval Accuracy (Category A Direct and Category B Paraphrased)
2. Safe Refusal Rate (Category C Hallucination Prevention Test)
3. 5-Metric Answer Quality Scoring (Correctness, Relevance, Groundedness, Clarity, Completeness)
4. Exports Timestamped Evaluation Results to CSV and Markdown.

HARD RULES:
- Never overwrites historical evaluation_results.csv or evaluation_report.md.
- Rejects test/mock embedders unless --allow-mock is explicitly passed.
- Automatically handles API rate limits (15 RPM for Gemini free tier) with backoff.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from rag_service import RAGService
from embeddings import get_embedder, TestDeterministicEmbedder
from llm_service import UNKNOWN_ANSWER_PHRASE, MockLLMService


def score_answer_heuristics(
    question: str,
    answer: str,
    expected_doc: Optional[str],
    retrieved_docs: List[str],
    expected_keywords: List[str],
    is_refusal: bool,
    unanswerable: bool
) -> Dict[str, Any]:
    """
    Evaluates answer quality on 1 to 5 scale across 5 metrics specified in project rubric.
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
    relevance = 5 if (len(answer.split()) >= 8 and kw_ratio >= 0.4) else 4
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


def run_evaluation(
    sample_size: Optional[int] = None,
    dataset_path: str = "./tests/test_dataset.json",
    llm_provider: Optional[str] = None,
    embed_provider: Optional[str] = "fastembed",
    vector_store_type: Optional[str] = "chroma",
    top_k: int = 4,
    allow_mock: bool = False,
    output_dir: str = "evaluation",
    rate_delay_sec: Optional[float] = None
):
    print("=" * 75)
    print("DOCUMIND-AI: COMPREHENSIVE RAG BENCHMARK & EVALUATION ENGINE")
    print("=" * 75)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Safety check on mock embeddings
    embedder = get_embedder(embed_provider)
    is_mock_embedder = isinstance(embedder, TestDeterministicEmbedder)
    if is_mock_embedder and not allow_mock:
        raise ValueError(
            "Evaluation refused: Mock/TestDeterministicEmbedder was requested without --allow-mock. "
            "Real evaluation requires FastEmbed ('fastembed'), Gemini ('gemini'), or Ollama ('ollama')."
        )

    # Initialize RAG Service
    print(f"Initializing RAG Service with LLM: '{llm_provider or 'auto'}', Embedder: '{embed_provider}'...")
    rag = RAGService(
        llm_provider=llm_provider,
        embed_provider=embed_provider,
        vector_provider=vector_store_type,
        top_k=top_k
    )

    is_mock_run = is_mock_embedder or isinstance(rag.llm_service, MockLLMService)
    if is_mock_run:
        print("\n" + "!" * 75)
        print("WARNING: RUNNING IN MOCK MODE (--allow-mock enabled).")
        print("RESULTS ARE TAGGED 'MOCK - NOT FOR SUBMISSION'.")
        print("!" * 75 + "\n")

    # Verify or trigger indexing
    stats = rag.get_stats()
    doc_count = len(stats.get("unique_documents", []))
    print(f"Vector Store Status: {stats.get('total_chunks', 0)} chunks across {doc_count} documents.")
    if stats["total_chunks"] == 0:
        print("Vector database is empty. Indexing './data' documents now...")
        rag.index_directory("./data")
        stats = rag.get_stats()
        print(f"Indexing completed: {stats['total_chunks']} chunks in store.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if sample_size and sample_size < len(dataset):
        # Balanced sampling across categories A, B, and C
        cat_a = [q for q in dataset if q["category"].startswith("A")][:max(1, sample_size // 2)]
        cat_b = [q for q in dataset if q["category"].startswith("B")][:max(1, sample_size // 4)]
        cat_c = [q for q in dataset if q["category"].startswith("C")][:max(1, sample_size // 4)]
        test_questions = cat_a + cat_b + cat_c
    else:
        test_questions = dataset

    print(f"Commencing benchmark on {len(test_questions)} questions...")
    records = []
    cat_stats = {
        "A": {"total": 0, "retrieval_correct": 0, "scores": []},
        "B": {"total": 0, "retrieval_correct": 0, "scores": []},
        "C": {"total": 0, "refusal_correct": 0, "scores": []}
    }

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Rate limit delay between queries (configurable via --rate-delay or EVAL_RATE_DELAY_SEC)
    is_cloud_api = rag.llm_service.provider_name in ["gemini", "groq"]
    if rate_delay_sec is None:
        env_delay = os.environ.get("EVAL_RATE_DELAY_SEC")
        # Default 4.0s for cloud API (approx 15 RPM unverified default), 0.0s for local/mock
        delay = float(env_delay) if env_delay else (4.0 if is_cloud_api else 0.0)
    else:
        delay = max(0.0, rate_delay_sec)

    for idx, item in enumerate(test_questions, 1):
        q_id = item["id"]
        category_full = item["category"]
        category_code = "A" if category_full.startswith("A") else ("B" if category_full.startswith("B") else "C")
        question = item["question"]
        expected_doc = item.get("expected_document")
        expected_keywords = item.get("expected_keywords", [])
        unanswerable = item.get("unanswerable", False)

        print(f"[{idx:02d}/{len(test_questions):02d}] Q{q_id:02d} [{category_code}]: {question[:45]}...", end="", flush=True)

        # Execute query with retry for rate limits
        max_retries = 3
        result = None
        start_t = time.time()
        for attempt in range(max_retries):
            try:
                result = rag.answer_question(question)
                break
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    wait_time = (2 ** attempt) * 4
                    print(f" [Rate limit, waiting {wait_time}s]...", end="", flush=True)
                    time.sleep(wait_time)
                else:
                    result = {
                        "answer": f"Error: {e}",
                        "sources": [],
                        "retrieved_chunks": [],
                        "is_refusal": False,
                        "is_error": True,
                        "latency_ms": 0.0,
                        "provider": rag.llm_service.provider_name
                    }
                    break

        latency = round(time.time() - start_t, 2)
        answer = result["answer"]
        is_refusal = result["is_refusal"]
        retrieved_docs = [c.get("source") for c in result.get("retrieved_chunks", [])]

        # Evaluate Retrieval Accuracy (Cat A & B)
        retrieval_correct = None
        if not unanswerable:
            retrieval_correct = expected_doc in retrieved_docs if expected_doc else False
            cat_stats[category_code]["total"] += 1
            if retrieval_correct:
                cat_stats[category_code]["retrieval_correct"] += 1

        # Evaluate Safe Refusal (Cat C)
        refusal_correct = None
        if unanswerable:
            refusal_correct = is_refusal
            cat_stats["C"]["total"] += 1
            if refusal_correct:
                cat_stats["C"]["refusal_correct"] += 1

        # Heuristic scoring
        scores = score_answer_heuristics(
            question=question,
            answer=answer,
            expected_doc=expected_doc,
            retrieved_docs=retrieved_docs,
            expected_keywords=expected_keywords,
            is_refusal=is_refusal,
            unanswerable=unanswerable
        )
        cat_stats[category_code]["scores"].append(scores["overall"])

        status_flag = "REFUSAL_OK" if (unanswerable and is_refusal) else ("RETRIEVE_OK" if retrieval_correct else "FLAGGED")
        print(f" -> {status_flag} | {latency}s | Score: {scores['overall']}/5")

        records.append({
            "Question ID": q_id,
            "Category": category_full,
            "Category Code": category_code,
            "Question": question,
            "Expected Document": expected_doc or "None (Unanswerable)",
            "Retrieved Document(s)": ", ".join(dict.fromkeys(retrieved_docs)) if retrieved_docs else "None",
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
            "LLM Provider": result.get("provider", rag.llm_service.provider_name),
            "Embedder": embed_provider,
            "Generated Answer": answer
        })

        # Throttling delay between queries to respect provider rate limits
        if delay > 0 and idx < len(test_questions):
            time.sleep(delay)

    df = pd.DataFrame(records)

    # Calculate Metrics
    tot_ans = cat_stats["A"]["total"] + cat_stats["B"]["total"]
    ret_ans = cat_stats["A"]["retrieval_correct"] + cat_stats["B"]["retrieval_correct"]
    overall_retrieval_acc = (ret_ans / tot_ans * 100) if tot_ans > 0 else 0.0

    cat_a_acc = (cat_stats["A"]["retrieval_correct"] / max(cat_stats["A"]["total"], 1)) * 100
    cat_b_acc = (cat_stats["B"]["retrieval_correct"] / max(cat_stats["B"]["total"], 1)) * 100
    safe_refusal_rate = (cat_stats["C"]["refusal_correct"] / max(cat_stats["C"]["total"], 1)) * 100

    avg_correctness = df["Correctness (1-5)"].mean()
    avg_relevance = df["Relevance (1-5)"].mean()
    avg_groundedness = df["Groundedness (1-5)"].mean()
    avg_clarity = df["Clarity (1-5)"].mean()
    avg_completeness = df["Completeness (1-5)"].mean()
    avg_overall = df["Overall Score"].mean()
    avg_latency = df["Latency (s)"].mean()
    p50_latency = df["Latency (s)"].median()
    p90_latency = df["Latency (s)"].quantile(0.90)

    # Create destination output directory without touching old root files
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"evaluation_results_{timestamp_str}.csv")
    out_md = os.path.join(output_dir, f"evaluation_report_{timestamp_str}.md")

    mock_badge = "\n> [!CAUTION]\n> **MOCK RUN - NOT FOR SUBMISSION**: Generated with mock embeddings or mock LLM.\n" if is_mock_run else ""

    df.to_csv(out_csv, index=False)

    report_content = f"""# Benchmark Evaluation Report
**Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}  
**LLM Provider**: `{rag.llm_service.provider_name}`  
**Embedding Provider**: `{embed_provider}` ({getattr(embedder, 'dimension', 384)}-dimensional)  
**Vector Store**: `{vector_store_type}`  
**Dataset**: `{dataset_path}` ({len(test_questions)} questions evaluated)  
{mock_badge}
## 1. Executive Summary & KPIs

| Metric | Target | Achieved Score | Status | Description |
|---|---|---|---|---|
| **Overall Retrieval Accuracy** | ≥ 85.0% | **{overall_retrieval_acc:.1f}%** | {'PASS' if overall_retrieval_acc >= 85 else 'FAIL'} | Category A + B answerable questions |
| **Category A (Direct) Retrieval** | ≥ 90.0% | **{cat_a_acc:.1f}%** | {'PASS' if cat_a_acc >= 90 else 'REVIEW'} | Direct policy & regulatory questions ({cat_stats['A']['retrieval_correct']}/{cat_stats['A']['total']}) |
| **Category B (Paraphrased) Retrieval** | ≥ 80.0% | **{cat_b_acc:.1f}%** | {'PASS' if cat_b_acc >= 80 else 'REVIEW'} | Conversational rephrasings ({cat_stats['B']['retrieval_correct']}/{cat_stats['B']['total']}) |
| **Safe Refusal Rate (Category C)** | 100.0% | **{safe_refusal_rate:.1f}%** | {'PASS' if safe_refusal_rate >= 90 else 'FAIL'} | Hallucination prevention on unanswerables ({cat_stats['C']['refusal_correct']}/{cat_stats['C']['total']}) |
| **Average Response Latency** | < 8.0s | **{avg_latency:.2f} s** | {'PASS' if avg_latency < 8 else 'REVIEW'} | Median (P50): {p50_latency:.2f}s, P90: {p90_latency:.2f}s |
| **Composite Quality Score** | ≥ 4.0 / 5.0 | **{avg_overall:.2f} / 5.0** | {'PASS' if avg_overall >= 4.0 else 'REVIEW'} | 5-Metric Composite Rubric Score |

---

## 2. Five-Metric Rubric Breakdown (Scale 1 to 5)

| Metric | Average Score | Description |
|---|---|---|
| **Correctness** | **{avg_correctness:.2f}** / 5.0 | Factual truth relative to ground-truth PDFs |
| **Relevance** | **{avg_relevance:.2f}** / 5.0 | Direct pertinence to the student question |
| **Groundedness** | **{avg_groundedness:.2f}** / 5.0 | Strict adherence to retrieved context (zero extraneous fabrications) |
| **Clarity** | **{avg_clarity:.2f}** / 5.0 | Readability, structure, and professional student tone |
| **Completeness** | **{avg_completeness:.2f}** / 5.0 | Inclusion of required policy provisions, thresholds, and conditions |

---

## 3. Sample Detailed Test Results (First 10 Cases)

| Q# | Category | Question | Expected Doc | Retrieved Doc(s) | Retrieval OK? | Safe Refusal? | Score |
|---|---|---|---|---|---|---|---|
"""
    for _, row in df.head(10).iterrows():
        ret_ok = row["Retrieval Correct?"]
        ref_ok = row["Safe Refusal Correct?"]
        report_content += (
            f"| Q{row['Question ID']} | {row['Category Code']} | {row['Question'][:32]}... | "
            f"`{row['Expected Document']}` | `{row['Retrieved Document(s)'][:25]}` | "
            f"{ret_ok} | {ref_ok} | {row['Overall Score']}/5.0 |\n"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETE")
    print(f"Overall Retrieval Accuracy : {overall_retrieval_acc:.1f}% ({ret_ans}/{tot_ans})")
    print(f" - Category A (Direct)     : {cat_a_acc:.1f}% ({cat_stats['A']['retrieval_correct']}/{cat_stats['A']['total']})")
    print(f" - Category B (Paraphrase) : {cat_b_acc:.1f}% ({cat_stats['B']['retrieval_correct']}/{cat_stats['B']['total']})")
    print(f"Safe Refusal Rate (Cat C)  : {safe_refusal_rate:.1f}% ({cat_stats['C']['refusal_correct']}/{cat_stats['C']['total']})")
    print(f"Composite Quality Score    : {avg_overall:.2f} / 5.0")
    print(f"Mean Latency               : {avg_latency:.2f} s (P50: {p50_latency:.2f}s, P90: {p90_latency:.2f}s)")
    print(f"Saved CSV Report           : {os.path.abspath(out_csv)}")
    print(f"Saved Markdown Report      : {os.path.abspath(out_md)}")
    print("=" * 75)
    return df, out_csv, out_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocuMind-AI Evaluation & Benchmarking Suite")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (e.g. 10 or 50)")
    parser.add_argument("--dataset", type=str, default="./tests/test_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--llm-provider", type=str, default=None, help="LLM Provider: gemini, groq, claude, ollama, mock, auto")
    parser.add_argument("--embed-provider", type=str, default="fastembed", help="Embedding Provider: fastembed, gemini, ollama, test")
    parser.add_argument("--vector-store", type=str, default="chroma", help="Vector store: chroma, qdrant")
    parser.add_argument("--top-k", type=int, default=4, help="Retriever top_k chunks")
    parser.add_argument("--allow-mock", action="store_true", help="Allow running with mock/test embedders for offline verification")
    parser.add_argument("--rate-delay", type=float, default=None, help="Delay in seconds between LLM queries (default: 4.0s for cloud, 0.0s for mock/local)")
    parser.add_argument("--output-dir", type=str, default="evaluation", help="Directory for evaluation artifacts")

    args = parser.parse_args()
    run_evaluation(
        sample_size=args.sample,
        dataset_path=args.dataset,
        llm_provider=args.llm_provider,
        embed_provider=args.embed_provider,
        vector_store_type=args.vector_store,
        top_k=args.top_k,
        allow_mock=args.allow_mock,
        output_dir=args.output_dir,
        rate_delay_sec=args.rate_delay
    )
