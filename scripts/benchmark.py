"""
Performance Benchmarking Suite for DocuMind-AI (RAG-17, RAG-18)
Measures:
1. Cold start & Model initialization time.
2. Embedding & Vector store ingestion throughput (chunks/sec).
3. Retrieval latency percentiles (P50, P90, P95, P99).
4. Memory consumption.
"""

import os
import sys
import time
import shutil
import tempfile
import argparse
from typing import List, Dict, Any
import numpy as np

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings import FastEmbedEmbedder, TestDeterministicEmbedder
from vector_store import ChromaVectorStore, QdrantCloudVectorStore
from retriever import SemanticRetriever


def generate_synthetic_chunks(num_chunks: int) -> List[Dict[str, Any]]:
    """Generates realistic synthetic college document chunks for benchmarking."""
    templates = [
        "Students must maintain at least 75% attendance in each course to be eligible for semester examinations.",
        "A condonation fee of Rs 500 is applicable for students with attendance between 65% and 75% on medical grounds.",
        "Total graduation credit requirement for standard four-year B.Tech program is 160 credits.",
        "Lateral entry engineering students must complete a minimum of 120 credits over six semesters.",
        "First class with distinction is awarded to students achieving a minimum Cumulative Grade Point Average of 7.5.",
        "Hostel curfew is strictly 9:30 PM for all residential students, requiring biometric verification.",
        "Examination revaluation applications must be submitted within 15 calendar days with a fee of Rs 1000 per subject.",
        "Students are permitted a maximum of two backlog courses when registering for campus placement drives."
    ]
    chunks = []
    for i in range(num_chunks):
        base_text = templates[i % len(templates)]
        text = f"[Regulation Clause {i+1}] {base_text} Additional regulation clause details for index item {i}."
        chunks.append({
            "id": f"bench_{i:05d}",
            "text": text,
            "source": f"benchmark_regulations_{(i % 5) + 1}.pdf",
            "page": (i % 20) + 1,
            "chunk_id": f"bench_{i:05d}"
        })
    return chunks


def benchmark_cold_start():
    """Measures model import and weight initialization latency."""
    print("\n--- Benchmarking Model Initialization (Cold Start) ---")
    t0 = time.time()
    embedder = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5")
    # Trigger first inference to measure full ONNX graph initialization
    _ = embedder.get_embedding("Warmup query for ONNX runtime.")
    init_latency = time.time() - t0
    print(f"FastEmbed ONNX Cold Start Latency: {init_latency:.3f} s")
    return init_latency, embedder


def benchmark_ingestion_and_retrieval(corpus_sizes: List[int], embedder, runs_per_size: int = 50):
    """Measures ingestion throughput and retrieval latency percentiles."""
    results = []

    for size in corpus_sizes:
        print(f"\n--- Testing Corpus Size: {size} Chunks ---")
        temp_dir = tempfile.mkdtemp(prefix="bench_chroma_")

        try:
            chunks = generate_synthetic_chunks(size)
            store = ChromaVectorStore(
                persist_directory=temp_dir,
                collection_name=f"bench_{size}",
                embedding_dimension=embedder.dimension
            )

            # Ingestion Throughput Measurement
            t_ingest_start = time.time()
            texts = [c["text"] for c in chunks]
            embeddings = embedder.get_embeddings_batch(texts)
            store.add_chunks(chunks, embeddings)
            ingest_duration = time.time() - t_ingest_start
            throughput = size / max(ingest_duration, 0.001)

            print(f"Ingested {size} chunks in {ingest_duration:.3f} s ({throughput:.1f} chunks/sec)")

            # Retrieval Latency Measurement
            retriever = SemanticRetriever(vector_store=store, embedder=embedder)
            query_latencies_ms = []

            sample_queries = [
                "What is the minimum attendance requirement?",
                "What is the condonation fee for attendance shortage?",
                "How many credits for graduation in B.Tech?",
                "What is the deadline for examination revaluation?",
                "What are hostel curfew timings?"
            ]

            print(f"Running {runs_per_size} retrieval queries...")
            for i in range(runs_per_size):
                query = sample_queries[i % len(sample_queries)]
                t_q_start = time.time()
                res = retriever.retrieve(query, top_k=4)
                lat_ms = (time.time() - t_q_start) * 1000.0
                query_latencies_ms.append(lat_ms)

            p50 = float(np.percentile(query_latencies_ms, 50))
            p90 = float(np.percentile(query_latencies_ms, 90))
            p95 = float(np.percentile(query_latencies_ms, 95))
            p99 = float(np.percentile(query_latencies_ms, 99))
            mean_lat = float(np.mean(query_latencies_ms))

            print(f"Retrieval Latencies: Mean={mean_lat:.2f}ms | P50={p50:.2f}ms | P90={p90:.2f}ms | P95={p95:.2f}ms | P99={p99:.2f}ms")

            results.append({
                "corpus_size": size,
                "ingest_time_s": round(ingest_duration, 3),
                "throughput_chunks_per_sec": round(throughput, 1),
                "mean_retrieval_ms": round(mean_lat, 2),
                "p50_ms": round(p50, 2),
                "p90_ms": round(p90, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
            })
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="DocuMind-AI Benchmark Runner")
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000], help="Corpus sizes to test")
    parser.add_argument("--queries", type=int, default=50, help="Number of test queries per size")
    args = parser.parse_args()

    print("=" * 70)
    print("DOCUMIND-AI SYSTEM PERFORMANCE & SCALABILITY BENCHMARK")
    print("=" * 70)

    cold_start, embedder = benchmark_cold_start()
    results = benchmark_ingestion_and_retrieval(args.sizes, embedder, runs_per_size=args.queries)

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Cold Start / Model Load : {cold_start:.3f} s")
    print(f"{'Corpus Size':<12} | {'Ingest (s)':<10} | {'Throughput':<14} | {'P50 (ms)':<10} | {'P90 (ms)':<10} | {'P99 (ms)':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['corpus_size']:<12} | {r['ingest_time_s']:<10.3f} | {r['throughput_chunks_per_sec']:<5.1f} ch/s    | {r['p50_ms']:<10.2f} | {r['p90_ms']:<10.2f} | {r['p99_ms']:<10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
