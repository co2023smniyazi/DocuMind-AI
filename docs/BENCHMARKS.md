# DocuMind-AI: Performance & Scalability Benchmarks

> [!IMPORTANT]
> **LOCAL TEST ENVIRONMENT DISCLOSURE**:  
> The empirical figures presented below were measured **LOCALLY on a Windows 11 PC (Python 3.11.9, x86_64)** using local ChromaDB storage and the local `fastembed` ONNX Runtime (`BAAI/bge-small-en-v1.5`, 384 dimensions) on **2026-10-25**.  
> **These measurements DO NOT include network latency to Qdrant Cloud or external cloud LLM APIs, and ARE NOT cloud or Streamlit Community Cloud measurements.**  
> Scalability is experimentally verified up to the tested limit of **1,000 chunks**; performance beyond 1,000 chunks is unverified.

---

## 1. Local PC Benchmark Measurements (2026-10-25)

**Environment**: Local Windows 11 Workstation, Python 3.11.9  
**Vector Store**: Local ChromaDB (temporary directory on local SSD)  
**Embedder**: In-process ONNX Runtime (`BAAI/bge-small-en-v1.5`, 384-dim)  
**Corpus**: 250, 500, and 1,000 synthetic chunks derived from college regulation templates  
**Script**: `python scripts/benchmark.py --sizes 250 500 1000 --queries 30 --vector-store chroma`

| Corpus Size (Chunks) | Ingestion Duration (s) | Ingestion Throughput (chunks/s) | Mean Retrieval (ms) | P50 Latency (ms) | P90 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) |
|---|---|---|---|---|---|---|---|
| **250** | 3.889 s | **64.3 ch/s** | 15.42 ms | **15.73 ms** | 17.79 ms | 19.19 ms | 20.53 ms |
| **500** | 8.020 s | **62.3 ch/s** | 16.43 ms | **16.05 ms** | 20.78 ms | 20.80 ms | 21.30 ms |
| **1000** | 16.887 s | **59.2 ch/s** | 21.06 ms | **20.81 ms** | 25.42 ms | 25.88 ms | 27.23 ms |

### Cold-Start & Process Footprint (Local Measurements):
* **FastEmbed ONNX Cold-Start Latency:** **1.150 seconds** (time to load ONNX graph and initialize session).
* **Local Python Process Memory (RSS):** **~145 MB** (measured for the active Python 3.11 benchmark process on Windows).  
  *(Note: Streamlit Community Cloud provides dynamic RAM typically between 690 MB and 2.7 GB per container. While 145 MB is well below the baseline 1 GB allocation, actual container memory in production will also include Streamlit server overhead and web UI session state).*
* **Model Weight Footprint:** **~67 MB** on disk (`BAAI/bge-small-en-v1.5` in `./fastembed_cache`).

---

## 2. Cloud CI / Qdrant Cloud Benchmarks

* **Status:** `[AWAITING CI RUN WITH CLOUD SECRETS]`  
* **Configuration:** Can be executed on-demand via GitHub Actions using [`.github/workflows/benchmark.yml`](file:///d:/New%20folder/cc%20project%203/.github/workflows/benchmark.yml) by setting `vector_store: qdrant`.  
* **Runner Environment:** GitHub-hosted `ubuntu-latest` runner querying a remote Qdrant Cloud cluster over HTTPS.

| Corpus Size (Chunks) | Target Cloud Cluster | Ingestion Duration (s) | Throughput (ch/s) | P50 Latency (ms) | P90 Latency (ms) |
|---|---|---|---|---|---|
| **250** | Qdrant Cloud Free Tier | *Not Run* | *Not Run* | *Not Run* | *Not Run* |
| **500** | Qdrant Cloud Free Tier | *Not Run* | *Not Run* | *Not Run* | *Not Run* |
| **1000** | Qdrant Cloud Free Tier | *Not Run* | *Not Run* | *Not Run* | *Not Run* |

*(Cloud metrics will reflect real network roundtrip latency to Qdrant Cloud clusters across geographic regions, expected to add approximately 30–80 ms per query depending on AWS/GCP region).*
