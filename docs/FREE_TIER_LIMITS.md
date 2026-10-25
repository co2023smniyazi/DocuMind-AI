# DocuMind AI — Verified Cloud Free Tier Limits & Policies

This document specifies the exact, verified free tier limits, rate caps, suspension policies, and privacy terms for every external cloud platform and library integrated into **DocuMind AI**.

**Audit Date:** October 2026  
**Budget Enforcement:** ₹0 (Zero INR / Zero USD). No credit cards or paid subscriptions are required.

---

## 1. Hosting: Streamlit Community Cloud

* **Service:** Streamlit Community Cloud (App Deployment)
* **Official Pricing & Limits Documentation:** 
  * [Streamlit App Resource Limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resource-limits)
  * [Streamlit Cloud Sleeping Apps FAQ](https://docs.streamlit.io/deploy/streamlit-community-cloud/status#wake-up-sleeping-apps)
* **Cost:** ₹0 ($0.00 Free forever for public open-source repositories).
* **Hardware & Resource Limits:**
  * **Memory (RAM):** Up to **1.0 GB RAM** allocated per container. If container memory exceeds this limit, the process is restarted by the orchestrator (OOM).
  * **CPU:** Shared vCPU.
  * **App Count:** 1 active public app on standard free Community accounts (up to 3 for legacy/verified accounts).
* **Lifecycle & Suspension Behavior:**
  * If an application receives no traffic for **7 consecutive days**, Streamlit Community Cloud automatically puts the container into **sleep mode**.
  * Any user visiting the public URL will see a *"This app is sleeping"* banner with a **"Yes, get this app back up!"** button. Clicking the button wakes the container in approximately **10 to 30 seconds**.
* **Storage Limitations:**
  * Ephemeral local filesystem. Any files written to the local disk during runtime (e.g., in `./data`) are destroyed on container reboot or redeploy.
  * *DocuMind AI Architecture Fix:* Uploaded PDFs are processed in-memory, chunked, embedded, and stored directly in vector database payloads.

---

## 2. Default Cloud LLM: Google Gemini API (Google AI Studio)

* **Service:** Google AI Studio Gemini API Free Tier
* **Official Pricing & Rate Limits Documentation:**
  * [Google AI Studio Pricing](https://ai.google.dev/pricing)
  * [Gemini API Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)
* **Default Model:** `gemini-2.0-flash` (configurable to `gemini-1.5-flash` or `gemini-1.5-pro`)
* **Cost:** ₹0 ($0.00 Free of charge; no credit card required to create an API key).
* **Verified Free Tier Quotas (`gemini-2.0-flash` / `gemini-1.5-flash`):**
  * **RPM (Requests Per Minute):** 15 RPM
  * **TPM (Tokens Per Minute):** 1,000,000 TPM
  * **RPD (Requests Per Day):** 1,500 RPD
* **Data Privacy Policy (Important):**
  * According to the official [Google AI Studio Terms of Service](https://ai.google.dev/terms):
    > *"When using the free of charge version of Google AI Studio, Google may use your prompts, responses, and related data to train and improve Google products and services. Human reviewers may read, annotate, and process your data."*
  * Therefore, do not pass highly sensitive personal identification numbers (e.g., Aadhaar, SSN, bank credentials) in documents while on the free tier.
* **Error & Retry Handling:**
  * HTTP status `429 (ResourceExhausted)` occurs if RPM > 15. DocuMind AI implements an exponential backoff retry loop (up to 3 attempts) and friendly UI warnings.

---

## 3. Secondary Cloud LLM: Groq Cloud

* **Service:** Groq Cloud Developer API Free Tier
* **Official Pricing & Rate Limits Documentation:**
  * [Groq Cloud Rate Limits](https://console.groq.com/docs/rate-limits)
  * [Groq Pricing Overview](https://groq.com/pricing/)
* **Default Model:** `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`
* **Cost:** ₹0 ($0.00 Free tier for developers).
* **Verified Free Tier Quotas (`llama-3.3-70b-versatile`):**
  * **RPM (Requests Per Minute):** 30 RPM
  * **TPM (Tokens Per Minute):** 6,000 to 30,000 TPM
  * **RPD (Requests Per Day):** 14,400 RPD
* **Behavior:**
  * Extremely low time-to-first-token (LPUs). Free tier accounts have per-minute burst rate tracking.

---

## 4. Local In-Container Embedding: FastEmbed ONNX

* **Library:** `fastembed` (Qdrant ONNX-based lightweight runtime)
* **Official Documentation:**
  * [FastEmbed GitHub & Benchmarks](https://qdrant.github.io/fastembed/)
* **Model:** `BAAI/bge-small-en-v1.5`
* **Cost:** ₹0 ($0.00 No API key, no external network calls after initial weight download).
* **Specifications:**
  * **Embedding Dimension:** 384 dimensions
  * **Disk Weight Size:** ~67 MB (`model_optimized.onnx`)
  * **Container RAM Overhead:** ~150 MB (fits well inside Streamlit Cloud's 1,000 MB RAM budget)
  * **Inference Speed:** ~15-30ms per 500-token chunk on standard shared vCPU.
* **Deterministic Fallback:**
  * A test-only `TestDeterministicEmbedder` (384 dimensions) is provided in `embeddings.py` for CI pipelines and offline unit testing.

---

## 5. Cloud Vector Database: Qdrant Cloud

* **Service:** Qdrant Cloud Managed Vector Database
* **Official Pricing & Limits Documentation:**
  * [Qdrant Cloud Pricing](https://qdrant.tech/pricing/)
  * [Qdrant Cloud FAQ](https://qdrant.tech/documentation/cloud/cloud-faq/)
* **Tier:** Free Tier (Managed Cluster)
* **Cost:** ₹0 ($0.00 Permanent free tier, no credit card required).
* **Specifications & Limits:**
  * **Memory:** 1 GB cluster RAM
  * **Capacity:** Up to ~1,000,000 vectors with 384 dimensions (FastEmbed)
  * **CPU:** 0.5 vCPU shared
* **Suspension Behavior (Crucial):**
  * Qdrant free tier clusters with **no query or indexing activity for 7 consecutive days** are automatically moved to **Suspended** status.
  * Suspended clusters do NOT delete data; they can be reactivated with **one click** from the [Qdrant Cloud Console](https://cloud.qdrant.io/).
  * DocuMind AI catches connection errors to suspended clusters and advises the user to click "Reactivate" in the Qdrant console, falling back seamlessly to ChromaDB or In-Memory storage.

---

## 6. Optional LLM: Anthropic Claude (Disabled by Default)

* **Service:** Anthropic Claude Messages API
* **Official Pricing Documentation:**
  * [Anthropic API Pricing](https://docs.anthropic.com/en/docs/initial-setup#pricing)
* **Default Setting:** `ENABLE_CLAUDE=false` (Hard Rule: Budget is ₹0).
* **Policy:**
  * Anthropic does NOT offer a continuous zero-credit free tier for API calls. API calls require prepaid credits.
  * Claude integration is strictly optional. It will only initialize if **both** conditions are met:
    1. `ENABLE_CLAUDE=true` is set in Streamlit Secrets or environment variables.
    2. A valid `ANTHROPIC_API_KEY` is provided.

---

## Summary Matrix

| Service | Role | Cost | Quota / Resource Limit | Suspension Rule |
|---|---|---|---|---|
| **Streamlit Cloud** | Web UI Hosting | ₹0 | 1 GB RAM / app | Sleeps after 7 days inactivity |
| **Google Gemini API** | Default Cloud LLM | ₹0 | 15 RPM, 1M TPM, 1,500 RPD | Active with free key; data logged |
| **FastEmbed ONNX** | Semantic Embeddings | ₹0 | In-memory (~150 MB RAM, 384 dim) | Never suspends; local execution |
| **Qdrant Cloud** | Vector DB | ₹0 | 1 GB Cluster (~1M vectors) | Suspends after 7 days inactivity |
| **Groq Cloud** | Fallback Cloud LLM | ₹0 | 30 RPM, 6k-30k TPM | Active with free key |
| **ChromaDB** | Local/Test DB | ₹0 | Local disk / in-memory | Local only |
| **Claude API** | Optional LLM | Paid | Pay-per-token | Disabled by default (`ENABLE_CLAUDE=false`) |

