# DocuMind-AI: Verified Cloud Free Tier Limits & Inactivity Policies

This document details the verified free tier limits, rate quotas, suspension policies, and privacy terms for every cloud service and library integrated into **DocuMind-AI**.

**Audit Date:** 2026-10-25  
**Verification Method:** Official documentation retrieved via Web Search on 2026-10-25  
**Budget Policy:** Strict ₹0.00 (Zero INR / Zero USD). No credit cards, subscriptions, or paid API credits required.

---

## 1. Hosting: Streamlit Community Cloud

* **Service:** Streamlit Community Cloud (Hosted App Engine)
* **Official Documentation URLs:**
  * Resource Limits: [https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resource-limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resource-limits)
  * App Hibernation: [https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-hibernation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-hibernation)
* **Date Checked:** 2026-10-25 (via Web Search of official documentation)
* **Cost:** ₹0.00 (Free for public open-source GitHub repositories)
* **Hardware & Compute Limits:**
  * **Memory (RAM):** Dynamically allocated between **690 MB minimum** and **2.7 GB maximum** (baseline around ~1.0 GB). If an application exceeds this allocation, it crashes with *"This app has gone over its resource limits."*
  * **CPU:** Dynamically allocated between **0.078 vCPU cores minimum** and **2.0 cores maximum**. Heavy inference or long loops cause throttling.
  * **Storage:** Ephemeral disk up to **50 GB maximum**. Storage resets on container reboot, crash, or redeployment.
* **Hibernation / Sleep Policy:**
  * Any app that receives **no traffic or user interaction for 12 hours** automatically enters **sleep mode (hibernation)**.
  * Visitors see a *"This app has gone to sleep due to inactivity"* banner with a wake-up button. Clicking the button restores the container in 15–30 seconds.
* **Architectural Safeguard:** All PDF processing is in-memory; chunk text and vector payloads are stored in Qdrant Cloud so data survives container sleep/reboot cycles.

---

## 2. Primary Cloud LLM: Google Gemini API (Google AI Studio)

* **Service:** Google AI Studio Free Tier
* **Official Documentation URLs:**
  * Pricing & Limits: [https://ai.google.dev/pricing](https://ai.google.dev/pricing)
  * Rate Limits: [https://ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)
  * Terms of Service: [https://ai.google.dev/terms](https://ai.google.dev/terms)
* **Date Checked:** 2026-10-25 (via Web Search of official documentation)
* **Cost:** ₹0.00 (Free of charge; no credit card required)
* **Verified Free Tier Quotas (`gemini-2.0-flash` & `gemini-1.5-flash`):**
  * **RPM (Requests Per Minute):** 15 RPM
  * **TPM (Tokens Per Minute):** 1,000,000 TPM
  * **RPD (Requests Per Day):** 1,500 RPD
* **Model Existence & Selection:**
  * `gemini-2.0-flash`: Active and supported on the free tier.
  * `gemini-1.5-flash`: Active fallback model on the free tier.
* **Data Privacy Notice:**
  * Under the free tier, Google terms state prompt and response data may be reviewed by human annotators to train/improve Google products. Sensitive institutional data must be anonymized.
* **Architectural Safeguard:** `FallbackLLMChain` catches HTTP 429 (`ResourceExhausted`), applies exponential backoff, and trips a circuit breaker to Groq Cloud.

---

## 3. Secondary Cloud LLM: Groq Cloud

* **Service:** Groq Cloud Developer Free Tier
* **Official Documentation URLs:**
  * Rate Limits: [https://console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits)
  * Pricing: [https://groq.com/pricing/](https://groq.com/pricing/)
* **Date Checked:** 2026-10-25 (via Web Search of official documentation)
* **Cost:** ₹0.00 (Free developer access; no payment method required)
* **Verified Free Tier Quotas:**
  * **`llama-3.3-70b-versatile`:** 30 RPM, 12,000 TPM, 1,000 RPD, 100,000 TPD.
  * **`llama-3.1-8b-instant`:** 30 RPM, 6,000 TPM, 14,400 RPD, 500,000 TPD.
* **Quota Scope:** Limits apply organization-wide, and whichever threshold (RPM, RPD, TPM, TPD) is reached first triggers rate limiting.
* **Architectural Safeguard:** Configured as hot-standby fallback in `FallbackLLMChain`.

---

## 4. Persistent Cloud Vector Database: Qdrant Cloud

* **Service:** Qdrant Cloud Managed Vector Database
* **Official Documentation URLs:**
  * Pricing: [https://qdrant.tech/pricing/](https://qdrant.tech/pricing/)
  * Cloud FAQ: [https://qdrant.tech/documentation/cloud/cloud-faq/](https://qdrant.tech/documentation/cloud/cloud-faq/)
* **Date Checked:** 2026-10-25 (via Web Search of official documentation)
* **Cost:** ₹0.00 (Permanent free tier; zero credit card required)
* **Verified Free Tier Cluster Specifications:**
  * **Architecture:** 1 single-node cluster
  * **vCPU:** 0.5 shared vCPU
  * **RAM:** 1.0 GB cluster RAM
  * **Disk Storage:** 4.0 GB disk (capable of hosting up to ~1,000,000 768-dim vectors or ~2,000,000 384-dim vectors)
* **Inactivity & Suspension / Deletion Policy:**
  * **Suspension:** Free tier clusters with **no indexing or search activity for 1 week (7 days)** are automatically **suspended**.
  * **Reactivation:** Suspended clusters retain data and can be reactivated with 1 click from the Qdrant Cloud Console.
  * **Permanent Deletion:** If a suspended cluster is **not reactivated within 4 weeks (28 days)** of suspension, it is **permanently deleted**.
* **Architectural Safeguard:** The application detects suspended cluster connection errors and displays an informative alert in the UI guiding the user to reactivate via console.

---

## 5. In-Container Semantic Embeddings: FastEmbed ONNX

* **Library:** `fastembed` (Qdrant ONNX-based embedding engine)
* **Official Documentation URL:** [https://qdrant.github.io/fastembed/](https://qdrant.github.io/fastembed/)
* **Date Checked:** 2026-10-25
* **Cost:** ₹0.00 (No API key, runs in-process)
* **Specifications:**
  * **Model:** `BAAI/bge-small-en-v1.5`
  * **Dimensions:** 384 dimensions
  * **On-Disk Model Size:** ~67 MB
  * **In-Memory Process Footprint:** ~145 MB (measured locally in Python 3.11 process)
* **Architectural Safeguard:** Eliminates external embedding API dependency and prevents rate limiting during document ingestion.

---

## 6. Continuous Integration & Automation: GitHub Actions

* **Service:** GitHub Actions
* **Official Documentation URL:** [https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions)
* **Date Checked:** 2026-10-25 (via Web Search of official documentation)
* **Cost:** ₹0.00
* **Verified Quotas:**
  * **Public Repositories:** **100% Free and Unlimited** runner minutes on standard GitHub-hosted Linux (Ubuntu) runners.
  * **Private Repositories:** **2,000 free minutes per month** on standard GitHub Free accounts.
* **Architectural Safeguard:** Full offline unit test suite executes in ~12 seconds, consuming negligible runner quota.

---

## 7. Optional Multi-Cloud Claude Adapters (Disabled by Default)

* **Providers:** Anthropic API, Amazon Bedrock, Google Vertex AI
* **Status:** Implemented, mock-tested with unit tests, **NOT live-tested**.
* **Cost:** Paid (No permanent free tier exists for Claude APIs).
* **Configuration:** `ENABLE_CLAUDE=false` in all committed configurations. Enabling requires user's own paid credentials.

---

## Summary Verification Table

| Service | Free Tier Allocation | Suspension / Sleep Rule | Official Documentation URL | Checked Date | Method |
|---|---|---|---|---|---|
| **Streamlit Cloud** | ~1.0 GB RAM (690MB–2.7GB), 0.078–2 vCPU | Sleeps after **12 hours** of inactivity | [Streamlit Limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resource-limits) | 2026-10-25 | Web Search |
| **Google Gemini API** | 15 RPM, 1M TPM, 1,500 RPD (`gemini-2.0-flash`) | Stays active with free key; prompts logged | [Gemini Pricing](https://ai.google.dev/pricing) | 2026-10-25 | Web Search |
| **Groq Cloud** | 30 RPM, 12k TPM, 1,000 RPD (`llama-3.3-70b`) | Organization-level caps | [Groq Limits](https://console.groq.com/docs/rate-limits) | 2026-10-25 | Web Search |
| **Qdrant Cloud** | 1 GB RAM, 4 GB Disk, 0.5 vCPU | Suspends after **1 week**; deleted after **4 weeks** | [Qdrant Pricing](https://qdrant.tech/pricing/) | 2026-10-25 | Web Search |
| **GitHub Actions** | Unlimited (public repos) / 2k min/mo (private) | Per calendar month quota | [GitHub Billing](https://docs.github.com/en/billing) | 2026-10-25 | Web Search |
| **FastEmbed ONNX** | In-container process (~145 MB RAM) | N/A (In-process execution) | [FastEmbed Docs](https://qdrant.github.io/fastembed/) | 2026-10-25 | Code Inspection |
| **Claude Adapters** | None (Paid API) | Kept OFF (`ENABLE_CLAUDE=false`) | [Anthropic Pricing](https://www.anthropic.com/pricing) | 2026-10-25 | Code Inspection |
