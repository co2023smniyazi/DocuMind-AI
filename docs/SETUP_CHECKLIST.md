# DocuMind-AI: Cloud Deployment & Setup Checklist

This document provides a verified, step-by-step setup guide for deploying DocuMind-AI to the cloud with **₹0 budget and zero credit card requirements**.

---

## 1. Prerequisites (All 100% Free)

You will need:
1. A GitHub account (free)
2. A Google account (free)
3. A Qdrant Cloud account (free, no credit card required)

---

## 2. Step 1: Obtain Free Google Gemini API Key

1. Navigate to [Google AI Studio](https://aistudio.google.com/apikey).
2. Sign in with your standard Google account.
3. Click **"Get API key"** -> **"Create API key in new project"**.
4. Copy your key (starts with `AIzaSy...`).
5. Keep it confidential. **Never commit this key to GitHub or paste it in public forums.**

> [!NOTE]
> Google AI Studio free tier gives **15 Requests Per Minute (RPM)**, **1,500 Requests Per Day (RPD)**, and **1,000,000 Tokens Per Minute (TPM)** completely free with zero payment method attached.
> Reference: [Google AI Pricing](https://ai.google.dev/pricing)

---

## 3. Step 2: Create Free Qdrant Cloud Cluster

1. Navigate to [Qdrant Cloud Console](https://cloud.qdrant.io/).
2. Sign up / Sign in with GitHub or email.
3. Click **"Create Cluster"**.
4. Select the **"Free Tier"** (1 GB RAM / 1 GB Storage / Shared CPU).
5. Choose any convenient cloud provider and region (e.g., AWS Frankfurt, GCP Iowa).
6. Click **"Create"** (takes ~45 seconds to initialize).
7. Under cluster details:
   - Copy the **Cluster URL** (e.g. `https://xxxxxx-xxxx-xxxx.eu-central.aws.cloud.qdrant.io:6333`).
   - Click **"API Keys"** -> generate and copy an **API Key**.

> [!NOTE]
> Qdrant Free Tier does not require any payment method. Free clusters enter hibernation after 7 days of inactivity. If suspended, simply click **"Resume"** in the Qdrant console; no stored data is lost.

---

## 4. Step 3: Deploy to Streamlit Community Cloud

1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"Create app"** -> **"Deploy an app"**.
3. Configure the deployment settings:
   - **Repository**: `co2023smniyazi/DocuMind-AI`
   - **Branch**: `cloud-scalable-rag` *(IMPORTANT: Use this branch)*
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain or use the default.
4. Click **"Advanced settings..."** before deploying.
5. In the **Secrets** text box, paste the following configuration (replace placeholders with your real keys):

```toml
# Primary Free Cloud LLM
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "AIzaSyYourRealGeminiApiKeyHere"
GEMINI_MODEL = "gemini-2.0-flash"

# Primary Free Cloud Vector Database
VECTOR_PROVIDER = "qdrant"
QDRANT_URL = "https://your-cluster-id.eu-central.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-real-qdrant-api-key-here"
QDRANT_COLLECTION = "college_documents"

# Embedding Engine (ONNX in-container execution, zero external keys)
EMBED_PROVIDER = "fastembed"

# Optional Cloud LLMs (Leave as is for Rs 0 operation)
ENABLE_CLAUDE = "false"
```

6. Click **"Save"**, then click **"Deploy"**.
7. Streamlit Cloud will pull the repository, install dependencies from `requirements.txt`, initialize the FastEmbed ONNX model, and launch the web app.

---

## 5. Step 4: Verify Deployment via Health Check

1. Once the app loads, navigate to the **"Health Check"** page via the left sidebar (`/Health_Check`).
2. Verify all system checks:
   - [x] **FastEmbed ONNX Embedder**: Status Healthy (384 dimensions, Cosine similarity verified).
   - [x] **Vector Database**: Connected to Qdrant Cloud.
   - [x] **Google Gemini LLM**: Configured and Available.
   - [x] **Secrets Audit**: Confirms keys are detected and properly masked.
3. Click **"Test Live LLM Generation"** to confirm end-to-end cloud inference.

---

## 6. Step 5: Document Indexing & Q&A Verification

1. Return to the main page (**"DocuMind-AI Assistant"**).
2. The default college regulations (`academic_regulations.pdf`, `examination_manual.pdf`, etc.) can be indexed immediately via the sidebar button **"Index Documents"** or uploaded dynamically via the file uploader.
3. Test a Direct Query:
   - *"What is the minimum attendance required to appear for the semester examination?"*
   - Verify answer includes **75%** and cites `academic_regulations.pdf: Page 2`.
4. Test an Unanswerable Query (Hallucination Check):
   - *"What is the swimming pool membership fee for students?"*
   - Verify answer returns: *"Information not found in the available documents."*
