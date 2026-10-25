"""
DocuMind AI - System Health Diagnostics Page (Cloud-Ready & Free-Tier Audited)
Automated diagnostics for:
1. In-memory PDF Ingestion
2. Semantic Embeddings (FastEmbed ONNX / Gemini)
3. Vector Database (Qdrant Cloud / ChromaDB)
4. Cloud LLM Answer Generation (Gemini / Groq / Claude / Mock)
"""

import io
import time
import streamlit as st
import pandas as pd
from pypdf import PdfWriter

from document_loader import load_pdf_from_bytes
from embeddings import get_embedder, FastEmbedEmbedder, BaseEmbedder
from vector_store import get_vector_store, BaseVectorStore
from llm_service import get_llm_service, get_secret, BaseLLMService

st.set_page_config(
    page_title="DocuMind AI - Health Diagnostics",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .health-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F766E;
        margin-bottom: 0.2rem;
    }
    .health-sub {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .check-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warn {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="health-header">🏥 Cloud System Diagnostics & Health Check</div>', unsafe_allow_html=True)
st.markdown('<div class="health-sub">Live automated verification across PDF ingestion, FastEmbed embeddings, Qdrant/Chroma vector DB, and Cloud LLM.</div>', unsafe_allow_html=True)

col_run1, col_run2 = st.columns([1, 4])
with col_run1:
    run_all = st.button("🔄 Run Full Diagnostics", type="primary", use_container_width=True)
with col_run2:
    test_llm_live = st.checkbox("Include live Cloud LLM API call (uses 1 request from free quota)", value=False)

st.divider()

def mask_secret(value: str) -> str:
    if not value:
        return "Not Set"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"

# Diagnostic 1: PDF Ingestion
def test_pdf_ingestion():
    t0 = time.time()
    try:
        # Create minimal synthetic valid PDF with 1 blank page
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        stream = io.BytesIO()
        writer.write(stream)
        raw_bytes = stream.getvalue()

        pages, summary = load_pdf_from_bytes(raw_bytes, "health_probe.pdf")
        elapsed = round((time.time() - t0) * 1000, 1)

        if summary.get("error"):
            return {"status": "FAIL", "name": "PDF Ingestion", "latency_ms": elapsed, "detail": summary["error"]}
        
        return {
            "status": "PASS",
            "name": "PDF Ingestion",
            "latency_ms": elapsed,
            "detail": f"Successfully parsed in-memory PDF ({len(raw_bytes)} bytes, {summary['total_pages']} page)"
        }
    except Exception as e:
        elapsed = round((time.time() - t0) * 1000, 1)
        return {"status": "FAIL", "name": "PDF Ingestion", "latency_ms": elapsed, "detail": str(e)}

# Diagnostic 2: Semantic Embedder
def test_embeddings():
    t0 = time.time()
    try:
        embedder: BaseEmbedder = get_embedder()
        sample_text = "DocuMind AI college document retrieval verification sentence."
        vec = embedder.get_embedding(sample_text)
        elapsed = round((time.time() - t0) * 1000, 1)

        dim = len(vec)
        is_nonzero = any(abs(v) > 1e-6 for v in vec)
        
        if dim > 0 and is_nonzero:
            return {
                "status": "PASS",
                "name": "Vector Embeddings",
                "latency_ms": elapsed,
                "detail": f"{embedder.__class__.__name__} ({dim} dims, vector non-zero)"
            }
        else:
            return {
                "status": "FAIL",
                "name": "Vector Embeddings",
                "latency_ms": elapsed,
                "detail": f"Generated empty or all-zero vector (dim: {dim})"
            }
    except Exception as e:
        elapsed = round((time.time() - t0) * 1000, 1)
        return {"status": "FAIL", "name": "Vector Embeddings", "latency_ms": elapsed, "detail": str(e)}

# Diagnostic 3: Vector Store Probe (Upsert -> Query -> Delete)
def test_vector_store():
    t0 = time.time()
    probe_source = "__documind_health_probe__"
    try:
        embedder: BaseEmbedder = get_embedder()
        store: BaseVectorStore = get_vector_store(embedding_dimension=getattr(embedder, "dimension", 384))
        
        # Test vector
        test_vec = embedder.get_embedding("College attendance rule probe test.")
        chunk = {
            "id": f"{probe_source}_chk1",
            "text": "Health probe chunk text for college attendance verification.",
            "source": probe_source,
            "page": 1,
            "chunk_index": 1
        }

        # 1. Add chunk
        store.add_chunks([chunk], [test_vec])

        # 2. Search chunk
        results = store.similarity_search(test_vec, top_k=2)
        found = any(r.get("source") == probe_source for r in results)

        # 3. Clean up probe chunk
        store.delete_document(probe_source)
        
        elapsed = round((time.time() - t0) * 1000, 1)
        stats = store.get_stats()

        if found:
            return {
                "status": "PASS",
                "name": "Vector Database (Upsert/Query/Delete)",
                "latency_ms": elapsed,
                "detail": f"Provider: {stats.get('provider')} | Probe lifecycle verified"
            }
        else:
            return {
                "status": "WARN",
                "name": "Vector Database",
                "latency_ms": elapsed,
                "detail": f"Probe inserted but not retrieved in top results. Provider: {stats.get('provider')}"
            }
    except Exception as e:
        elapsed = round((time.time() - t0) * 1000, 1)
        # Attempt cleanup
        try:
            store.delete_document(probe_source)
        except Exception:
            pass
        return {"status": "FAIL", "name": "Vector Database", "latency_ms": elapsed, "detail": str(e)}

# Diagnostic 4: LLM Service
def test_llm(perform_call: bool):
    t0 = time.time()
    try:
        llm: BaseLLMService = get_llm_service()
        model_name = getattr(llm, "model_name", "Mock")
        provider_name = llm.__class__.__name__

        if not llm.is_available():
            elapsed = round((time.time() - t0) * 1000, 1)
            return {
                "status": "FAIL",
                "name": "Cloud LLM Service",
                "latency_ms": elapsed,
                "detail": f"{provider_name} ({model_name}) is offline or missing API key."
            }

        if not perform_call:
            elapsed = round((time.time() - t0) * 1000, 1)
            return {
                "status": "PASS",
                "name": "Cloud LLM Service",
                "latency_ms": elapsed,
                "detail": f"{provider_name} ({model_name}) configured and ready. Check checkbox above for live probe."
            }

        # Live probe call
        prompt = "Answer with the single word 'OPERATIONAL' if you are working."
        resp = llm.generate_response(prompt)
        elapsed = round((time.time() - t0) * 1000, 1)

        return {
            "status": "PASS",
            "name": "Cloud LLM Service (Live Call)",
            "latency_ms": elapsed,
            "detail": f"Model: {model_name} | Response: '{resp.strip()[:60]}'"
        }
    except Exception as e:
        elapsed = round((time.time() - t0) * 1000, 1)
        return {"status": "FAIL", "name": "Cloud LLM Service", "latency_ms": elapsed, "detail": str(e)}

# Execute checks
results = []
with st.spinner("Running diagnostics..."):
    results.append(test_pdf_ingestion())
    results.append(test_embeddings())
    results.append(test_vector_store())
    results.append(test_llm(test_llm_live))

# Summary Metrics
col1, col2, col3 = st.columns(3)
pass_count = sum(1 for r in results if r["status"] == "PASS")
fail_count = sum(1 for r in results if r["status"] == "FAIL")
warn_count = sum(1 for r in results if r["status"] == "WARN")

col1.metric("Tests Passed", f"{pass_count} / {len(results)}")
col2.metric("Failures", f"{fail_count}")
col3.metric("Warnings", f"{warn_count}")

# Detailed Results Cards
st.markdown("### 📋 Diagnostic Test Results")
for r in results:
    badge_class = "badge-pass" if r["status"] == "PASS" else ("badge-warn" if r["status"] == "WARN" else "badge-fail")
    icon = "✅" if r["status"] == "PASS" else ("⚠️" if r["status"] == "WARN" else "❌")
    
    st.markdown(f"""
    <div class="check-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; font-size: 1.1rem;">{icon} {r['name']}</span>
            <span class="{badge_class}">{r['status']} ({r['latency_ms']} ms)</span>
        </div>
        <p style="margin-top: 0.5rem; color: #4B5563; font-size: 0.95rem;">{r['detail']}</p>
    </div>
    """, unsafe_allow_html=True)

# Environment Secrets Audit (Masked)
st.markdown("### 🔒 Environment Secrets Audit (Masked)")
secrets_audit = [
    {"Key": "GEMINI_API_KEY", "Status": "Configured" if get_secret("GEMINI_API_KEY") else "Missing", "Masked Value": mask_secret(get_secret("GEMINI_API_KEY"))},
    {"Key": "GROQ_API_KEY", "Status": "Configured" if get_secret("GROQ_API_KEY") else "Missing", "Masked Value": mask_secret(get_secret("GROQ_API_KEY"))},
    {"Key": "QDRANT_URL", "Status": "Configured" if get_secret("QDRANT_URL") else "Missing (Using Chroma/Memory)", "Masked Value": mask_secret(get_secret("QDRANT_URL"))},
    {"Key": "QDRANT_API_KEY", "Status": "Configured" if get_secret("QDRANT_API_KEY") else "Missing", "Masked Value": mask_secret(get_secret("QDRANT_API_KEY"))},
    {"Key": "ENABLE_CLAUDE", "Status": "Set" if get_secret("ENABLE_CLAUDE") else "Default (false)", "Masked Value": get_secret("ENABLE_CLAUDE", "false")},
    {"Key": "ANTHROPIC_API_KEY", "Status": "Configured" if get_secret("ANTHROPIC_API_KEY") else "Not Set", "Masked Value": mask_secret(get_secret("ANTHROPIC_API_KEY"))},
]
st.dataframe(pd.DataFrame(secrets_audit), use_container_width=True)

st.info("💡 To configure or modify secrets on Streamlit Community Cloud: Go to **App Dashboard > Settings > Secrets**.")

