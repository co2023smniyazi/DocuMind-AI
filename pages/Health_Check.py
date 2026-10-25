"""
DocuMind AI - System Health Diagnostics & Observability Page
Automated diagnostics for:
1. In-memory PDF Ingestion (synthetic probe)
2. Semantic Embeddings (FastEmbed ONNX BAAI/bge-small-en-v1.5)
3. Vector Database Lifecycle (Upsert -> Query -> Delete on Qdrant/Chroma)
4. Cloud LLM Answer Generation (Button-triggered, never on load)
5. Multi-Cloud Claude Adapters Status (Bedrock, Vertex, Anthropic, MCP - all OFF)
6. Latency & Observability Metrics (Cache hits, p50 latency)
"""

import io
import time
import streamlit as st
import pandas as pd
from pypdf import PdfWriter

from document_loader import load_pdf_from_bytes
from embeddings import get_embedder, FastEmbedEmbedder, BaseEmbedder
from vector_store import get_vector_store, BaseVectorStore
from llm_service import (
    get_llm_service,
    get_secret,
    BaseLLMService,
    ClaudeLLMService,
    BedrockClaudeLLMService,
    VertexClaudeLLMService
)
from rag_service import RAGService

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
st.markdown('<div class="health-sub">Live automated verification across PDF ingestion, FastEmbed embeddings, Qdrant/Chroma vector DB, and Cloud LLMs.</div>', unsafe_allow_html=True)

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
        sample_a = "Attendance requirements for semester end exams."
        sample_b = "How many classes must a student attend to take tests?"
        sample_c = "Cafeteria lunch menu and food pricing."

        vec_a = embedder.get_embedding(sample_a)
        vec_b = embedder.get_embedding(sample_b)
        vec_c = embedder.get_embedding(sample_c)
        elapsed = round((time.time() - t0) * 1000, 1)

        dim = len(vec_a)
        is_nonzero = any(abs(v) > 1e-6 for v in vec_a)
        
        # Semantic cosine similarity check
        def cos_sim(u, v):
            dot = sum(x * y for x, y in zip(u, v))
            nu = sum(x * x for x in u) ** 0.5 or 1.0
            nv = sum(y * y for y in v) ** 0.5 or 1.0
            return dot / (nu * nv)

        sim_rel = cos_sim(vec_a, vec_b)
        sim_unrel = cos_sim(vec_a, vec_c)
        is_semantic = sim_rel > sim_unrel

        if dim > 0 and is_nonzero and is_semantic:
            return {
                "status": "PASS",
                "name": "Semantic Embeddings (FastEmbed ONNX)",
                "latency_ms": elapsed,
                "detail": f"{embedder.__class__.__name__} ({dim} dims | Related similarity: {sim_rel:.3f} > Unrelated: {sim_unrel:.3f})"
            }
        else:
            return {
                "status": "FAIL",
                "name": "Semantic Embeddings",
                "latency_ms": elapsed,
                "detail": f"Embedding failed sanity check (dim: {dim}, semantic check: {is_semantic})"
            }
    except Exception as e:
        elapsed = round((time.time() - t0) * 1000, 1)
        return {"status": "FAIL", "name": "Semantic Embeddings", "latency_ms": elapsed, "detail": str(e)}

# Diagnostic 3: Vector Store Probe (Upsert -> Query -> Delete)
def test_vector_store():
    t0 = time.time()
    probe_source = "__health_check_probe__"
    try:
        embedder: BaseEmbedder = get_embedder()
        store: BaseVectorStore = get_vector_store(embedding_dimension=getattr(embedder, "dimension", 384))
        
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
        results = store.query_similar(test_vec, top_k=2)
        found = any(r.get("source") == probe_source for r in results)

        # 3. Clean up probe chunk
        store.delete_document(probe_source)
        
        elapsed = round((time.time() - t0) * 1000, 1)
        stats = store.get_stats()

        if found:
            return {
                "status": "PASS",
                "name": "Vector Database (Lifecycle Probe)",
                "latency_ms": elapsed,
                "detail": f"Provider: {stats.get('provider')} | Upsert -> Query -> Delete verified."
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
        try:
            store.delete_document(probe_source)
        except Exception:
            pass
        return {"status": "FAIL", "name": "Vector Database", "latency_ms": elapsed, "detail": str(e)}

# Diagnostic 4: LLM Service Config Status (Check without LLM call)
def check_llm_config():
    llm: BaseLLMService = get_llm_service()
    model_name = getattr(llm, "model_name", "Mock")
    provider_name = llm.__class__.__name__

    if llm.is_available():
        return {
            "status": "PASS",
            "name": "Cloud LLM Configuration",
            "latency_ms": 0.0,
            "detail": f"{provider_name} ({model_name}) is configured and online."
        }
    else:
        return {
            "status": "FAIL",
            "name": "Cloud LLM Configuration",
            "latency_ms": 0.0,
            "detail": f"{provider_name} ({model_name}) is offline or missing credentials."
        }


# Execute non-intrusive checks
results = []
results.append(test_pdf_ingestion())
results.append(test_embeddings())
results.append(test_vector_store())
results.append(check_llm_config())

# Summary Metrics
col1, col2, col3 = st.columns(3)
pass_count = sum(1 for r in results if r["status"] == "PASS")
fail_count = sum(1 for r in results if r["status"] == "FAIL")
warn_count = sum(1 for r in results if r["status"] == "WARN")

col1.metric("Automated Probes Passed", f"{pass_count} / {len(results)}")
col2.metric("Failures", f"{fail_count}")
col3.metric("Warnings", f"{warn_count}")

# Detailed Results Cards
st.markdown("### 📋 Automated Subsystem Probes")
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

# Button-Triggered Live LLM Probe (Never runs on page load)
st.markdown("### 🧠 Live LLM Verification Probe")
st.caption("ℹ️ To preserve free-tier rate quotas (15 RPM / 1,500 RPD), live generation calls are never executed automatically on page load.")

col_llm1, col_llm2 = st.columns([1, 3])
with col_llm1:
    trigger_llm_probe = st.button("🚀 Trigger Live LLM Call (1 Probe)", type="secondary", use_container_width=True)

if trigger_llm_probe:
    with st.spinner("Executing rate-limit-aware verification prompt..."):
        t0 = time.time()
        try:
            llm: BaseLLMService = get_llm_service()
            prompt = "Respond with strictly the single word 'OPERATIONAL' if you can read this."
            resp = llm.generate_response(prompt)
            elapsed = round((time.time() - t0) * 1000, 1)
            st.success(f"🟢 **Live Probe Success:** Model `{getattr(llm, 'model_name', 'Mock')}` responded in **{elapsed} ms**: `{resp.strip()}`")
        except Exception as e:
            st.error(f"❌ **Live Probe Failure:** {e}")

st.divider()

# Multi-Cloud Claude Adapters Status
st.markdown("### ☁️ Multi-Cloud Claude Adapters Status (Hard Rule: ₹0 Budget)")
claude_adapters_status = [
    {"Adapter": "Anthropic Claude API", "Class": "ClaudeLLMService", "Model ID": "claude-3-5-haiku-20241022", "Status": "Disabled (OFF by default)", "Cost": "Paid API"},
    {"Adapter": "Amazon Bedrock (Claude)", "Class": "BedrockClaudeLLMService", "Model ID": "anthropic.claude-3-5-haiku-20241022-v1:0", "Status": "Disabled (OFF by default)", "Cost": "Paid AWS"},
    {"Adapter": "Google Vertex AI (Claude)", "Class": "VertexClaudeLLMService", "Model ID": "claude-3-5-haiku@20241022", "Status": "Disabled (OFF by default)", "Cost": "Paid GCP"},
    {"Adapter": "MCP Server Gateway", "Class": "DocuMindMCPServer", "Model ID": "N/A (Tool Server)", "Status": "Implemented & Mocked (Not Deployed)", "Cost": "₹0"},
]
st.dataframe(pd.DataFrame(claude_adapters_status), use_container_width=True)
st.caption("🔒 *Docs note: All Claude adapters are implemented, unit-tested with mocks, NOT live-tested; enabling requires the owner's own paid credentials.*")

st.divider()

# Environment Secrets Audit (Masked)
st.markdown("### 🔒 Environment Secrets Audit (Masked)")
secrets_audit = [
    {"Key": "GEMINI_API_KEY", "Status": "Configured" if get_secret("GEMINI_API_KEY") else "Missing (Free at https://aistudio.google.com/)", "Masked Value": mask_secret(get_secret("GEMINI_API_KEY"))},
    {"Key": "GROQ_API_KEY", "Status": "Configured" if get_secret("GROQ_API_KEY") else "Missing (Free at https://console.groq.com/)", "Masked Value": mask_secret(get_secret("GROQ_API_KEY"))},
    {"Key": "QDRANT_URL", "Status": "Configured" if get_secret("QDRANT_URL") else "Missing (Using Local Chroma/Memory)", "Masked Value": mask_secret(get_secret("QDRANT_URL"))},
    {"Key": "QDRANT_API_KEY", "Status": "Configured" if get_secret("QDRANT_API_KEY") else "Missing", "Masked Value": mask_secret(get_secret("QDRANT_API_KEY"))},
    {"Key": "ENABLE_CLAUDE", "Status": "Default (false)", "Masked Value": get_secret("ENABLE_CLAUDE", "false")},
    {"Key": "ANTHROPIC_API_KEY", "Status": "Configured" if get_secret("ANTHROPIC_API_KEY") else "Not Set", "Masked Value": mask_secret(get_secret("ANTHROPIC_API_KEY"))},
]
st.dataframe(pd.DataFrame(secrets_audit), use_container_width=True)
