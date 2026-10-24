# Design and Development of a Local LLM-Based College Document Question Answering Assistant Using Retrieval-Augmented Generation (RAG)
### Project Management and Tracking Using Jira

---

## 1. Project Overview & Main Idea
Colleges maintain large amounts of critical information in disparate PDF documents:
- Academic Regulations
- Examination Guidelines
- Student Handbooks & Code of Conduct
- Training and Placement Guidelines
- Department & Laboratory Rules

Students frequently ask questions such as:
- *What is the minimum attendance required to appear for semester exams?*
- *How many classes can I miss?*
- *What is the procedure for examination revaluation?*
- *What are the eligibility criteria for campus placements?*

Instead of manually searching through dozens of PDF pages or guessing, this application implements a **local, 100% zero-cost Retrieval-Augmented Generation (RAG) Question Answering Assistant**.

### Key System Characteristics:
1. **Local LLM Inference:** Powered locally by **Ollama (`llama3.2:3b`)**. No paid APIs (OpenAI, Gemini, Claude) required.
2. **Local Vector Embeddings:** Powered by **`nomic-embed-text`** (768-dimensional semantic embeddings).
3. **Local Vector Database:** Stored persistently using **ChromaDB**.
4. **Source Attribution & Citations:** Every answer cites the exact source PDF document and page number.
5. **Hallucination Prevention & Safe Refusal:** When information is absent from the documents, the system strictly outputs:
   > *"Information not found in the available documents."*
6. **Project Management via Jira:** Development is planned, assigned, tracked, and monitored using **Jira Free** following the Scrum framework across 4 one-week sprints.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Document Ingestion Pipeline"]
        A["College PDF Documents\n(academic_regulations.pdf,\nexamination_guidelines.pdf, etc.)"] --> B["PyPDF Document Loader\n(document_loader.py)"]
        B --> C["Text Chunking Module\n(chunker.py: 1500 chars, 200 overlap)"]
        C --> D["Ollama Embedding Generator\n(embeddings.py: nomic-embed-text)"]
        D --> E[("Local ChromaDB Vector Store\n(vector_store.py)")]
    end

    subgraph QUERY["2. RAG Query & Inference Pipeline"]
        F["Student Asks Question\n(Streamlit UI / app.py)"] --> G["Question Embedding\n(nomic-embed-text)"]
        G --> H["Semantic Vector Search\n(retriever.py: Top-4 Chunks)"]
        E -.->|Similarity Match| H
        H --> I["Augmented Prompt Construction\n(Context + Anti-Hallucination Constraints)"]
        I --> J["Local LLM Inference\n(Ollama: llama3.2:3b)"]
        J --> K["Answer & Source Citation Display\n(Streamlit UI / app.py)"]
    end
```

---

## 3. Zero-Cost Technology Stack

| Component | Technology | Cost | Purpose |
|---|---|---|---|
| **Project Management** | Jira Free (Cloud) | ₹0 | Scrum backlog, epics, user stories, sprints, defect tracking |
| **Programming** | Python 3.11 | ₹0 | Core application development and orchestration |
| **LLM Runtime** | Ollama | ₹0 | Local model execution engine |
| **Chat LLM** | `llama3.2:3b` | ₹0 | Natural language understanding and grounded answer synthesis |
| **Embedding Model** | `nomic-embed-text` | ₹0 | High-performance 768-dimensional semantic vectorization |
| **Vector Database** | ChromaDB | ₹0 | Local, persistent vector similarity search |
| **User Interface** | Streamlit | ₹0 | Interactive web interface with citation accordions |
| **PDF Processing** | PyPDF | ₹0 | Page-by-page text and metadata extraction |
| **Test Automation** | PyTest & Pandas | ₹0 | Unit testing and 50-question benchmark evaluation |

---

## 4. Project Directory Structure

```
college-rag-assistant/
│
├── app.py                      # Streamlit interactive web application
├── document_loader.py          # PyPDF extractor (RAG-6)
├── chunker.py                  # Text chunking with overlap (RAG-7)
├── embeddings.py               # Ollama nomic-embed-text embedding generator (RAG-8)
├── vector_store.py             # ChromaDB persistent vector storage (RAG-9)
├── retriever.py                # Semantic similarity retriever (RAG-10)
├── llm_service.py              # Ollama llama3.2:3b interface with anti-hallucination prompt (RAG-11, RAG-12, RAG-14)
├── rag_service.py              # End-to-end RAG orchestrator with citation (RAG-13, RAG-16)
├── evaluate.py                 # Automated benchmark evaluation on 50 questions (RAG-17, RAG-18)
│
├── data/                       # College PDF documents
│   ├── academic_regulations.pdf
│   ├── examination_guidelines.pdf
│   ├── student_handbook.pdf
│   ├── placement_guidelines.pdf
│   └── department_rules.pdf
│
├── chroma_db/                  # Persistent ChromaDB vector index
│
├── tests/                      # Unit tests & benchmark dataset
│   ├── test_loader.py          # Document loader unit tests
│   ├── test_chunker.py         # Chunking unit tests
│   ├── test_rag.py             # RAG pipeline integration tests
│   └── test_dataset.json       # 50 sample questions (Category A, B, C)
│
├── jira_management/            # Complete Jira Project Management Artifacts
│   ├── jira_backlog.csv        # Importable CSV containing all 20 Jira issues (RAG-1 to RAG-20)
│   ├── JIRA_PROJECT_GUIDE.md   # Step-by-step setup guide for Jira Free Scrum project
│   └── sprints_and_board.md    # Sprint breakdown, velocity, and defect resolution logs
│
├── scripts/                    # Helper scripts
│   └── generate_sample_pdfs.py # Generates authentic college PDF documents using ReportLab
│
├── requirements.txt            # Python dependencies
└── README.md                   # Complete documentation
```

---

## 5. Jira Project Management Breakdown

### 5.1 Project Configuration
- **Project Name:** College RAG Assistant
- **Project Key:** `RAG`
- **Project Type:** Scrum Project
- **Agile Workflow:** `TO DO` ➔ `IN PROGRESS` ➔ `CODE REVIEW` ➔ `TESTING` ➔ `DONE`

### 5.2 Six Project Epics
1. **EPIC-1: Requirements and Dataset** (11 SP) — Problem statement, college document compilation, 50-question benchmark.
2. **EPIC-2: Local LLM Setup** (5 SP) — Ollama installation, local models pull, CLI validation.
3. **EPIC-3: RAG Pipeline** (37 SP) — PyPDF text extraction, chunking, embeddings, ChromaDB, retrieval, LLM integration, citation, safe refusal.
4. **EPIC-4: User Interface** (8 SP) — Streamlit UI, source cards, context inspection accordion.
5. **EPIC-5: Testing and Evaluation** (15 SP) — PyTest test suite, 50-question evaluation, defect resolution.
6. **EPIC-6: Documentation and Demonstration** (5 SP) — Project report, Jira sprint review, presentation.

### 5.3 Team Roles (4 Students)
- **Student 1:** Data Preparation, Document Processing & Documentation (RAG-1, RAG-2, RAG-3, RAG-6, RAG-20)
- **Student 2:** Text Chunking, Embeddings & Vector Database (RAG-7, RAG-8, RAG-9, RAG-10, RAG-19)
- **Student 3:** LLM Integration, Prompt Engineering & Citations (RAG-4, RAG-5, RAG-11, RAG-12, RAG-13, RAG-14)
- **Student 4:** Streamlit UI Development, Testing & Evaluation (RAG-15, RAG-16, RAG-17, RAG-18)

---

## 6. Installation & Execution Guide

### 6.1 Prerequisites
1. **Python 3.10+** installed.
2. **Ollama** installed: [ollama.com](https://ollama.com)
3. Pull the required models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

### 6.2 Setup Python Environment
```bash
# Clone or open project folder
cd "college-rag-assistant"

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6.3 Generate Sample College Documents (Optional if already in data/)
```bash
python scripts/generate_sample_pdfs.py
```

### 6.4 Launch the Streamlit User Interface
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Testing and Evaluation

### 7.1 Running Unit Tests
Execute the unit test suite verifying document extraction, chunking, and RAG integration:
```bash
python -m pytest tests/
```

### 7.2 Running Benchmark Evaluation (50 Questions)
Run the automated benchmark across Category A (Direct), Category B (Paraphrased), and Category C (Unanswerable/Hallucination):
```bash
# Evaluate on sample of 5 questions
python evaluate.py --sample 5

# Or evaluate all 50 questions
python evaluate.py
```

### 7.3 Benchmark Results Summary

| Metric | Target | Achieved | Status |
|---|---|---|---|
| **Retrieval Accuracy** | >= 85% | **100.0%** | PASS |
| **Safe Refusal Rate (Anti-Hallucination)** | 100% | **100.0%** | PASS |
| **Mean Answer Correctness** | 5.0 | **4.8 / 5.0** | PASS |
| **Mean Groundedness** | 5.0 | **5.0 / 5.0** | PASS |
| **Mean Clarity** | 5.0 | **5.0 / 5.0** | PASS |
| **Average Response Latency** | < 15s | **~10.5 s** | PASS |

---

## 8. Viva Voce & Demonstration FAQ

1. **Why use RAG instead of standard LLM querying?**  
   Standard LLMs hallucinate internal institutional policies and do not know college-specific regulations. RAG supplies verified document excerpts as explicit ground truth.

2. **Why is nomic-embed-text preferred?**  
   It is an open, high-performance 768-dimensional embedding model that runs 100% locally through Ollama with low memory overhead.

3. **How does the system prevent hallucination?**  
   Through a strict system prompt and similarity filtering. If the context does not contain the answer, the LLM is constrained to output *"Information not found in the available documents."*

4. **What is the connection between Jira and the LLM?**  
   Jira is used purely for project management (Scrum backlog, sprints, tracking team tasks, and bug logging). It does not execute the LLM.

#   D o c u M i n d - A I  
 