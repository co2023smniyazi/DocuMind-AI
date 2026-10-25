# Jira Project Management and Tracking Guide
**Project:** College Document Q&A Assistant Using Local RAG  
**Project Key:** `RAG`  
**Framework:** Agile / Scrum  

---

## 1. Overview of Jira's Role
In this final-year mini-project:
- **LLM/RAG Application:** The software project being built.
- **Jira:** The project management tool used to define requirements, break down epics into user stories, assign sprint tasks across the 4-student team, track workflow transitions, and log testing defects.
- **Key Note:** Jira is strictly for project management and sprint tracking. It does not run the LLM.

```
                    JIRA
                     |
             Project Management
                     |
        +------------+------------+
        |            |            |
    Requirements  Development   Testing
        |            |            |
        +------------+------------+
                     |
                     v
              LLM MINI PROJECT
                     |
                     v
         College Document Q&A
              using RAG
```

---

## 2. Step 1: Create Jira Free Account & Project
1. Go to [Atlassian Jira](https://www.atlassian.com/software/jira) and sign up for a **Free Jira Cloud** account (₹0 cost).
2. Choose **Scrum Project Template**.
3. Set:
   - **Project Name:** `College RAG Assistant`
   - **Project Key:** `RAG`
   - **Access:** Team-Managed or Company-Managed Scrum.

---

## 3. Step 2: Configure Jira Workflow
Under **Project Settings > Workflows**, configure the 5-stage agile lifecycle:

```
[ TO DO ] ──> [ IN PROGRESS ] ──> [ CODE REVIEW ] ──> [ TESTING ] ──> [ DONE ]
```

### Transition Example (RAG-8: Generate Embeddings):
1. **TO DO:** Issue sits in Sprint 2 Backlog.
2. **IN PROGRESS:** Student 2 begins coding `embeddings.py` using Ollama `nomic-embed-text`.
3. **CODE REVIEW:** Student 2 raises a pull request; Student 3 reviews batching and error handling.
4. **TESTING:** Student 4 executes `pytest tests/test_rag.py`.
5. **DONE:** Tests pass with 768-dim vector embeddings confirmed.

---

## 4. Step 3: Create the 6 Epics

| Epic Key | Epic Name | Description | Total Story Points |
|---|---|---|---|
| **EPIC-1** | Requirements and Dataset | Problem definition, document collection, 50 sample questions | 11 SP |
| **EPIC-2** | Local LLM Setup | Ollama runtime, model downloading, and local verification | 5 SP |
| **EPIC-3** | RAG Pipeline | PDF extraction, chunking, embeddings, ChromaDB, retrieval, LLM integration | 37 SP |
| **EPIC-4** | User Interface | Streamlit web interface and source citation visualization | 8 SP |
| **EPIC-5** | Testing and Evaluation | Functional unit testing, retrieval accuracy, hallucination benchmark | 15 SP |
| **EPIC-6** | Documentation and Demonstration | Final project report, Jira tracking summary, viva demonstration | 5 SP |

---

## 5. Step 4: Import Product Backlog (RAG-1 to RAG-20)
You can directly import all 20 backlog items into Jira using the included `jira_management/jira_backlog.csv` file:
1. In Jira, navigate to **Settings > System > External System Import**.
2. Select **CSV Import**.
3. Upload `jira_backlog.csv`.
4. Map the fields:
   - `Summary` -> Summary
   - `Issue Type` -> Issue Type
   - `Description` -> Description
   - `Epic Link` -> Epic Link
   - `Story Points` -> Story Points
   - `Priority` -> Priority
   - `Assignee` -> Assignee
5. Click **Begin Import**. All 20 items will be populated into the Jira backlog instantly!

---

## 6. Step 5: User Stories & Acceptance Criteria

### RAG-6: PDF Text Extractor
- **User Story:** As a system developer, I want to extract text from uploaded PDF documents so that their contents can be processed by the RAG system.
- **Acceptance Criteria:**
  1. System accepts PDF documents.
  2. Text is extracted from all readable pages.
  3. File name is retained.
  4. Empty pages do not cause application failure.

### RAG-10: Semantic Retrieval
- **User Story:** As a user, I want the system to retrieve document sections related to my question so that the LLM receives relevant information.
- **Acceptance Criteria:**
  1. User question is converted into an embedding.
  2. Vector database is searched.
  3. Top relevant chunks are returned.
  4. Source information is preserved.

### RAG-11: LLM Answer Generation
- **User Story:** As a student, I want to ask questions in natural language so that I can easily understand information contained in college documents.
- **Acceptance Criteria:**
  1. User can enter a question.
  2. Relevant document chunks are retrieved.
  3. Retrieved chunks are provided to the LLM.
  4. LLM generates an understandable answer.
  5. Answer is based on retrieved context.
  6. Source document is displayed.

### RAG-14: Prevent Hallucination
- **User Story:** As a user, I want the application to indicate when information is unavailable so that I am not provided with fabricated information.
- **Acceptance Criteria:**
  1. If information is not sufficiently supported by retrieved documents: `"Information not found in the available documents."` must be returned.

---

## 7. Step 6: 4-Week Sprint Schedule

### Sprint 1: Foundation (Week 1)
- **Goal:** Set up local environment, gather documents, verify local Ollama LLM.
- **Items:** RAG-1, RAG-2, RAG-3, RAG-4, RAG-5 (16 SP)

### Sprint 2: RAG Development (Week 2)
- **Goal:** Convert raw PDF documents into a searchable semantic knowledge base.
- **Items:** RAG-6, RAG-7, RAG-8, RAG-9, RAG-10 (21 SP)

### Sprint 3: Complete LLM Application (Week 3)
- **Goal:** Connect retrieval with local LLM, implement strict anti-hallucination prompt, build Streamlit UI.
- **Items:** RAG-11, RAG-12, RAG-13, RAG-14, RAG-15, RAG-16 (22 SP)

### Sprint 4: Testing, Evaluation & Submission (Week 4)
- **Goal:** Run 50-question benchmark, calculate accuracy and safe refusal rate, fix defects, present demo.
- **Items:** RAG-17, RAG-18, RAG-19, RAG-20 (20 SP)

---

## 8. Step 7: Team Member Role Distribution (3–4 Students)

| Student | Assigned Role | Primary Jira Tasks |
|---|---|---|
| **Student 1** | Data Preparation & Document Processing | RAG-1, RAG-2, RAG-3, RAG-6, RAG-20 |
| **Student 2** | Embeddings, Vector Storage & Retrieval | RAG-7, RAG-8, RAG-9, RAG-10, RAG-19 |
| **Student 3** | LLM Integration & Prompt Engineering | RAG-4, RAG-5, RAG-11, RAG-12, RAG-13, RAG-14 |
| **Student 4** | Streamlit UI, Testing & Evaluation | RAG-15, RAG-16, RAG-17, RAG-18 |

