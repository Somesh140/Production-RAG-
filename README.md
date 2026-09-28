<div align="center">

# 🏦 Production RAG & Enterprise Knowledge Systems

### Turning a single-document RAG proof of concept into a production Enterprise Knowledge System

*A guided project for a retail bank: multi-source ingestion, enterprise metadata, retrieval optimization, grounded cited answers, observability, and containerized deployment.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-Vector%20DB-5A31F4)](https://www.trychroma.com/)
[![FAISS](https://img.shields.io/badge/FAISS-comparison%20path-0467DF)](https://github.com/facebookresearch/faiss)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-1C3C3C)](https://smith.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

---

## 🎯 Overview

A retail bank has a working RAG proof of concept that answers questions from **one** policy
document. It can't go enterprise-wide. It can't ingest multiple formats, keeps no document
context, can't filter by department, gives you no traceability, and has no deployment path.

In this bootcamp you take that POC and harden it, step by step, into a **production-ready
Enterprise Knowledge System** covering Retail Banking, Loans, Credit Cards, Operations, and
Compliance.

| # | Topic | Activity | Tier |
|---|---|---|---|
| 1 | Enterprise RAG architecture (POC → production) | 1.1 | 🟢 Beginner |
| 2 | Multi-source ingestion: PDF, Word, HTML, Markdown | 1.2 | 🟢 Beginner |
| 3 | Document processing & enterprise metadata | 1.3 | 🟢 Beginner |
| 4 | Chunking + embeddings into a vector DB (Chroma, with a FAISS comparison) | 2.1 | 🟡 Intermediate |
| 5 | Retrieval optimization: metadata filtering + hybrid retrieval | 2.2 | 🟡 Intermediate |
| 6 | Grounded response generation with citations | 2.3 | 🟡 Intermediate |
| 7 | Observability with LangSmith | 3.1 | 🟠 Advanced |
| 8 | Automated testing (pytest smoke suite) | 3.2 | 🟠 Advanced |
| 9 | Deployment with Docker | 3.3 | 🟠 Advanced |

**Estimated completion time:** ~3 hours · **Domain:** BFSI · **Lab:** Linux VM + VSCode

**Checkpoints** (pause here, everyone confirms, then move on):
1. Everyone can name the 7 POC→production gaps.
2. All 4 formats (PDF/DOCX/HTML/MD) load without error; the ungoverned file is identified.
3. Every chunk in the index carries department/doc_type/version metadata.
4. `split_text()` returns non-empty overlapping chunks for all three strategies.
5. `HybridRetriever.retrieve()` returns ranked chunks with metadata filtering working.
6. `generate_answer()` returns a cited answer for an in-corpus question and a refusal for an out-of-corpus one.
7. A traced query (or its local-trace fallback) shows retrieval + generation steps.
8. `pytest` smoke suite passes.
9. `docker compose up` serves the knowledge portal.

---

## 🧭 Start here

`hub.html` (repo root) is a single static page linking to the learner-facing
surfaces below, with the exact launch command for each. Open it directly in a browser,
or serve it however you like. It makes no network calls of its own and starts nothing, so
you still start each app with its own command.

## 📚 The guided workbook

The primary learner surface is the guided workbook: one Streamlit app,
`app/workbook_app.py`, covers all 9 activities. Edit the real `src/` file in your own IDE,
then run that activity's checks from the workbook. They run against exactly what's saved
on disk (a fresh subprocess each time). If you're stuck, run the completed reference
solution instead. Launch it with `streamlit run app/workbook_app.py`.

> [!TIP]
> Everything in `src/` marked `TODO` / `raise NotImplementedError` is what **you** implement.
> `reference_solution/` holds the completed version, so try each one yourself first.

> [!NOTE]
> `rag_course/index.html` is an instructor-only scrollytelling walkthrough of the pipeline
> (live widgets, no key/install needed) used for demos and intuition-building before the
> workbook. It is not distributed to learners and is git-ignored, see
> `scripts/build_course_data.py` if you need to regenerate its data locally.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- An OpenRouter API key (on your lab desktop, double-click the **Lab Details** icon for the key and a copy button)
- Docker (**Activity 3.3 only**) - the lab VMs are Linux and do not ship with Docker pre-installed. If
  `docker --version` fails, install it yourself:
  ```bash
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker $USER && newgrp docker
  sudo systemctl enable --now docker
  ```
  Then confirm with `docker --version && docker compose version`. If your account has no `sudo` access
  on the lab VM, Docker install is an instructor/lab-provisioning task, not something you can do
  yourself - flag it rather than getting stuck on Activity 3.3.
  If a `docker` command then fails with `permission denied while trying to connect to the docker API
  at unix:///var/run/docker.sock`, `newgrp docker` didn't take effect in your current terminal - either
  prefix the command with `sudo` for now, or close and reopen your terminal (or VS Code) after the
  `usermod` step so the new group membership actually applies, then verify with `groups` that `docker`
  is listed.

### Setup

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
# Edit .env and paste the key from Lab Details into OPENAI_API_KEY
```

```bash
# The guided workbook. The Code Assistant is built in (a floating
# panel on every activity page) - there is no separate app/port for it.
streamlit run app/workbook_app.py
```

> [!IMPORTANT]
> Run every command from the **project root**: modules resolve `data/` and `config/` relative to it.

---

## 🏗️ Project Structure

```
config/                 llm / ingestion / prompts YAML
data/
  knowledge_base/        BFSI corpus in 4 formats (18 governed docs) + manifest.csv
  validation/            end-to-end test scenarios
scripts/                instructor/maintainer tooling only, git-ignored, not part of the learner surface:
  generate_corpus.py    rebuilds the corpus + manifest (reviewable source, not opaque binaries)
  build_course_data.py  precomputes the instructor course's data (offline, no API)
  verify_reference.py   independent end-to-end check of the reference solution
  check_dependencies.py dependency vulnerability scan
src/
  ingestion/  loaders.py (1.2)   document_processor.py (1.3)
  chunking/   chunker.py (2.1)
  indexing/   embeddings.py (provided)   vector_store.py (2.1)
  retrieval/  retriever.py (2.2)
  generation/ answer_generator.py (2.3)
  observability/ tracing.py (3.1)
  pipeline.py            wires the stages (provided, complete)
  llm/, utils/           provided infrastructure
rag_course/index.html    instructor-only interactive scrollytelling walkthrough (self-contained,
                          git-ignored, not distributed to learners)
app/workbook_app.py      the guided workbook (primary learner surface): all 9 activities, run each against src/ via a
                          fresh subprocess, or run the reference solution instead. Also embeds the Code
                          Assistant (app/code_assistant_panel.py) as a floating chat panel docked to the
                          top-right of every activity page, scoped to whatever activity you're viewing,
                          and ends its Final Review page with a live "ask your own pipeline" section
                          (app/knowledge_portal_panel.py) - no separate app for either
app/knowledge_portal.py  the finished-product UI (provided, complete)
app/knowledge_portal_panel.py  shared ask/answer UI (department filter, hybrid toggle, example queries,
                          cited answer, retrieved passages) - used both by the "Try It Live" section at
                          the end of app/workbook_app.py's Final Review page and by knowledge_portal.py
app/code_assistant_panel.py  the Code Assistant: activity catalog, live TODO-instruction loading,
                          reference_solution/ grounding, reference-dump guard, and the chat UI itself.
                          The only place this chat exists - rendered as a floating panel in
                          app/workbook_app.py, no separate app or port
tests/test_pipeline.py   pytest smoke suite (3.2)
reference_solution/src/  completed versions of every TODO module
Dockerfile, docker-compose.yml   provided, complete (3.3)
```

---

## 🗂️ The sample corpus

`data/knowledge_base/` holds **18 governed documents** across five departments in four
formats (PDF, DOCX, HTML, Markdown), plus one deliberately **ungoverned** file
(`operations/draft_notes.md`, absent from `manifest.csv`) so Activity 1.3's exclusion path
is real. `manifest.csv` carries the governance metadata every chunk is tagged with:
`department, doc_type, product, version, effective_date, owner, title`. The Personal Loan
policy exists as a current v2.0 and a superseded v1.0 so Activity 2.2 can show version-aware
retrieval.

---

## 🛠️ Technology Stack

| Component | Technology |
|:--|:--|
| Language | Python 3.10+ |
| RAG framework | LangChain (`langchain-community`): loaders, splitters, retrievers |
| LLM / embeddings | `openai` SDK against an OpenAI-compatible API (OpenRouter), `openai/gpt-4`, `text-embedding-3-small` |
| Vector store | **FAISS** (default), **Chroma** (comparison path; needs Python <=3.13, no Python 3.14 wheel yet); switch via `VECTOR_STORE` in `.env` |
| Hybrid retrieval | `rank-bm25` lexical score fused with vector similarity |
| Observability | LangSmith (`langsmith`, two env vars, one traced entry point) |
| Testing | pytest smoke suite (structural, no network) |
| Packaging | Docker + Docker Compose |
| Learning experience | Streamlit guided workbook (`app/workbook_app.py`) for learners + a self-contained scrollytelling HTML (vanilla JS, instructor-only) |
| Finished-product UI | Streamlit: the knowledge portal (`app/knowledge_portal.py`), also embedded live at the end of the workbook's Final Review page |
| Code assistant | Chat-based code generation grounded in `reference_solution/`; a single floating panel built into the workbook (`app/code_assistant_panel.py`), no separate app |
| Config | YAML (`config/*.yaml`) + `.env` via `python-dotenv` |

---

## 📄 License

Educational project for the Agentic AI for Enterprise Delivery Bootcamp.
