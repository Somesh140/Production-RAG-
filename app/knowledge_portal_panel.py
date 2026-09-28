"""
Shared "ask the finished system a question" panel.

Extracted out of app/knowledge_portal.py the same way the Code Assistant's
chat logic lives in app/code_assistant_panel.py: so the exact same
retrieve+generate UI (department filter, hybrid toggle, example queries,
cited answer, retrieved passages) can be rendered two ways -
  - as its own full page (app/knowledge_portal.py, unchanged), and
  - as an embedded "Try It Live" section at the end of the guided workbook's
    Final Review page, so a learner who just finished all 9 activities can
    immediately ask their own pipeline a real question without switching to
    a fourth app/tab.

This module never runs in KP_DEMO_MODE (that mode exists so an instructor/
grader can inspect the completed reference_solution/ pipeline in isolation -
irrelevant inside the learner's own workbook, which is about *their* src/
code) and never calls st.set_page_config or st.stop(): the caller owns page
framing, and a missing OPENAI_API_KEY is handled as an inline message so the
rest of the caller's page still renders.
"""
import json
import os
from pathlib import Path
from typing import Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent

DEPARTMENTS = ["(all departments)", "retail_banking", "loans", "cards", "operations", "compliance"]
MAX_QUESTION_LENGTH = 2000

# Same mapping app/knowledge_portal.py uses - kept in sync manually since the
# two modules intentionally stay independent (this one never imports
# knowledge_portal.py, to avoid dragging its DEMO_MODE/page-config code into
# the workbook process).
STAGE_FAILURES = {
    "_load_markdown is not implemented": (
        "Activity 1.2 (Multi-Source Ingestion)", "`loaders._load_markdown`"
    ),
    "_load_pdf is not implemented": (
        "Activity 1.2 (Multi-Source Ingestion)", "`loaders._load_pdf`"
    ),
    "_load_docx is not implemented": (
        "Activity 1.2 (Multi-Source Ingestion)", "`loaders._load_docx`"
    ),
    "_load_html is not implemented": (
        "Activity 1.2 (Multi-Source Ingestion)", "`loaders._load_html`"
    ),
    "build_metadata is not implemented": (
        "Activity 1.3 (Metadata & Governance)", "`document_processor.build_metadata`"
    ),
    "process_document is not implemented": (
        "Activity 1.3 (Metadata & Governance)", "`document_processor.process_document`"
    ),
    "split_text is not implemented": (
        "Activity 2.1 (Chunking + Vector Store Indexing)", "`chunker.split_text`"
    ),
    "build_documents is not implemented": (
        "Activity 2.1 (Chunking + Vector Store Indexing)", "`vector_store.build_documents`"
    ),
    "index_chunks is not implemented": (
        "Activity 2.1 (Chunking + Vector Store Indexing)", "`vector_store.index_chunks`"
    ),
    "search is not implemented": (
        "Activity 2.1 (Chunking + Vector Store Indexing)", "`vector_store.search`"
    ),
    "HybridRetriever.retrieve is not implemented": (
        "Activity 2.2 (Hybrid Retrieval)", "`HybridRetriever.retrieve`"
    ),
    "generate_answer is not implemented": (
        "Activity 2.3 (Grounded Answer Generation)", "`answer_generator.generate_answer`"
    ),
    "run_query_with_tracing is not implemented": (
        "Activity 3.1 (Observability & Tracing)", "`tracing.run_query_with_tracing`"
    ),
}


def stage_failed_message(exc: Exception) -> str:
    text = str(exc)
    for needle, (activity, func) in STAGE_FAILURES.items():
        if needle in text:
            return (
                f"This needs {activity} finished first: {func} "
                "isn't implemented yet. Finish that activity above, then try again here."
            )
    return f"Stage failed: {text}"


@st.cache_resource(show_spinner="Setting up the knowledge system (the first run embeds the corpus)...")
def _get_system():
    from src.pipeline import KnowledgeSystem

    return KnowledgeSystem(reuse_persisted=True)


@st.cache_data
def _load_example_scenarios() -> list:
    path = PROJECT_ROOT / "data" / "validation" / "test_scenarios.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("scenarios", [])
    except Exception:
        return []


def render_portal(key_prefix: str = "") -> None:
    """Render the ask/answer UI. Safe to call once per page; uses
    key_prefix to keep its widget keys distinct from anything else on the
    same page (e.g. app/workbook_app.py's own buttons)."""
    key = os.getenv("OPENAI_API_KEY")
    if not key or key in ("demo", "your_openai_or_openrouter_api_key_here"):
        st.info("Add your `OPENAI_API_KEY` to `.env` to try the finished system here.")
        return

    question_state_key = f"{key_prefix}kp_current_question"
    if question_state_key not in st.session_state:
        st.session_state[question_state_key] = ""

    def _use_example(text: str) -> None:
        st.session_state[question_state_key] = text

    scenarios = _load_example_scenarios()
    refusal_scenarios = [s for s in scenarios if s.get("expected_department") is None]
    regular_scenarios = [s for s in scenarios if s.get("expected_department") is not None]

    filt_col, toggle_col = st.columns([2, 1])
    with filt_col:
        dept = st.selectbox("Department filter", DEPARTMENTS, key=f"{key_prefix}kp_dept")
    with toggle_col:
        hybrid = st.toggle("Hybrid (vector + BM25)", value=True, key=f"{key_prefix}kp_hybrid")

    if scenarios:
        with st.expander("Example questions"):
            for s in regular_scenarios:
                st.button(s["question"], key=f"{key_prefix}kp_example_{s['id']}",
                          on_click=_use_example, args=(s["question"],))
            if refusal_scenarios:
                st.caption("Try one that should be refused")
                for s in refusal_scenarios:
                    st.button(s["question"], key=f"{key_prefix}kp_example_{s['id']}",
                              on_click=_use_example, args=(s["question"],))

    question = st.text_input(
        "Ask a question",
        key=question_state_key,
        placeholder="e.g. What is the minimum balance for a metro savings account?",
    )
    ask = st.button("Ask", type="primary", key=f"{key_prefix}kp_ask")

    if not ask:
        return

    question = question.strip()
    if not question:
        st.warning("Type a question first.")
        return
    if len(question) > MAX_QUESTION_LENGTH:
        st.warning(f"That question is too long ({len(question)} characters). Keep it under {MAX_QUESTION_LENGTH}.")
        return

    try:
        system = _get_system()
    except NotImplementedError as e:
        st.error(stage_failed_message(e))
        return
    except Exception as e:
        st.error(f"Something went wrong with the model call: {e}")
        return

    metadata_filter = None if dept == "(all departments)" else {"department": dept}
    try:
        with st.spinner("Finding passages and writing the answer..."):
            result = system.answer_question(question, metadata_filter=metadata_filter, hybrid=hybrid)
    except NotImplementedError as e:
        st.error(stage_failed_message(e))
        return
    except Exception as e:
        st.error(f"Something went wrong with the model call: {e}")
        return

    st.markdown('<div class="kp-result-enter">', unsafe_allow_html=True)
    if result.get("refused"):
        st.markdown(f'<div class="refused"><b>No grounded answer.</b> {result["answer"]}</div>',
                    unsafe_allow_html=True)
    else:
        citations = result.get("citations", [])
        st.markdown("##### Answer")
        st.write(result["answer"])
        if citations:
            st.markdown("##### Citations")
            for c in citations:
                ver = f" &middot; v{c['version']}" if c.get("version") else ""
                st.markdown(f'<div class="cite">[{c["marker"]}] {c["source"]}{ver}</div>', unsafe_allow_html=True)

    errs = result.get("validation_errors")
    if errs:
        st.warning(f"Answer validation flagged: {errs}")

    with st.expander("Retrieved passages"):
        for h in result.get("retrieved_chunks", []):
            st.markdown(
                f'<div class="passage-card"><b>{h["source"]}</b> &middot; score {h.get("score", 0):.3f} '
                f'&middot; <code>{h.get("chunk_id", "")}</code></div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)
