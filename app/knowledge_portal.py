"""
Enterprise Knowledge Portal - the finished-product UI. Provided, complete.

A customer-support agent types a question; the portal retrieves from the
governed knowledge base (optionally scoped to a department) and shows a
grounded, cited answer plus the passages it used. It calls
src.pipeline.KnowledgeSystem - nothing here reimplements a pipeline stage.

Until the lab's activities are done, the imported stages raise
NotImplementedError; the portal catches that and shows what's left.

Known gap, deliberately not addressed by this bootcamp's scope: a delivered
answer and its citations aren't recorded anywhere once shown here. A real
bank deployment would need every generated answer retained on a compliance
schedule and preservable under a legal hold the moment it's disputed - this
portal has no such record today, so a specific past answer can't be
reconstructed later if someone challenges it. Worth a real design
conversation (where it logs, who can read it, how long it's kept) before
building it, not a silent addition here.
"""
import json
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.demo_backend import (  # noqa: E402
    is_demo_mode,
    prepare_demo_backend,
    reactivate_demo_backend,
)

# Must run before any `from src...` import in this process (see
# demo_backend.py) so a demo-mode launch resolves KnowledgeSystem and the
# six activity modules from reference_solution/, not the learner's own src/.
#
# Streamlit reruns this whole script top-to-bottom on every widget
# interaction (filter change, "Ask" click, expander open, example-query
# click) within the same long-lived process. Gate the expensive merge build
# (mkdtemp + 2 copytree + 7 copy2 + sys.path growth) behind st.session_state
# so it only happens once per session; every later rerun just re-asserts
# this session's already-built merge_root via the cheap
# reactivate_demo_backend() instead of rebuilding it from scratch.
DEMO_MODE = is_demo_mode()
if DEMO_MODE:
    if "kp_demo_merge_root" not in st.session_state:
        st.session_state.kp_demo_merge_root = prepare_demo_backend()
    else:
        reactivate_demo_backend(st.session_state.kp_demo_merge_root)

from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=True)
except Exception:
    pass

# dotenv's override=True load above re-asserts whatever VECTOR_STORE is set
# in .env on every rerun (Streamlit reruns this whole script top-to-bottom on
# every widget interaction), which would overwrite demo mode's own
# VECTOR_STORE=faiss (set once in prepare_demo_backend(), see
# app/demo_backend.py) after the very first rerun. Re-assert faiss here,
# after dotenv loads, on every rerun while DEMO_MODE is on, so demo mode
# always uses faiss regardless of what a learner's own .env is set to.
if DEMO_MODE:
    os.environ["VECTOR_STORE"] = "faiss"

st.set_page_config(page_title="Enterprise Knowledge Portal", page_icon="🏦", layout="wide")

st.markdown("""
<style>
  /* Shared typography stack (matches rag_course/index.html). */
  body, .block-container {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }

  .sub-header {color: #5a6a75; margin-bottom: 1.2rem;}

  /* Shared header pattern (copied verbatim from app/workbook_app.py so all
     three Streamlit apps share the same visual opening signature). */
  .hero-banner {
      background: linear-gradient(120deg, #0b2545 0%, #13315c 35%, #1f6f78 70%, #13315c 100%);
      background-size: 300% 300%;
      animation: heroShift 14s ease-in-out infinite;
      border-radius: 14px;
      padding: 1.6rem 2rem;
      margin-bottom: 1.25rem;
      box-shadow: 0 4px 18px rgba(11, 37, 69, 0.25);
  }
  @keyframes heroShift {
      0% {background-position: 0% 50%;}
      50% {background-position: 100% 50%;}
      100% {background-position: 0% 50%;}
  }
  @media (prefers-reduced-motion: reduce) {
      .hero-banner {animation: none;}
  }
  .hero-title {font-size: 1.7rem; font-weight: 700; color: #ffffff; margin: 0;}
  .hero-subtitle {font-size: 0.95rem; color: #cfe3ea; margin-top: 0.3rem;}

  /* Existing citation/refusal styling - unchanged, already correct. */
  .cite {background:#eef5f6; color:#13315c; border-left:4px solid #1f6f78; padding:.5rem .8rem; border-radius:6px; margin:.3rem 0;}
  .refused {background:#fbf6e7; color:#13315c; border-left:4px solid #b08900; padding:.8rem 1rem; border-radius:8px;}

  /* Card system: white panel, hairline border, two-layer shadow - extended
     here to the retrieved-passage list (previously a plain unbounded
     st.markdown per passage inside the expander). */
  .passage-card {
      background: #ffffff;
      color: #13315c;
      border: 1px solid #e6e5df;
      border-radius: 14px;
      padding: .6rem .9rem;
      margin-bottom: .6rem;
      box-shadow: 0 2px 8px rgba(11, 37, 69, .05), 0 12px 30px rgba(11, 37, 69, .07);
  }

  /* Shared vertical rhythm between major blocks. */
  h3, h4 {margin-top: 2rem; margin-bottom: 1rem;}
  [data-testid="stExpander"] {margin-top: 1rem; margin-bottom: 1.5rem;}

  /* Keyboard focus visibility. */
  .block-container button:focus-visible,
  .block-container a:focus-visible,
  .block-container [tabindex]:focus-visible {
      outline: 2.5px solid #c65d3b;
      outline-offset: 2px;
  }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero-banner">'
    '<p class="hero-title">🏦 Enterprise Knowledge Portal</p>'
    '<p class="hero-subtitle">Grounded answers from the bank\'s governed knowledge base.</p>'
    '</div>',
    unsafe_allow_html=True,
)


def _render_demo_mode_banner() -> None:
    """Unmissable, persistent indicator shown on every screen state while
    KP_DEMO_MODE=1 is active - this is running reference_solution/'s
    completed pipeline, not the learner's own src/ code. Called once at
    import time (top banner) and again inside the sidebar, so it stays
    visible before and after a question is asked."""
    st.error(
        "DEMO MODE (KP_DEMO_MODE=1): this portal is running the finished "
        "**reference_solution/**, not your own `src/`. Unset this variable "
        "to see your own pipeline here.",
        icon="⚠️",
    )


if DEMO_MODE:
    _render_demo_mode_banner()

DEPARTMENTS = ["(all departments)", "retail_banking", "loans", "cards", "operations", "compliance"]

MAX_QUESTION_LENGTH = 2000

# Maps a substring of a known NotImplementedError message (raised by a
# still-TODO src/ function) to a one-liner naming the activity and function
# that raised it. Built from each module's own "TODO (Activity N.N)"
# docstring header and tier marker, not guessed:
#   Activity 1.2 (Beginner)     src/ingestion/loaders.py        _load_*
#   Activity 1.3 (Beginner)     src/ingestion/document_processor.py
#   Activity 2.1 (Intermediate) src/chunking/chunker.py         split_text
#   Activity 2.1 (Intermediate) src/indexing/vector_store.py    build_documents/index_chunks/search
#   Activity 2.2 (Intermediate) src/retrieval/retriever.py      HybridRetriever.retrieve
#   Activity 2.3 (Intermediate) src/generation/answer_generator.py generate_answer
#   Activity 3.1 (Advanced)     src/observability/tracing.py    run_query_with_tracing
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
    """Map a NotImplementedError to a learner-friendly message that leads
    with the actionable instruction (which activity to complete in the
    workbook first) and keeps the specific function name available as a
    supporting detail, not the headline. Falls back to the raw exception
    text for anything that doesn't match a known stage, so the real error
    is never hidden."""
    text = str(exc)
    for needle, (activity, func) in STAGE_FAILURES.items():
        if needle in text:
            return (
                f"This needs {activity} finished first: {func} "
                "isn't implemented yet. Finish that activity in the "
                "workbook, then try again here."
            )
    return f"Stage failed: {text}"


@st.cache_resource(show_spinner="Setting up the knowledge system (the first run embeds the corpus)...")
def get_system(demo_base_path: str | None = None):
    from src.pipeline import KnowledgeSystem

    return KnowledgeSystem(reuse_persisted=True, base_path=demo_base_path)


@st.cache_data
def load_example_scenarios() -> list:
    """Load the demo example queries. Only the `question` field is ever
    surfaced in the UI - expected_sources/must_include/note are the answer
    key for the Activity 3.2 pytest smoke suite and must not leak here."""
    path = PROJECT_ROOT / "data" / "validation" / "test_scenarios.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("scenarios", [])
    except Exception as e:
        logger.warning(f"Could not load example scenarios: {e}")
        return []


key = os.getenv("OPENAI_API_KEY")
if not key or key in ("demo", "your_openai_or_openrouter_api_key_here"):
    st.warning("Add your `OPENAI_API_KEY` to `.env` to run the portal. It uses the key to embed queries and call the model.")
    st.stop()

if "current_question" not in st.session_state:
    st.session_state.current_question = ""


def _use_example_question(text: str) -> None:
    st.session_state.current_question = text


with st.sidebar:
    if DEMO_MODE:
        st.warning("**DEMO MODE:** answers come from `reference_solution/`, "
                   "not your `src/`.", icon="⚠️")
    st.header("Retrieval settings")
    dept = st.selectbox("Department filter", DEPARTMENTS)
    hybrid = st.toggle("Hybrid (vector + BM25)", value=True)
    st.caption("Change the department or switch hybrid off to see how the answer and sources change.")

scenarios = load_example_scenarios()
refusal_scenarios = [s for s in scenarios if s.get("expected_department") is None]
regular_scenarios = [s for s in scenarios if s.get("expected_department") is not None]

if scenarios:
    with st.expander("Example questions"):
        for s in regular_scenarios:
            st.button(s["question"], key=f"example_{s['id']}",
                      on_click=_use_example_question, args=(s["question"],))
        if refusal_scenarios:
            st.caption("Try one that should be refused")
            for s in refusal_scenarios:
                st.button(s["question"], key=f"example_{s['id']}",
                          on_click=_use_example_question, args=(s["question"],))

question = st.text_input("Ask a question", key="current_question",
                          placeholder="e.g. What is the minimum balance for a metro savings account?")
ask = st.button("Ask", type="primary")

if ask:
    question = question.strip()
    if not question:
        st.warning("Type a question first.")
    elif len(question) > MAX_QUESTION_LENGTH:
        st.warning(f"That question is too long ({len(question)} characters). "
                   f"Keep it under {MAX_QUESTION_LENGTH}.")
    else:
        logger.info(f"Question asked ({len(question)} chars): {question[:120]!r}")

        try:
            demo_base_path = str(st.session_state.kp_demo_merge_root) if DEMO_MODE else None
            system = get_system(demo_base_path)
        except NotImplementedError as e:
            logger.error(f"Pipeline build failed: {e}")
            st.error(stage_failed_message(e))
            st.stop()
        except Exception as e:
            logger.error(f"Pipeline build failed: {e}")
            st.error(f"Something went wrong with the model call: {e}")
            st.stop()

        metadata_filter = None if dept == "(all departments)" else {"department": dept}
        try:
            with st.spinner("Finding passages and writing the answer..."):
                result = system.answer_question(question, metadata_filter=metadata_filter, hybrid=hybrid)
        except NotImplementedError as e:
            logger.error(f"Pipeline query failed: {e}")
            st.error(stage_failed_message(e))
            st.stop()
        except Exception as e:
            logger.error(f"Pipeline query failed: {e}")
            st.error(f"Something went wrong with the model call: {e}")
            st.stop()

        if result.get("refused"):
            logger.info("Question refused: out of scope")
            st.markdown(f'<div class="refused"><b>No grounded answer.</b> {result["answer"]}</div>',
                        unsafe_allow_html=True)
        else:
            citations = result.get("citations", [])
            logger.info(f"Answer generated with {len(citations)} citation(s)")
            st.markdown("### Answer")
            st.write(result["answer"])
            st.markdown("### Citations")
            for c in citations:
                ver = f" · v{c['version']}" if c.get("version") else ""
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
