"""
Guided Project Workbook: interactive Streamlit runner.

The single entry point for the hands-on build (replaces the marimo reactive
notebooks under notebooks_marimo_archive/ - see BUILD_PLAN.md for the
redesign note). Edit the actual src/ files in your own IDE, then run each
activity's validation against them via a fresh subprocess (so results always
reflect exactly what's saved on disk, with no stale-import/kernel-state
surprises). Every script below is self-contained per activity (it rebuilds
any earlier-activity object it needs, or falls back gracefully when it
can't), so activities can be run in any order.

This app does NOT gate the whole page on OPENAI_API_KEY the way
app/knowledge_portal.py and app/code_assistant.py do: most of these 9
activities (1.1, 1.2, 1.3, 2.2's retrieval-fusion logic, 3.1's tracing
mechanics, 3.2, 3.3) need no key at all. Only the handful of steps that
truly call the embeddings/chat API (2.1's index_chunks/search, 2.3's
generate_answer) surface a missing/invalid key as a [WARN] in that
activity's own output, via try/except - never a page-level block.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.code_assistant_panel import render_chat as render_code_assistant_chat  # noqa: E402
from app.knowledge_portal_panel import render_portal as render_knowledge_portal  # noqa: E402
# override=True: load_dotenv() otherwise refuses to replace an env var that's
# already set in the shell, so a stray OPENAI_API_KEY left over from some
# other project (or a global shell profile export) would silently shadow
# this project's own .env with no error, just a confusing wrong-key failure.
load_dotenv(PROJECT_ROOT / ".env", override=True)

st.set_page_config(
    page_title="Guided Project Workbook",
    page_icon=None,
    layout="wide",
    # "expanded" forces the sidebar drawer open on first load regardless of
    # viewport size. On desktop the sidebar renders as an in-flow column next
    # to the main content, so this is harmless there - but on narrow
    # (phone/small-tablet) viewports Streamlit's own responsive CSS turns the
    # sidebar into a fixed-position overlay drawer with a dark scrim behind
    # it. Forcing it open on load means that scrim covers almost the entire
    # viewport on first render, leaving only a thin sliver of the main
    # content visible at the edge - the "content clipped to ~115-150px,
    # rest blank dark background" symptom. "auto" lets Streamlit's own
    # breakpoint decide: expanded on desktop-width viewports, collapsed
    # (drawer closed, main content full-width) on narrow ones.
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
    /* The floating top toolbar (Deploy button, hamburger menu) is fixed to
       the viewport, not the page - it stays in place as content scrolls, so
       top padding on .block-container only protects the initial render
       position and does nothing for content that later scrolls up into that
       band. The real fix is .streamlit/config.toml's client.toolbarMode =
       "minimal", which removes the toolbar entirely. This rule is a
       defensive backstop in case toolbarMode ever gets reset (e.g. a
       different config.toml picked up at a different cwd): it forces the
       toolbar out of the flow so it can never overlap page content even if
       it's still rendered. Some top padding is still kept so content isn't
       flush against the very top edge of the viewport. */
    .block-container {padding-top: 2rem;}
    [data-testid="stToolbar"] {visibility: hidden; height: 0; position: absolute;}
    /* The rule above hides the whole toolbar element, but Streamlit renders
       the sidebar's own open/collapse toggle button
       ([data-testid="stExpandSidebarButton"]) AS A CHILD of that same
       toolbar element whenever the sidebar starts collapsed - which is
       exactly what happens on narrow (phone/small-tablet) viewports now
       that initial_sidebar_state is "auto" (see the set_page_config comment
       above). Hiding the toolbar was silently taking the toggle down with
       it, leaving no way to open the sidebar - and therefore no way to
       reach Phase/activity nav - below the desktop breakpoint. visibility
       is inherited, so an explicit visibility:visible here re-enables just
       this one descendant even though its ancestor is hidden; position:
       fixed anchors it to the viewport instead of the now-zero-height,
       position:absolute toolbar box it lives inside, so its own geometry
       no longer depends on that collapsed ancestor at all. */
    [data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
        display: flex !important;
        position: fixed !important;
        top: 0.6rem;
        left: 0.6rem;
        height: auto !important;
        z-index: 999999;
        pointer-events: auto !important;
    }
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
    textarea {font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;}

    /* Shared typography stack (matches rag_course/index.html). */
    body, .block-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Card treatment (white panel, hairline border, two-layer shadow - same
       treatment app/knowledge_portal.py's .cite/.refused/.passage-card
       already use), applied here to the sidebar activity picker. The
       stRadio widget has no wrapping element to add a class to, so it's
       styled by its own existing data-testid instead of a div/class. No
       generic .card class exists in this file: nothing in workbook_app.py
       emits class="card" (every other card-look treatment targets a
       specific Streamlit testid instead), so it isn't defined here either -
       reintroduce it only if/when something actually needs it. */
    [data-testid="stSidebar"] div[data-testid="stRadio"] {
        background: #ffffff;
        border: 1px solid #e6e5df;
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 2px 8px rgba(11, 37, 69, .05), 0 12px 30px rgba(11, 37, 69, .07);
    }
    /* Status badges: color + left-border + icon + label, consistent with
       the portal's .cite/.refused pattern - never color alone. Only
       pass/fail are ever emitted (see the Environment Validation expander
       below): missing_packages/key_missing/missing_paths are each strictly
       binary, so there is no badge-warn state to reach. If a genuinely
       partial/warn condition is added later, reintroduce a .badge-warn rule
       alongside it then. */
    .badge-pass, .badge-fail {
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin: 0.4rem 0;
        font-size: 0.92rem;
    }
    .badge-pass {background: #eaf5ef; color: #0b2545; border-left: 4px solid #1f6f78;}
    .badge-fail {background: #fbeceb; color: #0b2545; border-left: 4px solid #b3372c;}

    /* Shared vertical rhythm between major blocks. */
    .hero-banner {margin-top: 0.5rem;}
    h3, h4 {margin-top: 2rem; margin-bottom: 1rem;}
    .cite, .refused, .badge-pass, .badge-fail {margin-bottom: 1rem;}
    [data-testid="stExpander"] {margin-top: 1rem; margin-bottom: 1rem;}

    /* Keyboard focus visibility. */
    .block-container button:focus-visible,
    .block-container a:focus-visible,
    .block-container [tabindex]:focus-visible {
        outline: 2.5px solid #c65d3b;
        outline-offset: 2px;
    }

    /* Floating Code Assistant panel (see render_code_assistant_dock()
       below): pins its trigger button to the right edge of the viewport so
       a learner can open the assistant from anywhere on an activity page
       without hunting for a separate tab/app. There is only one
       st.popover() on this page, so targeting the generic stPopover testid
       is safe - no extra wrapper markup needed. */
    div[data-testid="stPopover"] {
        position: fixed !important;
        top: 5.5rem;
        right: 1.5rem;
        left: auto !important;
        width: fit-content !important;
        z-index: 999998;
    }
    div[data-testid="stPopover"] > div > button {
        border-radius: 999px !important;
        box-shadow: 0 4px 14px rgba(11, 37, 69, 0.35);
        background: #13315c !important;
        color: #ffffff !important;
        border: none !important;
    }
    div[data-testid="stPopoverBody"] {
        width: min(420px, 90vw) !important;
        max-height: 75vh;
        overflow-y: auto;
    }

    /* Embedded "Try It Live" portal (Final Review page, see
       render_knowledge_portal() below) - same .cite/.refused/.passage-card
       treatment app/knowledge_portal.py uses on its own page, reused here so
       the finished-system answer looks identical wherever it's asked from. */
    .cite {background:#eef5f6; color:#13315c; border-left:4px solid #1f6f78; padding:.5rem .8rem; border-radius:6px; margin:.3rem 0;}
    .refused {background:#fbf6e7; color:#13315c; border-left:4px solid #b08900; padding:.8rem 1rem; border-radius:8px;}
    .passage-card {
        background: #ffffff;
        color: #13315c;
        border: 1px solid #e6e5df;
        border-radius: 14px;
        padding: .6rem .9rem;
        margin-bottom: .6rem;
        box-shadow: 0 2px 8px rgba(11, 37, 69, .05), 0 12px 30px rgba(11, 37, 69, .07);
    }
    /* A freshly-generated answer/citation block eases in instead of
       snapping into place - a small cue that this is a live result, not
       static page content. */
    .kp-result-enter {animation: kpResultFadeIn .45s ease-out;}
    @keyframes kpResultFadeIn {
        from {opacity: 0; transform: translateY(8px);}
        to {opacity: 1; transform: translateY(0);}
    }
    @media (prefers-reduced-motion: reduce) {
        .kp-result-enter {animation: none;}
    }

    /* "Try It Live" section banner: a distinct accent (teal-forward, not the
       navy hero-banner gradient) so it visually reads as "the finished
       product, live" rather than another lesson block. */
    .live-banner {
        background: linear-gradient(120deg, #1f6f78 0%, #13315c 60%, #0b2545 100%);
        background-size: 250% 250%;
        animation: heroShift 16s ease-in-out infinite;
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin: 2rem 0 1.25rem 0;
        box-shadow: 0 4px 18px rgba(11, 37, 69, 0.25);
    }
    @media (prefers-reduced-motion: reduce) {
        .live-banner {animation: none;}
    }
    .live-banner .hero-title {font-size: 1.4rem;}
    .pulse-dot {
        display: inline-block;
        width: 9px; height: 9px;
        border-radius: 50%;
        background: #5ee6b8;
        margin-right: .5rem;
        box-shadow: 0 0 0 rgba(94, 230, 184, 0.6);
        animation: pulseDot 1.8s ease-out infinite;
    }
    @keyframes pulseDot {
        0%   {box-shadow: 0 0 0 0 rgba(94, 230, 184, 0.55);}
        70%  {box-shadow: 0 0 0 8px rgba(94, 230, 184, 0);}
        100% {box-shadow: 0 0 0 0 rgba(94, 230, 184, 0);}
    }
    @media (prefers-reduced-motion: reduce) {
        .pulse-dot {animation: none;}
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# One simple pipeline diagram (Overview -> Solution Architecture tab). Pure
# inline SVG, no JS, so it renders the same in the sandboxed preview as
# anywhere else Streamlit renders it.
# ---------------------------------------------------------------------------

PIPELINE_DIAGRAM_SVG = """
<style>
    html, body {margin: 0; padding: 0; background: #ffffff;}
    .pd-node rect {fill: #eef3f6; stroke: #0b2545; stroke-width: 1.5; rx: 8;}
    .pd-node text {fill: #0b2545; font: 600 12px sans-serif; text-anchor: middle;}
    .pd-node .pd-sub {font: 400 10px sans-serif; fill: #4a5a6a;}
    .pd-line {stroke: #1f6f78; stroke-width: 2; fill: none;}
    .pd-section {font: 700 13px sans-serif; fill: #0b2545;}
</style>
<svg viewBox="0 0 900 260" style="width:100%; height:auto; max-width:860px; display:block; margin:0.5rem auto;">
    <defs>
        <marker id="pd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#1f6f78"/>
        </marker>
    </defs>
    <text class="pd-section" x="10" y="20">Phase 1: Ingestion &amp; Governance</text>
    <text class="pd-section" x="330" y="20">Phase 2: Index, Retrieve, Generate</text>
    <text class="pd-section" x="680" y="20">Phase 3: Operate</text>

    <path class="pd-line" marker-end="url(#pd-arrow)" d="M115,60 L129,60"/>
    <path class="pd-line" marker-end="url(#pd-arrow)" d="M245,60 L259,60"/>
    <path class="pd-line" marker-end="url(#pd-arrow)" d="M375,60 L389,60"/>
    <path class="pd-line" marker-end="url(#pd-arrow)" d="M505,60 L519,60"/>
    <path class="pd-line" marker-end="url(#pd-arrow)" d="M635,60 L649,60"/>
    <path class="pd-line" marker-end="url(#pd-arrow)" d="M765,60 L779,60"/>

    <g class="pd-node"><rect x="10" y="35" width="105" height="50"/><text x="62" y="56">Loaders</text><text class="pd-sub" x="62" y="72">1.2</text></g>
    <g class="pd-node"><rect x="130" y="35" width="115" height="50"/><text x="187" y="56">Metadata</text><text class="pd-sub" x="187" y="72">1.3</text></g>
    <g class="pd-node"><rect x="260" y="35" width="115" height="50"/><text x="317" y="56">Chunking</text><text class="pd-sub" x="317" y="72">2.1</text></g>
    <g class="pd-node"><rect x="390" y="35" width="115" height="50"/><text x="447" y="56">Vector Index</text><text class="pd-sub" x="447" y="72">2.1</text></g>
    <g class="pd-node"><rect x="520" y="35" width="115" height="50"/><text x="577" y="56">Hybrid Retrieval</text><text class="pd-sub" x="577" y="72">2.2</text></g>
    <g class="pd-node"><rect x="650" y="35" width="115" height="50"/><text x="707" y="56">Generation</text><text class="pd-sub" x="707" y="72">2.3</text></g>
    <g class="pd-node"><rect x="780" y="35" width="110" height="50"/><text x="835" y="56">Cited Answer</text></g>

    <path class="pd-line" marker-end="url(#pd-arrow)" d="M120,150 L204,150"/>
    <path class="pd-line" marker-end="url(#pd-arrow)" d="M355,150 L439,150"/>
    <g class="pd-node"><rect x="10" y="125" width="110" height="50"/><text x="65" y="146">Tracing</text><text class="pd-sub" x="65" y="162">3.1</text></g>
    <g class="pd-node"><rect x="205" y="125" width="150" height="50"/><text x="280" y="146">pytest smoke suite</text><text class="pd-sub" x="280" y="162">3.2</text></g>
    <g class="pd-node"><rect x="440" y="125" width="150" height="50"/><text x="515" y="146">Docker</text><text class="pd-sub" x="515" y="162">3.3</text></g>

    <text x="10" y="220" style="font: 400 11px sans-serif; fill: #4a5a6a;">Each stage lives in its own src/ module. app/knowledge_portal.py (finished product) calls src.pipeline.KnowledgeSystem and never reimplements a stage.</text>
</svg>
"""


# ---------------------------------------------------------------------------
# Execution + file helpers
# ---------------------------------------------------------------------------

# reference_solution/src is a *partial* mirror: it contains only the TODO
# modules for the 6 packages this course's activities implement (ingestion,
# chunking, indexing, retrieval, generation, observability) - no config/,
# no data/, no src/utils/, no src/llm/. A plain
# `sys.path.insert(0, "reference_solution")` would shadow the whole `src`
# package and break every import of src.utils/src.llm/src.pipeline.
#
# Build a merged overlay directory instead: a full copy of the real src/,
# config/, and data/, with only reference_solution's own files copied on top
# of src/. This is the same mechanism scripts/verify_reference.py already
# uses (see its REFERENCE_MODULES list) to independently verify the
# reference solution; reused here unchanged so the two never drift apart.
#
# Nothing here is cached across calls: src/, config/, and data/ are all
# always re-copied fresh on every "See reference solution run" click. An
# earlier version of this code cached src/ (comment used to read "src/utils,
# src/llm, and the corpus never change") on the theory that only
# config/*.yaml and data/ get edited by a learner. That assumption broke
# once a later refactor started threading an explicit base_path parameter
# through src/utils/config_loader.py: a pre-refactor cached copy of src/
# under %TEMP%\\bc4_workbook_reference_merge then caused a live
# `ImportError: cannot import name 'resolve_path' from
# 'src.utils.config_loader'` the next time "See reference solution run" was
# clicked, because the stale cached src/utils/config_loader.py predated
# resolve_path() while reference_solution/'s overlay files (freshly copied
# each time) already called it. src/ is small (this project's whole src/
# tree, well under what a copytree takes noticeable time for), so the
# always-fresh cost here is negligible next to the risk of silently running
# a stale src/utils or src/llm against a newer reference_solution/ overlay.
REFERENCE_MODULES = [
    "ingestion/loaders.py", "ingestion/document_processor.py", "chunking/chunker.py",
    "indexing/vector_store.py", "retrieval/retriever.py", "generation/answer_generator.py",
    "observability/tracing.py",
]

REFERENCE_PATH_PREFIX = """import sys as _sys, os as _os, shutil as _shutil, tempfile as _tempfile
_merge_root = _os.path.join(_tempfile.gettempdir(), "bc4_workbook_reference_merge")
_merge_src = _os.path.join(_merge_root, "src")
# src/, config/, and data/ are all always re-copied fresh on every call (see
# the comment above REFERENCE_MODULES for why src/ is no longer cached).
if _os.path.isdir(_merge_src):
    _shutil.rmtree(_merge_src)
_shutil.copytree("src", _merge_src)
for _name in ("config", "data"):
    _dst = _os.path.join(_merge_root, _name)
    if _os.path.isdir(_dst):
        _shutil.rmtree(_dst)
    _shutil.copytree(_name, _dst)
for _rel in [
    "ingestion/loaders.py", "ingestion/document_processor.py", "chunking/chunker.py",
    "indexing/vector_store.py", "retrieval/retriever.py", "generation/answer_generator.py",
    "observability/tracing.py",
]:
    _shutil.copy2(_os.path.join("reference_solution", "src", _rel), _os.path.join(_merge_src, _rel))
_sys.path.insert(0, _merge_root)
_os.chdir(_merge_root)
"""


def run_snippet(code: str, timeout: int = 180, use_reference: bool = False) -> dict:
    """Run `code` in a fresh subprocess using this same interpreter, from the
    project root, so every run picks up exactly what's saved on disk right
    now, no module-cache staleness possible.

    With use_reference=True, the process chdirs into a merged overlay
    directory instead (real src/, config/, and data/ plus
    reference_solution's own files on top of src/, see
    REFERENCE_PATH_PREFIX), so every `from src...` import resolves to the
    completed reference implementation for the modules this lab's activities
    implement, and to the real, already-complete src/ for everything else
    (src/utils, src/llm, src/pipeline) - letting a stuck learner see the
    completed solution actually run, not just read its code.
    """
    code_to_run = (REFERENCE_PATH_PREFIX + code) if use_reference else code
    env = os.environ.copy()
    # Windows defaults a child process's stdout/stderr to the system codepage
    # (cp1252), which crashes on non-ASCII output some activity scripts
    # print. Force UTF-8 for the subprocess itself.
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code_to_run],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode, "timed_out": False}
    except subprocess.TimeoutExpired as e:
        return {"stdout": e.stdout or "", "stderr": e.stderr or "", "returncode": None, "timed_out": True}


def classify_run(result: dict) -> str:
    if result["timed_out"]:
        return "error"
    if result["returncode"] != 0:
        return "error"
    combined = result["stdout"] + result["stderr"]
    if "NotImplementedError" in combined or "[WARN]" in combined:
        return "warn"
    return "pass"


def read_file(rel_path: str) -> str:
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        return f"# File not found: {rel_path}"
    return path.read_text(encoding="utf-8-sig")


if "activity_status" not in st.session_state:
    st.session_state.activity_status = {}
if "activity_output" not in st.session_state:
    st.session_state.activity_output = {}
if "activity_status_mtime" not in st.session_state:
    # Modification time(s) of an activity's `files`, recorded at the moment
    # its status was last set by a "Run Activity" click - lets the progress
    # tracker detect a file edited after the run that produced the current
    # "pass" (see activity_run_is_stale() below).
    st.session_state.activity_status_mtime = {}


# ---------------------------------------------------------------------------
# Activity content: 9 activities across 3 phases (the whole lab)
# ---------------------------------------------------------------------------

ACTIVITIES = [
    dict(
        id="1.1", phase=1, title="Explore the Enterprise Knowledge System",
        module="*No module to implement: orientation activity*",
        business_context="""A retail bank has a working RAG proof of concept that answers questions from
**one** policy document. It can't go enterprise-wide. It can't ingest multiple formats, it keeps no document
context, it has no departmental filtering, it gives you no traceability, and it has no deployment path. In
this lab you take that POC and, step by step, turn it into a production-ready Enterprise Knowledge System
for Retail Banking, Loans, Credit Cards, Operations, and Compliance.""",
        objective="Find every module you'll implement in this lab, and confirm the corpus, the manifest, and all six stage modules are in place and import cleanly.",
        instructions="""1. Open `src/pipeline.py` and read its docstring. It wires the six stages (ingest -> process -> chunk -> index -> retrieve -> generate) into two entry points, `build_index()` and `answer_question()`. You'll never edit this file. Each activity below fills in one stage it calls.
2. Skim `data/knowledge_base/manifest.csv`: 18 governed documents across 5 departments (retail_banking, loans, cards, operations, compliance), each with department/doc_type/product/version/effective_date/owner. Notice that `data/knowledge_base/operations/draft_notes.md` exists as a file but has **no** manifest row. That's the deliberate ungoverned document Activity 1.3 must exclude.
3. Open `app/knowledge_portal.py`, the finished-product UI. It only calls `src.pipeline.KnowledgeSystem`. Every stage's real logic lives in `src/`.
4. Scroll down and click **Run Activity 1.1** below to confirm the manifest loads and all six stage modules import without error.""",
        ai_assist="**AI Assist:** Ask your AI coding assistant to trace through `src/pipeline.py`'s `KnowledgeSystem.__init__` and `answer_question()` and name, for each line, which activity's TODO function it calls. It's good orientation before you touch any code.",
        files=[],
        target_methods={},
        reference_only_files=["config/ingestion_config.yaml", "config/llm_config.yaml", "config/prompts.yaml"],
        has_reference_run=False,
        milestone="You can explain how the pipeline fits together and find every module you'll implement.",
        validation="""Confirm that:
- The manifest loads with 18 entries
- `data/knowledge_base/operations/draft_notes.md` exists as a file but has no manifest row (you'll see this become the ungoverned-exclusion case in Activity 1.3)
- All six stage modules (`ingestion`, `chunking`, `indexing`, `retrieval`, `generation`, `observability`) import without error""",
        pass_label="[OK] Orientation check complete: nothing to implement here yet",
        script="""from pathlib import Path

from src.ingestion.document_processor import load_manifest
from src.ingestion import loaders  # noqa: F401
from src.chunking import chunker  # noqa: F401
from src.indexing import vector_store  # noqa: F401
from src.retrieval import retriever  # noqa: F401
from src.generation import answer_generator  # noqa: F401
from src.observability import tracing  # noqa: F401

manifest = load_manifest("data/knowledge_base/manifest.csv")
print(f"Manifest loaded: {len(manifest)} governed document(s).")
sample_key = sorted(manifest.keys())[0]
print(f"Sample manifest entry: {sample_key!r} -> {manifest[sample_key]}")

draft = Path("data/knowledge_base/operations/draft_notes.md")
print(f"\\noperations/draft_notes.md exists on disk: {draft.exists()}")
print(f"operations/draft_notes.md has a manifest row: {'operations/draft_notes.md' in manifest}")
if draft.exists() and "operations/draft_notes.md" not in manifest:
    print("[OK] Confirmed: an ungoverned file exists on disk with no manifest row (Activity 1.3's exclusion case).")

print("\\n[OK] All 6 pipeline stage modules imported successfully: "
      "ingestion, chunking, indexing, retrieval, generation, observability")
""",
    ),
    dict(
        id="1.2", phase=1, title="Multi-Source Ingestion",
        module="`src/ingestion/loaders.py`",
        business_context="""The POC could only read a single .txt/.md file. A real enterprise knowledge base
is spread across PDFs (regulatory circulars, product sheets), Word documents (policies, SOPs), HTML
(intranet pages), and Markdown. So the first thing a production pipeline needs is one loader per format,
each turning its files into plain text.""",
        objective="Implement the four per-format text extractors so every document in the corpus loads as plain text, whatever its original format.",
        instructions="""In `src/ingestion/loaders.py`:
1. `_load_markdown(path)` (walk through together, Beginner): read `path` as UTF-8 text and return it stripped. Chunking later on is Markdown-aware, so keep the raw Markdown as it is.
2. `_load_pdf(path)` (walk through together, Beginner): use `from pypdf import PdfReader`. Iterate `reader.pages`, call `page.extract_text()` on each (treat `None` as `""`), join with `"\\n\\n"`, return stripped.
3. `_load_docx(path)` (your turn, Beginner): use `import docx`. Collect `para.text` for every non-empty paragraph in `docx.Document(str(path)).paragraphs`, join with `"\\n"`, return stripped.
4. `_load_html(path)` (your turn, Beginner): use `from bs4 import BeautifulSoup`. Parse with the `"lxml"` parser, remove every `<script>`/`<style>` tag, call `soup.get_text(separator="\\n")`, then collapse three-or-more consecutive newlines to two with `re.sub(r"\\n\\s*\\n\\s*\\n+", "\\n\\n", text)` (`re` is already imported at the top of the file), and return the result stripped.

The dispatcher `load_document()` and the corpus walker `load_corpus()` are already written for you.""",
        ai_assist="**AI Assist:** Ask your AI coding assistant why `_load_pdf` and `_load_docx` share the same \"iterate items, join, strip\" shape, and what makes `_load_html` different (tag removal before text extraction, not just iteration).",
        files=["src/ingestion/loaders.py"],
        target_methods={"src/ingestion/loaders.py": ["_load_markdown", "_load_pdf", "_load_docx", "_load_html"]},
        milestone="All 4 formats (PDF, DOCX, HTML, Markdown) load without error.",
        validation="""Confirm that:
- Each of the 4 sample files loads with non-empty content
- Try a file of each format from a different department (e.g. `cards/credit_card_operating_manual.pdf`) to make sure the extractor works beyond the one sample file""",
        script="""from pathlib import Path

from src.ingestion.loaders import load_document

kb_root = Path("data/knowledge_base")
files = {
    "markdown": "retail_banking/savings_account_policy.md",
    "pdf": "loans/home_loan_policy.pdf",
    "docx": "loans/personal_loan_guidelines.docx",
    "html": "loans/auto_loan_product_sheet.html",
}

try:
    empty = []
    for fmt, rel in files.items():
        doc = load_document(kb_root / rel, kb_root=kb_root)
        preview = doc.content[:90].replace("\\n", " ")
        print(f"[{fmt:8}] {doc.source} -> {len(doc.content)} chars. Preview: {preview!r}")
        if not doc.content.strip():
            empty.append(rel)
    if empty:
        print(f"\\n[WARN] These file(s) loaded but produced empty content: {empty}")
    else:
        print("\\n[OK] All 4 formats (markdown, pdf, docx, html) loaded non-empty text via load_document().")
except NotImplementedError as e:
    print(f"[WARN] Not implemented yet: {e}")
    print("Complete src/ingestion/loaders.py first.")
""",
    ),
    dict(
        id="1.3", phase=1, title="Document Processing & Enterprise Metadata",
        module="`src/ingestion/document_processor.py`",
        business_context="""Retrieval quality in an enterprise knowledge base depends on more than
semantic similarity: a compliance officer asking about card chargebacks should get the *Cards* department's
current procedure, not a superseded draft from Operations. That's what metadata is for. Every chunk
carries department, doc_type, product, version, and effective_date, so the retriever (Activity 2.2) can
filter on them.""",
        objective="Implement build_metadata() and process_document() so every governed document gets its enterprise metadata, and an ungoverned one is kept out instead of being quietly indexed.",
        instructions="""In `src/ingestion/document_processor.py`:
1. `build_metadata(raw, manifest_row)` (steps 1-2 walked through together, steps 3-4 your turn):
   - Read the required field names with exactly this call - no `base_path` argument here, unlike `load_manifest()` above it in the same file: `build_metadata()` doesn't take a `base_path` parameter, so `base_path=base_path` would reference a name that doesn't exist in this function's scope: `load_yaml_config("config/ingestion_config.yaml")["metadata"]["required_fields"]` (department, doc_type, product, version, effective_date, source).
   - Build a dict: department/doc_type/product/version/effective_date from `manifest_row`, plus `source=raw.source`, `file_type=raw.file_type`, `title=manifest_row.get("title") or raw.source`, and copy `owner` if present.
   - Validate every required field is present and non-empty; raise `ValueError` naming the offending field(s) and `raw.source` if not.
   - Return the dict.
2. `process_document(raw, manifest)` (your turn, and this is where governance actually gets enforced):
   - Look up `manifest[_normalize_key(raw.source)]`. If missing, raise `KeyError` naming `raw.source` (an ungoverned document must not enter the index).
   - Call `build_metadata(raw, row)`.
   - Return `ProcessedDocument(content=raw.content, metadata=<that dict>)`.

`load_manifest()` and `process_corpus()` are already written for you. `process_corpus()` catches `KeyError`/`ValueError` per document, so one ungoverned file doesn't stop the whole ingestion run.""",
        ai_assist="**AI Assist:** Ask your AI coding assistant why `process_corpus()` catches the exception per-document instead of letting one bad document abort the whole corpus build. What would the alternative cost you in operations?",
        files=["src/ingestion/document_processor.py"],
        target_methods={"src/ingestion/document_processor.py": ["build_metadata", "process_document"]},
        milestone="A governed document is enriched with all required metadata; an ungoverned document is excluded, not indexed.",
        validation="""Confirm that:
- A real governed document (e.g. `retail_banking/savings_account_policy.md`) processes with all 6 required metadata fields non-empty
- A document with no manifest row (`operations/draft_notes.md`) raises `KeyError` from `process_document()` rather than being silently processed""",
        script="""from src.ingestion.document_processor import load_manifest, process_document
from src.ingestion.loaders import RawDocument, load_document

kb_root = "data/knowledge_base"
manifest = load_manifest(f"{kb_root}/manifest.csv")

try:
    raw = load_document(f"{kb_root}/retail_banking/savings_account_policy.md", kb_root=kb_root)
except NotImplementedError:
    # Keep this activity self-contained even if Activity 1.2 isn't done yet.
    raw = RawDocument(content="(placeholder content - Activity 1.2 not complete)",
                      source="retail_banking/savings_account_policy.md", file_type="md")

try:
    processed = process_document(raw, manifest)
    print(f"Processed {processed.metadata.get('source')}:")
    for k, v in processed.metadata.items():
        print(f"  {k}: {v}")

    required = ["department", "doc_type", "product", "version", "effective_date", "source"]
    missing = [f for f in required if not processed.metadata.get(f)]
    if missing:
        print(f"\\n[WARN] Missing/empty required metadata field(s): {missing}")
    else:
        print("\\n[OK] Metadata carries all 6 required governance fields.")

    rogue = RawDocument(content="unmanaged draft", source="operations/draft_notes.md", file_type="md")
    try:
        process_document(rogue, manifest)
        print("[WARN] process_document() should have raised KeyError for operations/draft_notes.md (no manifest row)")
    except KeyError:
        print("[OK] Ungoverned document (operations/draft_notes.md) correctly raised KeyError and is excluded.")
except NotImplementedError as e:
    print(f"[WARN] Not implemented yet: {e}")
    print("Complete src/ingestion/document_processor.py first.")
""",
    ),
    dict(
        id="2.1", phase=2, title="Chunking + Vector Store Indexing",
        module="`src/chunking/chunker.py`, `src/indexing/vector_store.py`",
        business_context="""The POC embedded whole documents. That wrecks retrieval: a 1,500-word policy
returned as one hit buries the two relevant sentences in noise, and it eats the model's context budget.
Production RAG splits every document into small, overlapping passages, embeds them, and saves them in a
real vector store. At query time the store returns similarity scores and can filter by metadata.""",
        concept_overview="""This course puts chunking and vector-store indexing under one activity number
(2.1), because they're two halves of the same "make a document searchable" step. Split first
(`chunker.split_text`), then embed and save what you split (`vector_store.build_documents` /
`index_chunks` / `search`). The activity table in `README.md` has the official numbering.""",
        objective="Implement the three chunking strategies. Then implement the functions that turn chunks into Documents, embed and save them, and search them in a real vector store.",
        instructions="""In `src/chunking/chunker.py`:
1. `split_text(text, strategy, chunk_size, chunk_overlap)`: all three strategies come from `langchain_text_splitters`. Call `.split_text(text)` and return the result:
   - `"recursive"` (walk through together): `RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=...)`: the sensible default. It splits on paragraph, line, and word boundaries.
   - the `else: raise ValueError(...)` branch for an unknown strategy (walk through together).
   - `"fixed"` (your turn): `CharacterTextSplitter(separator="", chunk_size=..., chunk_overlap=...)`: a blunt fixed-width cut.
   - `"markdown"` (your turn): `MarkdownTextSplitter(chunk_size=..., chunk_overlap=...)`: keeps Markdown headings and sections intact.

In `src/indexing/vector_store.py` (all your turn):
2. `build_documents(chunks)`: `from langchain_core.documents import Document`. For each chunk, build `Document(page_content=chunk.content, metadata=chunk.metadata)`, coercing every metadata value to str/int/float/bool (Chroma rejects non-scalar values). Return the list.
3. `index_chunks(chunks, backend=None)`: resolve the backend via `_backend_name(backend)`, call `build_documents(chunks)`, build stable `ids` from `chunk.metadata["chunk_id"]` (so a re-run upserts instead of duplicating), call `_make_store(resolved, documents=documents, ids=ids)`, log, and return the store.
4. `search(store, query, top_k=5, metadata_filter=None)`: call `store.similarity_search_with_relevance_scores(query, k=top_k, filter=metadata_filter)`, wrap each `(Document, score)` pair as a `RetrievedChunk`, and return the list.""",
        ai_assist="**AI Assist:** Ask your AI coding assistant why `index_chunks` uses each chunk's stable `chunk_id` as the vector store's document id instead of letting the backend auto-generate one. What would re-running the index build do differently without that?",
        files=["src/chunking/chunker.py", "src/indexing/vector_store.py"],
        target_methods={
            "src/chunking/chunker.py": ["split_text"],
            "src/indexing/vector_store.py": ["build_documents", "index_chunks", "search"],
        },
        milestone="split_text() produces overlapping chunks for all 3 strategies, and a real vector store search returns ranked results for an indexed chunk.",
        validation="""Confirm that:
- All 3 strategies (`recursive`, `fixed`, `markdown`) return more than one chunk for a long input
- `build_documents()` produces `Document` objects whose metadata values are all scalar (str/int/float/bool)
- `index_chunks()` + `search()` round-trip: a real embed-and-query against the vector store returns the indexed chunk (this step needs a working `OPENAI_API_KEY` for the embeddings call. If it's missing or invalid you'll see a printed [WARN]. That's expected without a real key and isn't a bug in your code)

**Troubleshooting note:** if you see an `[INFO]` message saying Chroma is unavailable and the run is falling back to FAISS, that's expected on some machines. Chroma's native Rust binding (`chromadb_rust_bindings`) fails to load in certain environments (a missing DLL on Windows, a missing/incompatible shared library on Linux). You don't need to fix it. The activity still works correctly on FAISS.""",
        script="""from src.chunking.chunker import Chunk, split_text

sample_text = ("Minimum balance for a metro savings account is INR 5,000. "
              "Non-metro accounts require INR 2,500. Failing to maintain the "
              "minimum balance attracts a penalty of INR 300 per quarter. ") * 15

try:
    for strategy in ("recursive", "fixed", "markdown"):
        pieces = split_text(sample_text, strategy, chunk_size=200, chunk_overlap=40)
        print(f"[{strategy:10}] {len(pieces)} chunk(s). First: {pieces[0][:60]!r}")
        if len(pieces) < 2:
            print(f"[WARN] Expected multiple chunks for strategy={strategy!r}, got {len(pieces)}")
    print("\\n[OK] split_text() produced overlapping chunks for all 3 strategies.")
except NotImplementedError as e:
    print(f"[WARN] Not implemented yet: {e}")
    print("Complete split_text() in src/chunking/chunker.py first.")

from src.indexing.vector_store import build_documents, index_chunks, search  # noqa: E402

chunks = [
    Chunk(content="Minimum balance for a metro savings account is INR 5,000.",
          metadata={"source": "retail_banking/savings_account_policy.md",
                    "chunk_id": "retail_banking/savings_account_policy.md::0",
                    "department": "retail_banking", "version": "2.1"}),
    Chunk(content="Failing to maintain the minimum balance attracts a penalty of INR 300 per quarter.",
          metadata={"source": "retail_banking/savings_account_policy.md",
                    "chunk_id": "retail_banking/savings_account_policy.md::1",
                    "department": "retail_banking", "version": "2.1"}),
]

try:
    documents = build_documents(chunks)
    print(f"\\nbuild_documents(): {len(documents)} LangChain Document(s)")
    bad_meta = [(k, v) for d in documents for k, v in d.metadata.items()
                if not isinstance(v, (str, int, float, bool))]
    if bad_meta:
        print(f"[WARN] Non-scalar metadata value(s) found: {bad_meta}")
    else:
        print("[OK] build_documents() produced Documents with scalar-only metadata (Chroma-safe).")
except NotImplementedError as e:
    print(f"[WARN] Not implemented yet: {e}")
    print("Complete build_documents() in src/indexing/vector_store.py first.")
except Exception as e:
    print(f"[WARN] build_documents() raised an unexpected error: {type(e).__name__}: {e}")

try:
    import tempfile
    import os as _os
    import shutil as _shutil
    _vs_dir = tempfile.mkdtemp(prefix="workbook_activity_2_1_")
    _os.environ["VECTOR_STORE_DIR"] = _vs_dir
    try:
        try:
            store = index_chunks(chunks, backend="chroma")
        except NotImplementedError:
            raise
        except (ImportError, OSError) as chroma_err:
            # Chroma ships a native rust binding (chromadb_rust_bindings)
            # that fails to *load* (not run) on some machines: a "DLL load
            # failed while importing chromadb_rust_bindings" ImportError on
            # Windows, or an OSError from the dynamic loader on other
            # platforms. This is the same
            # signature scripts/verify_reference.py's own live check already
            # works around. Only these two exception types genuinely mean
            # "the backend itself failed to load" - anything else (TypeError,
            # KeyError, AttributeError, an assertion, ...) is a real bug in
            # the learner's own index_chunks()/build_documents()/search()
            # and must propagate uncaught rather than being silently retried
            # on FAISS (the same buggy code) and re-labeled as an
            # environment issue below.
            print(f"[INFO] chroma backend unavailable here ({type(chroma_err).__name__}: {str(chroma_err)[:150]}); using faiss instead")
            store = index_chunks(chunks, backend="faiss")
        hits = search(store, "minimum balance metro savings account", top_k=2)
        print(f"\\nindex_chunks() + search(): {len(hits)} hit(s)")
        for h in hits:
            print(f"  {h.chunk_id}  score={h.score:.3f}")
        if hits:
            print("[OK] index_chunks()/search() round-tripped a real embed-and-query against the vector store.")
        else:
            print("[WARN] search() returned no hits for a query that should match an indexed chunk.")
    finally:
        # Always clean up the temp vector store dir (success, warn, or
        # error) - the store has already been fully used (search() above
        # already completed) by the time this runs, so there's no race.
        _shutil.rmtree(_vs_dir, ignore_errors=True)
except NotImplementedError as e:
    print(f"[WARN] Not implemented yet: {e}")
    print("Complete index_chunks()/search() in src/indexing/vector_store.py first.")
except Exception as e:
    import openai as _openai
    if isinstance(e, _openai.OpenAIError):
        # A genuine embeddings-API problem (missing/invalid OPENAI_API_KEY,
        # rate limit, connection failure, ...): an environment issue, not
        # necessarily a code issue.
        print(f"[WARN] index_chunks()/search() needs a working OPENAI_API_KEY for the embeddings call "
              f"(this is an environment issue, not necessarily a code issue): {type(e).__name__}: {e}")
    else:
        # Anything else is a real bug in the learner's own code - let it
        # propagate with its full traceback instead of mislabeling it as an
        # API-key problem.
        raise
""",
    ),
    dict(
        id="2.2", phase=2, title="Retrieval Optimization: Hybrid Search",
        module="`src/retrieval/retriever.py`",
        business_context="""Pure vector search misses exact terms ("FOIR", "V-CIP", a circular number) and
can't tell a current policy from a superseded one. Production retrieval fixes both. It fuses a lexical (BM25)
signal with the vector signal, and it adds a metadata filter, so a Loans question is answered from current
Loans documents, not a superseded 2024 draft.""",
        concept_overview="""One open question this activity doesn't settle: chunk overlap (Activity 2.1) means
neighboring chunks from the same document share text, so two of the top-k results can be near-duplicates.
That hurts answer quality, and filtering and fusion alone don't fix it. We don't fix it here.""",
        objective="Implement HybridRetriever.retrieve() so a query returns ranked chunks that combine a vector-similarity signal with a BM25 lexical signal, both scoped by an optional department filter.",
        instructions="""In `src/retrieval/retriever.py`, `HybridRetriever.retrieve()`:

Walk through together (steps 1-2):
1. Resolve settings from `self.config`: `top_k` (arg or `config["top_k"]`), `score_threshold`, and the `hybrid` block (`config["hybrid"]` with `enabled` / `vector_weight` / `lexical_weight`). The `hybrid` argument overrides `config["hybrid"]["enabled"]` when not `None`.
2. Get vector hits: `self._vector_scores(query, top_k * 4, metadata_filter)` (over-fetch so fusion has candidates). Build `vec_raw = {chunk_id: hit.score}`.

Your turn (steps 3-5):
3. If hybrid is off: normalize `vec_raw`, attach the normalized score to each `RetrievedChunk`, drop anything below `score_threshold`, sort descending, return the top_k.
4. If hybrid is on: also get `lex_raw = self._bm25_scores(query, metadata_filter)`. Min-max normalize `vec_raw` and `lex_raw` separately (`_minmax_normalize`, already provided). For the union of chunk_ids, `fused = vector_weight * nvec.get(id, 0) + lexical_weight * nlex.get(id, 0)`. For an id present only in BM25, materialize a `RetrievedChunk` from `self._by_id[id]`.
5. Set each `RetrievedChunk.score` to its fused score, drop below `score_threshold`, sort descending, return the top_k.

The BM25 index, the vector delegate (`_vector_scores`/`_bm25_scores`), and score normalization are already written for you.""",
        ai_assist="**AI Assist:** Ask your AI coding assistant to trace through what happens to a chunk_id that scores high on BM25 but was never returned by the vector search's top_k*4 candidates. Does it still make it into the fused results, and why does that matter for a rare exact term like a circular number?",
        files=["src/retrieval/retriever.py"],
        target_methods={"src/retrieval/retriever.py": ["retrieve"]},
        milestone="HybridRetriever.retrieve() returns ranked chunks with the department metadata filter respected.",
        validation="""Confirm that:
- `retrieve()` returns a non-empty, score-descending list for an in-corpus question
- Every returned chunk's `metadata["department"]` matches the filter passed in (no leakage across departments)
- This check uses a lightweight in-memory stand-in store on purpose (real corpus chunks, lexical-overlap scoring instead of a real embeddings call). It runs the same with or without a configured `OPENAI_API_KEY`. `HybridRetriever.retrieve()` can't tell the difference, since it only ever calls `store.similarity_search_with_relevance_scores(...)`""",
        script="""import json

from src.chunking.chunker import chunk_corpus
from src.ingestion.document_processor import load_manifest, process_corpus
from src.ingestion.loaders import load_corpus
from src.retrieval.retriever import HybridRetriever

scenarios = json.load(open("data/validation/test_scenarios.json", encoding="utf-8"))["scenarios"]
scn02 = next(s for s in scenarios if s["id"] == "SCN-02")
question = scn02["question"]  # only ever surface the question field - not expected_sources/must_include/note

try:
    raws = load_corpus("data/knowledge_base")
    manifest = load_manifest("data/knowledge_base/manifest.csv")
    processed = process_corpus(raws, manifest)
    chunks = chunk_corpus(processed)
    loans_chunks = [c for c in chunks if c.metadata.get("department") == "loans"]
    print(f"Loaded {len(loans_chunks)} 'loans' department chunk(s) for retrieval testing.")
except NotImplementedError as e:
    print(f"[WARN] Cannot proceed: an earlier activity (1.2/1.3/2.1) isn't complete yet ({e})")
    loans_chunks = None


class _StubDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class _StubStore:
    \"\"\"A lexical-overlap stand-in for a real vector store, so this
    activity's fusion logic can be checked without an embeddings API call.
    HybridRetriever.retrieve() only ever calls
    store.similarity_search_with_relevance_scores(...); it can't tell this
    apart from a real backend.\"\"\"

    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search_with_relevance_scores(self, query, k=5, filter=None):
        q_terms = set(query.lower().split())
        scored = []
        for c in self.chunks:
            if filter and any(c.metadata.get(key) != val for key, val in filter.items()):
                continue
            overlap = len(q_terms & set(c.content.lower().split()))
            scored.append((_StubDoc(c.content, c.metadata), overlap / (len(q_terms) or 1)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


if loans_chunks:
    try:
        retriever = HybridRetriever(_StubStore(loans_chunks), loans_chunks)
        hits = retriever.retrieve(question, metadata_filter={"department": "loans"})
        print(f"\\nQuestion: {question}")
        print(f"retrieve() returned {len(hits)} hit(s):")
        for h in hits:
            print(f"  {h.source}  score={h.score:.3f}")

        if not hits:
            print("[WARN] retrieve() returned no hits for an in-corpus question")
        elif any(h.metadata.get("department") != "loans" for h in hits):
            print("[WARN] metadata_filter leaked a non-'loans' chunk into the results")
        else:
            print("\\n[OK] HybridRetriever.retrieve() fused vector + BM25 signals and respected the department filter.")
    except NotImplementedError as e:
        print(f"[WARN] Not implemented yet: {e}")
        print("Complete HybridRetriever.retrieve() in src/retrieval/retriever.py first.")
""",
    ),
    dict(
        id="2.3", phase=2, title="Grounded Response Generation with Citations",
        module="`src/generation/answer_generator.py`",
        business_context="""The POC pasted the retrieved text into the prompt and returned whatever the
model said. You couldn't tell which sentence came from which document, and nothing stopped the model from
answering off its own training data. A production answer is grounded (context only), cited (every claim
points to a passage), and refuses when the context doesn't cover the question.""",
        objective="Implement generate_answer() so it returns a cited, grounded answer from the retrieved chunks, or a clean refusal when nothing supports one.",
        instructions="""In `src/generation/answer_generator.py`, `generate_answer()`:

Walk through together (steps 1-2):
1. `cfg = _generation_config()`. Keep at most `cfg["max_context_chunks"]` chunks (they arrive best-first).
2. If there are no chunks and `cfg["refuse_when_no_context"]` is true, return immediately: `{"answer": REFUSAL_TEXT, "citations": [], "retrieved_chunks": [], "refused": True}`.

Your turn (steps 3-6):
3. `context = build_context(chunks)`. Load the `"grounded_answer_prompt"` via `PromptManager`; format its `user_template` with `question=question, context=context`. Call `(llm or LLMClient()).generate([...system..., ...user...])`.
4. `answer` = the model text, stripped. `refused` = answer starts with `REFUSAL_TEXT` (case-insensitive) OR contains no `[n]` marker.
5. `citations = []` if refused else `parse_citations(answer, chunks)`.
6. Return `{"answer": answer, "citations": citations, "retrieved_chunks": [{"source":..., "version":..., "chunk_id":..., "score": ch.score} for ch in chunks], "refused": refused}`.

`build_context()`, `parse_citations()`, and `validate_answer()` are already written for you.""",
        ai_assist="**AI Assist:** Ask your AI coding assistant why refusal is detected two ways (the literal `REFUSAL_TEXT` prefix OR the absence of any `[n]` marker) instead of relying on the model to always use the exact refusal string.",
        files=["src/generation/answer_generator.py"],
        target_methods={"src/generation/answer_generator.py": ["generate_answer"]},
        milestone="generate_answer() returns a cited answer for an in-corpus question, grounded only in the passages it was given.",
        validation="""Confirm that:
- `result["refused"]` is `False` and `result["citations"]` is non-empty for the in-corpus question below
- `validate_answer(result)` returns `None` (no structural problems)
- This step calls the real chat completion API (`LLMClient.generate()`), so it needs a working `OPENAI_API_KEY`. A missing or invalid key shows up as a `[WARN]` in the output below instead of crashing the app. That's expected on a lab machine without a live key and isn't a bug in your code""",
        script="""import json
from pathlib import Path

from src.generation.answer_generator import generate_answer, validate_answer
from src.indexing.vector_store import RetrievedChunk
from src.ingestion.loaders import load_document

scenarios = json.load(open("data/validation/test_scenarios.json", encoding="utf-8"))["scenarios"]
scn01 = next(s for s in scenarios if s["id"] == "SCN-01")
question = scn01["question"]  # only ever surface the question field

kb_root = "data/knowledge_base"
rel_path = "retail_banking/savings_account_policy.md"
try:
    content = load_document(f"{kb_root}/{rel_path}", kb_root=kb_root).content
except NotImplementedError:
    # Keep this activity self-contained even if Activity 1.2 isn't done yet.
    content = Path(f"{kb_root}/{rel_path}").read_text(encoding="utf-8")

retrieved = [RetrievedChunk(
    content=content,
    metadata={"source": rel_path, "version": "2.1", "chunk_id": f"{rel_path}::0"},
    score=0.91,
)]

try:
    result = generate_answer(question, retrieved)
    print(f"Question: {question}")
    print(f"Refused: {result.get('refused')}")
    print(f"Answer: {result.get('answer')}")
    print(f"Citations: {result.get('citations')}")

    errors = validate_answer(result)
    if errors:
        print(f"\\n[WARN] validate_answer() flagged: {errors}")
    else:
        print("\\n[OK] generate_answer() produced a validated, cited answer.")
except NotImplementedError as e:
    print(f"[WARN] Not implemented yet: {e}")
    print("Complete generate_answer() in src/generation/answer_generator.py first.")
except Exception as e:
    print(f"[WARN] generate_answer() needs a working OPENAI_API_KEY for the chat completion call "
          f"(this is an environment issue, not necessarily a code issue): {type(e).__name__}: {e}")
""",
    ),
    dict(
        id="3.1", phase=3, title="Observability with LangSmith",
        module="`src/observability/tracing.py`",
        business_context="""Engineers running a production knowledge system need to see what a query actually
did: which chunks were retrieved, what the prompt was, how long generation took. They shouldn't have to
grep logs for it. LangSmith gives you that once one entry point is wrapped in tracing.""",
        concept_overview="""Tracing with LangSmith here takes two environment variables (`LANGCHAIN_TRACING_V2`,
`LANGCHAIN_API_KEY`) plus wrapping **one** orchestration entry point. No custom dashboards or evaluation
datasets. `configure_langsmith()` (already written) reads those two variables. Your job is the wrapping.
Without a configured key, the activity still runs and prints a local step-by-step trace to stdout (through
the prefilled `_run_with_local_trace()` helper), so you always get a visible trace. LangSmith is optional,
and you don't need it to finish this lab.""",
        objective="Implement run_query_with_tracing() so a full query runs inside tracing: LangSmith when it's configured, a local trace print when it isn't.",
        instructions="""In `src/observability/tracing.py`, `run_query_with_tracing()`:

Walk through together (step 1):
1. If `configure_langsmith()` is `False`, return `_run_with_local_trace(run_fn, *args, **kwargs)`. Tracing being off must never break the query.

Your turn (step 2):
2. Otherwise wrap `run_fn` with `langsmith.traceable(run_type="chain", name="enterprise_kb_query")` and call the wrapped function with the same args. Return its result (the wrapped call's result, not the original's).""",
        ai_assist="**AI Assist:** Ask your AI coding assistant what `langsmith.traceable()` actually does to the wrapped function's call, and why it matters that you return the *wrapped* callable's result, not the original's.",
        files=["src/observability/tracing.py"],
        target_methods={"src/observability/tracing.py": ["run_query_with_tracing"]},
        milestone="A query runs through run_query_with_tracing() and produces a visible trace, either LangSmith or the local fallback.",
        validation="""Confirm that:
- The run prints a result either way (traced or untraced): tracing being off must never break the query
- Without LangSmith configured, a local step-by-step trace prints (retrieved chunk count, refused flag, citations, elapsed time). That's the visible artifact for this activity
- If you configured real LangSmith credentials (`LANGCHAIN_TRACING_V2=true` and a real `LANGCHAIN_API_KEY` in `.env`), check https://smith.langchain.com under project `enterprise-knowledge-system` and confirm the run appears there as `enterprise_kb_query`""",
        script="""import time

from src.observability.tracing import configure_langsmith, run_query_with_tracing


def _sample_query(question: str) -> dict:
    \"\"\"A self-contained stand-in for a real retrieve+generate call, shaped
    exactly like generate_answer()'s result - keeps this activity testable
    without an embeddings/chat API call, since what's being validated here
    is the tracing wrapper's own mechanics, not generation quality.\"\"\"
    time.sleep(0.05)
    return {
        "answer": "The minimum balance for a metro savings account is INR 5,000 [1].",
        "citations": [{"marker": 1, "source": "retail_banking/savings_account_policy.md"}],
        "retrieved_chunks": [{"source": "retail_banking/savings_account_policy.md", "score": 0.91}],
        "refused": False,
    }


try:
    result = run_query_with_tracing(_sample_query, "What is the minimum balance for a metro savings account?")
    print(f"\\nTraced result: {result}")
    langsmith_on = configure_langsmith()
    print(f"\\nLangSmith configured: {langsmith_on}")
    print("[OK] run_query_with_tracing() ran (via LangSmith if configured, else the local trace above).")
except NotImplementedError as e:
    print(f"[WARN] Not implemented yet: {e}")
    print("Complete run_query_with_tracing() in src/observability/tracing.py first.")
""",
    ),
    dict(
        id="3.2", phase=3, title="Automated Testing",
        module="`tests/test_pipeline.py`",
        business_context="""Clicking through the portal by hand doesn't scale as a regression check. A
small automated pytest suite lets the team (and CI) confirm in seconds that the pipeline's structure
hasn't broken, with no manual review.""",
        concept_overview="""This is a **small, fixed smoke suite**. It checks shape (metadata completeness,
chunk sizing, citation mapping), never LLM output quality. None of the tests call the real OpenAI API, so
they run even without a configured key.""",
        objective="Implement the three TODO tests in tests/test_pipeline.py (the other two are already filled in).",
        instructions="""In `tests/test_pipeline.py`, implement the three TODO tests, following each docstring's numbered steps:
1. `test_all_governed_documents_have_required_metadata`: load the real corpus (`load_corpus`) and manifest (`load_manifest`), run `process_corpus`, assert at least 10 `ProcessedDocument`s come back and every one has a non-empty value for each name in `CFG["metadata"]["required_fields"]`.
2. `test_chunking_respects_configured_size`: build a long string, call `split_text(text, "recursive", chunk_size=300, chunk_overlap=40)`, assert more than one chunk, every chunk non-empty, and no chunk longer than `chunk_size * 1.5` characters.
3. `test_citation_parser_maps_markers_to_sources`: build three `RetrievedChunk` objects, call `parse_citations("Per policy [1] and the circular [2, 3].", chunks)`, assert markers `[1, 2, 3]` map to the three sources in order, and that a marker-free string returns `[]`.""",
        ai_assist="**AI Assist:** Ask your AI coding assistant to explain why this suite avoids calling the real LLM. What would calling it cost you in speed, determinism, and offline use, the very things a structural test suite is meant to keep?",
        files=["tests/test_pipeline.py"],
        target_methods={"tests/test_pipeline.py": [
            "test_all_governed_documents_have_required_metadata",
            "test_chunking_respects_configured_size",
            "test_citation_parser_maps_markers_to_sources",
        ]},
        milestone="A pytest suite automatically validates the pipeline's structural behavior.",
        validation="""Confirm that:
- All 5 tests in `tests/test_pipeline.py` pass (`5 passed` in pytest's summary line)
- Re-read any test that fails: the failure message names the exact assertion that didn't hold""",
        has_reference_run=False,
        script="""# Activity 3.2: run the pytest smoke suite. This never raises itself; it
# reports pytest's own pass/fail in the printed output instead, with an
# explicit warning marker on any failure so the app's own pass/warn/error
# status picks it up correctly either way.
import subprocess
import sys

proc = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_pipeline.py", "-v"],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
)
print(proc.stdout[-4000:])
if proc.returncode != 0:
    print(proc.stderr[-2000:])
    print(f"\\n[WARN] pytest exited with code {proc.returncode}: complete the TODO tests in tests/test_pipeline.py.")
else:
    print("\\n[OK] pytest suite passed.")
""",
    ),
    dict(
        id="3.3", phase=3, title="Deployment with Docker",
        module="*No module to implement: build and run the provided Dockerfile*",
        business_context="""A system that only runs "on my machine" isn't enterprise-ready. Docker packages
the whole pipeline (code, dependencies, runtime) into one reproducible image that runs the same in
any environment.""",
        objective="Build the provided Docker image, run it, and confirm the portal is reachable inside the container.",
        instructions="""There's no code cell for this activity: Docker build/run happens in your terminal, not inside this app.
1. From the project root, run `sudo docker build -t enterprise-knowledge-system .` (or `sudo docker compose build`).
2. Run `sudo docker compose up` (this reads your `.env` file through `env_file:` in `docker-compose.yml`, so check that `.env` exists first).
3. Open `http://localhost:8502` in a browser and confirm the Enterprise Knowledge Portal loads inside the container. (The container maps host port **8502** to container 8501, so it won't clash with this workbook app or a local `streamlit run` on 8501.)
4. Once the container is up, click **Run Activity 3.3** below to confirm the health endpoint responds from your host machine.
5. Stop the container with `sudo docker compose down` when you're done.

**Troubleshooting the lab VM's Linux Docker setup:**
- `docker: command not found` - Docker isn't installed on this machine yet. Install it: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`, then `sudo usermod -aG docker $USER && newgrp docker` and `sudo systemctl enable --now docker`. If you have no `sudo` access, this is a lab-provisioning gap, not something you can fix yourself - flag it instead of getting stuck here.
- `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock` - your user isn't in the `docker` group yet in this terminal session. Either prefix the command with `sudo` for now, or close and reopen your terminal (or VS Code) after running the `usermod` step above so the new group membership actually applies. Verify with `groups` that `docker` is listed.""",
        ai_assist="**AI Assist:** If `docker build` fails, paste the exact error into your AI coding assistant along with the `Dockerfile`: most first-time Docker build failures on a new machine are missing system dependencies or a stale image cache, not something wrong with this project's code.",
        files=[],
        reference_only_files=["Dockerfile", "docker-compose.yml"],
        has_reference_run=False,
        milestone="The knowledge portal runs successfully inside a Docker container.",
        validation="""Confirm that:
- The health check reports a successful response once the container is running
- The portal at `http://localhost:8502` looks and behaves the same as when run locally with `streamlit run app/knowledge_portal.py`
- A question asked inside the container succeeds, meaning your `OPENAI_API_KEY` reached the container correctly via `.env` passthrough""",
        script="""# Activity 3.3: verify the containerized portal is reachable. Run this
# AFTER `sudo docker compose up`, from your host machine. The container publishes
# on host port 8502 (-> container 8501), per docker-compose.yml.
import urllib.request
import urllib.error

try:
    with urllib.request.urlopen("http://localhost:8502/_stcore/health", timeout=5) as resp:
        print(f"[OK] Container health check responded with status {resp.status}.")
except (urllib.error.URLError, ConnectionRefusedError, TimeoutError) as e:
    print(f"[WARN] Could not reach the container at localhost:8502 ({e}).")
    print("   Run `sudo docker compose up` in a terminal first, then re-run this check.")
""",
    ),
]

ACTIVITIES_BY_ID = {a["id"]: a for a in ACTIVITIES}
PHASE_TITLES = {
    1: "Phase 1 - Ingestion & Governance",
    2: "Phase 2 - Index, Retrieve, Generate",
    3: "Phase 3 - Observability, Testing & Deployment",
}
PHASE_INTRO = {
    1: dict(
        objective="Turn a folder of mixed-format files into a governed corpus: every document loads as "
                  "plain text and carries the enterprise metadata the rest of the pipeline depends on.",
        outcome="By the end of this phase, all 4 formats load cleanly, every governed document is "
                "tagged with department/doc_type/product/version/effective_date, and an ungoverned file is left out.",
    ),
    2: dict(
        objective="Split the corpus into retrievable passages, index them into a real vector store, fuse "
                  "vector and lexical retrieval, and generate grounded, cited answers.",
        outcome="By the end of this phase, a real question against the corpus returns a ranked, "
                "department-filtered set of chunks and a cited, grounded answer (or a clean refusal).",
    ),
    3: dict(
        objective="Add tracing, automated tests, and a deployment path to the pipeline.",
        outcome="By the end of this phase, a query is traceable, an automated test suite passes against "
                "your implementation, and the knowledge portal runs inside Docker.",
    ),
}

# Short, single-line labels for the sidebar nav: the full descriptive title
# (used in PAGES, the hero banner, and Prev/Next buttons) wraps to 2-3 lines
# in the narrow sidebar and, repeated for all 9 activities, buries the actual
# navigation under a wall of text. The nav only needs enough to recognize
# "where am I"; the full title is one click away on the activity page itself.
NAV_SHORT_TITLES = {
    "1.1": "Explore the Pipeline",
    "1.2": "Multi-Source Loaders",
    "1.3": "Metadata & Governance",
    "2.1": "Chunk & Index",
    "2.2": "Hybrid Retrieval",
    "2.3": "Grounded Generation",
    "3.1": "LangSmith Tracing",
    "3.2": "Automated Testing",
    "3.3": "Containerize & Deploy",
}

REFERENCE_MAP = {
    "src/ingestion/loaders.py": "reference_solution/src/ingestion/loaders.py",
    "src/ingestion/document_processor.py": "reference_solution/src/ingestion/document_processor.py",
    "src/chunking/chunker.py": "reference_solution/src/chunking/chunker.py",
    "src/indexing/vector_store.py": "reference_solution/src/indexing/vector_store.py",
    "src/retrieval/retriever.py": "reference_solution/src/retrieval/retriever.py",
    "src/generation/answer_generator.py": "reference_solution/src/generation/answer_generator.py",
    "src/observability/tracing.py": "reference_solution/src/observability/tracing.py",
    "tests/test_pipeline.py": "reference_solution/tests/test_pipeline.py",
}


# ---------------------------------------------------------------------------
# Staleness: has an activity's file been edited since its last recorded run?
# ---------------------------------------------------------------------------

def _activity_files_mtime(activity: dict):
    """Latest modification time across an activity's `files`, or None if it
    has none to go stale against (1.1 orientation, 3.3 Docker - see
    render_activity_runner's has_reference_run comment for the same split)."""
    mtimes = []
    for rel in activity.get("files") or []:
        path = PROJECT_ROOT / rel
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else None


def activity_run_is_stale(activity: dict) -> bool:
    """True if the activity has a recorded status but its target file(s)
    have been modified since that status was recorded - e.g. a learner
    edited src/ back to broken after an earlier "Run Activity" click passed.
    Activities with no `files` (nothing to edit back to broken) are never
    stale."""
    if not activity.get("files"):
        return False
    if activity["id"] not in st.session_state.activity_status:
        return False
    recorded = st.session_state.activity_status_mtime.get(activity["id"])
    if recorded is None:
        return False
    current = _activity_files_mtime(activity)
    return current is not None and current > recorded


def activity_counts_as_pass(activity: dict) -> bool:
    """Whether an activity should count toward the progress tracker: a
    recorded "pass" that hasn't gone stale since."""
    return (
        st.session_state.activity_status.get(activity["id"]) == "pass"
        and not activity_run_is_stale(activity)
    )


# ---------------------------------------------------------------------------
# Sidebar: navigation + progress tracker
# ---------------------------------------------------------------------------

PAGES = ["Overview & Setup"] + [f"Activity {a['id']} - {a['title']}" for a in ACTIVITIES] + ["Final Review"]

# Two-level nav: a top-level segmented control for Overview / each Phase /
# Final Review, then (only for a Phase) a short radio scoped to that
# phase's activities. This keeps the always-visible list to at most 5
# items instead of all 9 activities at once, without the duplicate-selector
# trap other tools fall into (two separate widgets both choosing the same
# activity); this radio is the *only* control for "which activity."
TOP_LEVEL_OPTIONS = ["Overview"] + [f"Phase {p}" for p in PHASE_TITLES] + ["Final Review"]


def _nav_label(page: str) -> str:
    if page.startswith("Activity "):
        activity_id = page.replace("Activity ", "").split(" - ")[0]
        return f"{activity_id}  {NAV_SHORT_TITLES.get(activity_id, '')}"
    return page


def _phase_pages(phase_num: int) -> list:
    return [f"Activity {a['id']} - {a['title']}" for a in ACTIVITIES if a["phase"] == phase_num]


def _top_level_for(page: str) -> str:
    if page == PAGES[0] or page == PAGES[-1]:
        return TOP_LEVEL_OPTIONS[0] if page == PAGES[0] else TOP_LEVEL_OPTIONS[-1]
    activity_id = page.replace("Activity ", "").split(" - ")[0]
    return f"Phase {ACTIVITIES_BY_ID[activity_id]['phase']}"


if "nav_top" not in st.session_state:
    st.session_state.nav_top = TOP_LEVEL_OPTIONS[0]
# Widget session_state keys can't be reassigned after that widget has
# already been instantiated in the same run, so Previous/Next buttons
# (below) stage their target page here, applied before the widgets run.
if "pending_nav" in st.session_state:
    target = st.session_state.pop("pending_nav")
    st.session_state.nav_top = _top_level_for(target)
    if target not in (PAGES[0], PAGES[-1]):
        phase_num = ACTIVITIES_BY_ID[target.replace("Activity ", "").split(" - ")[0]]["phase"]
        st.session_state[f"nav_sub_{phase_num}"] = target

with st.sidebar:
    st.header("Guided Workbook")
    top_choice = st.segmented_control(
        "Section", TOP_LEVEL_OPTIONS, label_visibility="collapsed", key="nav_top"
    )
    if top_choice is None:
        # Clicking the active segment again deselects it in Streamlit. A
        # widget's session_state key can't be reassigned after that widget
        # has already run in this script pass, so defer the reset to
        # Overview via the same pending_nav mechanism the Previous/Next
        # buttons use (handled at the top of this block, before any widget
        # runs) and start a fresh run.
        st.session_state.pending_nav = PAGES[0]
        st.rerun()

    if top_choice in (TOP_LEVEL_OPTIONS[0], TOP_LEVEL_OPTIONS[-1]):
        selected_page = PAGES[0] if top_choice == TOP_LEVEL_OPTIONS[0] else PAGES[-1]
    else:
        phase_num = int(top_choice.split()[-1])
        phase_pages = _phase_pages(phase_num)
        sub_key = f"nav_sub_{phase_num}"
        if sub_key not in st.session_state:
            st.session_state[sub_key] = phase_pages[0]
        st.caption(f"Activities in {top_choice}")
        selected_page = st.radio(
            "Activity", phase_pages, format_func=_nav_label, label_visibility="collapsed", key=sub_key
        )

    st.divider()
    st.subheader("Progress")

    passed_count = sum(1 for a in ACTIVITIES if activity_counts_as_pass(a))
    total_count = len(ACTIVITIES)
    st.metric("Activities Complete", f"{passed_count} / {total_count}")
    st.progress(passed_count / total_count, text=f"{round(100 * passed_count / total_count)}% complete")

    if passed_count == total_count and not st.session_state.get("celebrated"):
        st.session_state.celebrated = True
        st.balloons()

    st.write("")
    for phase, phase_title in PHASE_TITLES.items():
        phase_activities = [a for a in ACTIVITIES if a["phase"] == phase]
        phase_done = sum(1 for a in phase_activities if activity_counts_as_pass(a))
        st.caption(f"{phase_title.split('-')[0].strip()}: {phase_done}/{len(phase_activities)}")
        st.progress(phase_done / len(phase_activities))


# ---------------------------------------------------------------------------
# Reusable page pieces
# ---------------------------------------------------------------------------

def render_file_hint(rel_path: str):
    ref_path = REFERENCE_MAP.get(rel_path)
    if ref_path and (PROJECT_ROOT / ref_path).exists():
        with st.expander(f"Need a hint? View reference solution for {rel_path}"):
            st.code(read_file(ref_path), language="python")


def extract_method_source(file_text: str, method_name: str) -> tuple:
    """Pull just one function/method's exact source (decorators included)
    out of a file's current text via ast, rather than the whole file, so
    the highlight below stays scoped to the few lines a participant is
    actually meant to touch. Returns (code, error): code is None if the
    file doesn't parse (a participant mid-edit with a syntax error) or the
    name isn't found; error carries the real SyntaxError detail so it can
    be shown instead of a generic "not found" message."""
    try:
        tree = ast.parse(file_text)
    except SyntaxError as e:
        return None, f"{e.msg} at line {e.lineno}, column {e.offset}"
    lines = file_text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            start_line = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return "\n".join(lines[start_line - 1:node.end_lineno]), None
    return None, None


def render_code_to_work_on(rel_path: str, method_names: list):
    """Live, read-only view of exactly the method(s) an activity asks the
    participant to implement, extracted fresh from their current src/ file
    on every render, isolated from the surrounding file so the ~10-30 lines
    that matter aren't buried in a full-file dump."""
    label_col, refresh_col = st.columns([5, 1])
    label_col.markdown("**Code you'll work on**: live from your current file, not the answer.")
    # This block only re-reads the file on a Streamlit rerun (any widget
    # interaction), saving in VS Code alone doesn't trigger one, so it can
    # sit stale while you're just looking at the page. Clicking anything
    # reruns the whole script anyway (read_file() below always re-reads
    # from disk); this button exists so there's something to click even
    # when you haven't touched any other widget.
    refresh_col.button("Refresh", key=f"refresh_code_{rel_path}", help="Re-check this file's status against what's currently saved on disk.")
    text = read_file(rel_path)
    for name in method_names:
        segment, error = extract_method_source(text, name)
        if segment is None:
            if error:
                st.caption(f"`{name}`: can't check TODO status yet: {error} in {rel_path}. Fix that first and the status below will work again.")
            else:
                st.caption(f"`{name}` not found in {rel_path} (it's been renamed or removed).")
            continue
        # Most stubs signal "not done" with `raise NotImplementedError`, but
        # a couple only leave a `# TODO:` comment behind, check for both, or
        # a half-finished method reads as done.
        done = "raise NotImplementedError" not in segment and "TODO" not in segment
        st.caption(("[done] " if done else "[ ] ") + f"`{name}`" + (": looks implemented" if done else ": still has a TODO"))
        st.code(segment, language="python")


STATUS_LABEL = {
    "pass": "[OK] All checks passed",
    "warn": "[WARN] Not fully implemented yet",
    "error": "[ERROR] The script raised an error",
}


def status_label_for(activity: dict, status: str) -> str:
    if status == "pass" and activity.get("pass_label"):
        return activity["pass_label"]
    return STATUS_LABEL.get(status, "Done")


def render_activity_runner(activity: dict):
    st.markdown("### Run & Validate")
    # The "See reference solution run" button re-runs this activity's own
    # check against reference_solution/. It only makes sense when the check
    # imports from src (something the participant built) AND the activity
    # itself opts in (has_reference_run isn't explicitly False): 1.1's
    # orientation script and 3.2's pytest subprocess and 3.3's Docker health
    # check aren't "run this against the finished src/" checks in the way
    # the other 6 are, so they get no run button. 3.2 still gets a
    # "Need a hint?" expander (reference_solution/tests/test_pipeline.py)
    # via REFERENCE_MAP; 3.3 has no reference_solution counterpart at all.
    has_reference_run = activity.get("has_reference_run", True) and (
        "from src" in activity["script"] or "import src" in activity["script"]
    )

    if has_reference_run:
        run_col, ref_col = st.columns(2)
    else:
        run_col = st.container()
    run_clicked = run_col.button(f"Run Activity {activity['id']}", type="primary", key=f"run_{activity['id']}")
    ref_clicked = has_reference_run and ref_col.button(
        "See reference solution run", key=f"ref_{activity['id']}",
        help="Runs this same check against the completed reference_solution/ instead of your src/. "
             "Use it if you're stuck and want to see what passing actually looks like. "
             "Doesn't count toward your progress."
    )

    if ref_clicked:
        with st.status("Running against reference_solution/ ...", expanded=True):
            result = run_snippet(activity["script"], use_reference=True)
            output = (result["stdout"] or "") + (("\n" + result["stderr"]) if result["stderr"] else "")
            st.code(output.strip() or "(no output)", language="text")
        st.session_state[f"reference_output_{activity['id']}"] = result
    elif f"reference_output_{activity['id']}" in st.session_state:
        with st.status("Reference solution output", expanded=False):
            result = st.session_state[f"reference_output_{activity['id']}"]
            output = (result["stdout"] or "") + (("\n" + result["stderr"]) if result["stderr"] else "")
            st.code(output.strip() or "(no output)", language="text")

    if run_clicked:
        with st.status("Running against your current src/ code...", expanded=True) as status_box:
            result = run_snippet(activity["script"])
            status = classify_run(result)
            output = (result["stdout"] or "") + (("\n" + result["stderr"]) if result["stderr"] else "")
            st.code(output.strip() or "(no output)", language="text")
            if result["timed_out"]:
                status_box.update(label="Timed out", state="error", expanded=True)
            else:
                status_box.update(
                    label=status_label_for(activity, status),
                    state="complete" if status == "pass" else ("error" if status == "error" else "complete"),
                    expanded=(status != "pass"),
                )
        st.session_state.activity_status[activity["id"]] = status
        st.session_state.activity_output[activity["id"]] = result
        # Record the files' mtime(s) at the moment of this run, so a later
        # edit to the same file(s) can be detected as staleness (see
        # activity_run_is_stale() above).
        st.session_state.activity_status_mtime[activity["id"]] = _activity_files_mtime(activity)
        # Rerun so the sidebar's progress tracker (rendered earlier in script
        # order, before this activity page) picks up the status set just now.
        st.rerun()

    elif activity["id"] in st.session_state.activity_output:
        # Re-show the last run's result on a fresh page load, without re-running it.
        result = st.session_state.activity_output[activity["id"]]
        status = st.session_state.activity_status.get(activity["id"])
        stale = activity_run_is_stale(activity)
        if stale:
            label = "[STALE] File edited since last run. Re-run to confirm"
            state = "error"
        else:
            label = status_label_for(activity, status)
            state = "complete" if status == "pass" else "error"
        with st.status(label, state=state, expanded=(stale or status != "pass")):
            output = (result["stdout"] or "") + (("\n" + result["stderr"]) if result["stderr"] else "")
            st.code(output.strip() or "(no output)", language="text")
            if stale:
                st.warning(
                    "You've edited this file since the last run. "
                    "Re-run the activity to confirm it still holds."
                )
        if status == "pass" and not stale:
            st.markdown(
                '<div class="badge-pass">[OK] Milestone reached: you\'re ready for the next activity.</div>',
                unsafe_allow_html=True,
            )

    if activity.get("validation"):
        # Collapsed by default: this is what to check in your own output
        # beyond "did it run," most useful once you actually have output to
        # read it against, not before.
        with st.expander("How to check your result"):
            st.markdown(activity["validation"])


# ---------------------------------------------------------------------------
# Page: Overview & Setup
# ---------------------------------------------------------------------------

if selected_page == "Overview & Setup":
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Guided Project: Production RAG & Enterprise Knowledge Systems</p>
        <p class="hero-subtitle">Production RAG for Retail Banking, Loans, Cards, Operations, and Compliance</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        f"{len(ACTIVITIES)} activities · {len(PHASE_TITLES)} phases · ~3 hrs · 18 governed documents · "
        f"Chroma/FAISS + BM25 hybrid retrieval + LangSmith + pytest + Docker"
    )

    st.markdown("""
A retail bank has a working RAG proof of concept that answers questions from **one** policy document.
It can't go enterprise-wide. It can't ingest multiple formats, it keeps no document context, it has no
departmental filtering, it gives you no traceability, and it has no deployment path. In this lab you take
that POC and, step by step, turn it into a **production-ready Enterprise Knowledge System** for Retail
Banking, Loans, Credit Cards, Operations, and Compliance.

The project structure, UI, config, and data are already built. Your job is ingestion, governance,
chunking, indexing, retrieval, generation, observability, testing, and deployment, one guided activity
at a time. This app replaced the earlier marimo notebook build (archived under
`notebooks_marimo_archive/`) as the main hands-on surface for this course.
""")

    if st.button("Start Activity 1.1: Explore the Pipeline", type="primary", use_container_width=True):
        st.session_state.pending_nav = PAGES[1]
        st.rerun()

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Business Context & Objectives", "What Makes This \"Production-Grade\"", "Solution Architecture"])

    with tab1:
        with st.container(border=True):
            st.markdown("##### What You'll Build")
            st.markdown("""
- Loaders for all 4 corpus formats (PDF, DOCX, HTML, Markdown), normalized to plain text
- Enterprise metadata that keeps ungoverned documents out of the index
- Three chunking strategies and a real, persisted vector store (Chroma, with a FAISS comparison path)
- A `HybridRetriever` that combines vector similarity with BM25 lexical search, scoped by department
- Grounded, cited answer generation with a built-in refusal case
- A LangSmith-traced query entry point (with an offline local-trace fallback)
- A pytest smoke suite that checks the pipeline's structure
- A knowledge system that runs inside Docker
""")
            st.caption("You build it across 9 guided activities in 3 phases, and you check each one before you move on.")

        with st.container(border=True):
            st.markdown("##### The Business Problem")
            st.markdown("""
Loan officers, compliance staff, and branch teams each need answers from a different slice of the bank's
policy corpus. The answers have to be current, not superseded, and cited well enough to trust in a
customer-facing decision. A single-document POC can't tell a current policy from a 2024 draft. It has no
departmental scoping, and none of the tracing, testing, and deployment a production system needs. This lab
closes that gap.
""")

        with st.container(border=True):
            st.markdown("##### Your Role")
            st.markdown("""
You're joining the bank's AI Platform Engineering team. Your job is to take a working prototype and
make it production-ready: multi-format ingestion, governed metadata, hybrid retrieval, grounded
generation, observability, tests, and a deployment path.
""")

        with st.container(border=True):
            st.markdown("##### Learning Objectives: by the end of this lab you'll be able to")
            lo_col1, lo_col2 = st.columns(2)
            with lo_col1:
                st.markdown("""
- Ingest a multi-format enterprise corpus (PDF, DOCX, HTML, Markdown)
- Enrich and govern documents with enterprise metadata
- Chunk and index documents into a real, persisted vector store
- Implement hybrid (vector + lexical) retrieval with metadata filtering
""")
            with lo_col2:
                st.markdown("""
- Generate grounded, cited answers that refuse out-of-scope questions
- Trace a query end-to-end with LangSmith (or a local fallback)
- Validate pipeline behavior with automated tests
- Containerize and deploy an enterprise AI application
""")

        with st.container(border=True):
            st.markdown("##### Dataset: already provided under `data/`")
            ds_col1, ds_col2 = st.columns(2)
            with ds_col1:
                st.markdown("""
**Enterprise Knowledge Base** (`knowledge_base/`)
18 governed documents (6 PDF / 5 DOCX / 3 HTML / 5 MD) across 5 departments, plus 1 ungoverned file and
a superseded-version pair (`loans/personal_loan_guidelines.docx` v2.0 vs. `_v1.docx` v1.0), all tracked
in `manifest.csv`.
""")
            with ds_col2:
                st.markdown("""
**Validation Scenarios** (`validation/test_scenarios.json`)
8 named scenarios (SCN-01 .. SCN-08). Activities 2.2 and 2.3 use them as example input, and so do the
reference tests in the Activity 3.2 pytest smoke suite. SCN-08 is the deliberate refusal case (there's no
FD document in the corpus).
""")

    with tab2:
        st.markdown("""
A **single-document POC** answers questions from one file. No metadata, no format variety, no filtering.
This bootcamp shows you the production version: **governed multi-format ingestion**, a **real persisted
vector store**, **hybrid retrieval** that combines lexical and semantic signals, and **grounded, cited
generation** that refuses when the corpus doesn't cover the question. All of it comes with tracing, tests,
and a deployment path.
""")
        st.markdown("""
Here's why that matters at production scale. The lab makes each point concrete:

- **Governance before indexing.** A document with no manifest row (`operations/draft_notes.md`) never reaches the index. A bank needs that same discipline before letting any content answer a customer-facing question.
- **Hybrid retrieval.** Pure vector search misses exact terms (a circular number, "FOIR") and can't tell a current policy from a superseded one. BM25 fusion plus a metadata filter fixes both.
- **Grounded, cited generation.** Every claim in an answer points to a numbered passage. When nothing in the corpus supports the question, the system refuses instead of hallucinating (see SCN-08).
- **Operability.** A pipeline that can't be traced (LangSmith, Activity 3.1), can't be tested automatically (pytest, Activity 3.2), and can't be deployed repeatably (Docker, Activity 3.3) is a prototype, not a production system, however good its retrieval and generation are.
""")

    with tab3:
        st.markdown("Here's how the pipeline is staged, and which activity implements each stage:")
        st.iframe(PIPELINE_DIAGRAM_SVG, height=260)
        st.markdown("""
Two things to notice. You'll run into both throughout the lab:
- **The portal (`app/knowledge_portal.py`) only ever calls `src.pipeline.KnowledgeSystem`.** The real ingestion, retrieval, and generation logic lives in `src/`, not in the portal or this workbook.
- **Testing, observability, and deployment sit beside the pipeline stages, not above them.** They wrap and check the system. They don't replace understanding what it actually retrieved and answered.
""")

    import importlib.util
    required_packages = ["openai", "langsmith", "chromadb", "faiss", "rank_bm25", "langchain", "streamlit", "yaml", "dotenv", "pytest"]
    missing_packages = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]

    # OPENAI_API_KEY works with either a real OpenAI key or an OpenRouter
    # key (see .env.example); LLM_BASE_URL picks which one src/llm/ actually
    # talks to.
    key = os.getenv("OPENAI_API_KEY")
    key_missing = not key or key in ("your_openai_api_key_here", "your_openai_or_openrouter_api_key_here", "demo")

    required_paths = [
        "config/ingestion_config.yaml", "config/llm_config.yaml", "config/prompts.yaml",
        "data/knowledge_base/manifest.csv", "data/validation/test_scenarios.json", "data/knowledge_base",
    ]
    missing_paths = [p for p in required_paths if not (PROJECT_ROOT / p).exists()]

    all_ok = not missing_packages and not key_missing and not missing_paths
    expander_label = "[OK] Environment ready" if all_ok else "[WARN] Environment Validation: action needed"

    with st.expander(expander_label, expanded=not all_ok):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Packages**")
            if missing_packages:
                st.markdown(
                    f'<div class="badge-fail">[ERROR] Missing: {", ".join(missing_packages)}<br>'
                    f'Run: <code>pip install -r requirements.txt</code></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="badge-pass">[OK] All required packages are installed.</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("**API Key**")
            if key_missing:
                st.markdown(
                    '<div class="badge-fail">[ERROR] OPENAI_API_KEY is not set.<br>'
                    'Copy <code>.env.example</code> to <code>.env</code> and add your OpenAI or OpenRouter key.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="badge-pass">[OK] OPENAI_API_KEY is configured.</div>', unsafe_allow_html=True)
                st.caption("Needed only for Activities 2.1 (embeddings), 2.3 (generation), and optionally 3.1 (real LangSmith run). "
                          "Every other activity runs without it.")

        with col3:
            st.markdown("**Project Assets**")
            if missing_paths:
                st.markdown(f'<div class="badge-fail">[ERROR] Missing: {", ".join(missing_paths)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="badge-pass">[OK] All {len(required_paths)} required project assets are present.</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Page: Final Review
# ---------------------------------------------------------------------------

elif selected_page == "Final Review":
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Final Review</p>
        <p class="hero-subtitle">Look back at what you've built, then run one last live demo</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
Here's what you've built across all three phases:
- **Ingestion & Governance**: 4-format loaders and manifest-driven metadata that keeps ungoverned documents out (Phase 1)
- **Index, Retrieve, Generate**: chunking, a real persisted vector store, hybrid retrieval, and grounded, cited generation (Phase 2)
- **Observability, Testing & Deployment**: a traced query entry point, an automated pytest suite, and a containerized system (Phase 3)
""")

    if st.button("Run Final Demo", type="primary"):
        with st.spinner("Running the full pipeline, start to finish..."):
            result = run_snippet("""import json

from src.chunking.chunker import chunk_corpus
from src.generation.answer_generator import generate_answer, validate_answer
from src.ingestion.document_processor import load_manifest, process_corpus
from src.ingestion.loaders import load_corpus
from src.observability.tracing import run_query_with_tracing
from src.retrieval.retriever import HybridRetriever

scenarios = {s["id"]: s for s in json.load(open("data/validation/test_scenarios.json", encoding="utf-8"))["scenarios"]}
question = scenarios["SCN-01"]["question"]

try:
    raws = load_corpus("data/knowledge_base")
    manifest = load_manifest("data/knowledge_base/manifest.csv")
    processed = process_corpus(raws, manifest)
    chunks = chunk_corpus(processed)
    print(f"Corpus: {len(raws)} raw document(s), {len(processed)} governed after processing, {len(chunks)} chunk(s).")

    dept_chunks = [c for c in chunks if c.metadata.get("department") == "retail_banking"]

    class _StubDoc:
        def __init__(self, content, metadata):
            self.page_content, self.metadata = content, metadata

    class _StubStore:
        def __init__(self, chunks):
            self.chunks = chunks
        def similarity_search_with_relevance_scores(self, query, k=5, filter=None):
            qs = set(query.lower().split())
            out = []
            for c in self.chunks:
                if filter and any(c.metadata.get(kk) != vv for kk, vv in filter.items()):
                    continue
                ov = len(qs & set(c.content.lower().split()))
                out.append((_StubDoc(c.content, c.metadata), ov / (len(qs) or 1)))
            out.sort(key=lambda x: x[1], reverse=True)
            return out[:k]

    retriever = HybridRetriever(_StubStore(dept_chunks), dept_chunks)
    hits = retriever.retrieve(question, metadata_filter={"department": "retail_banking"})
    print(f"Retrieval: {len(hits)} hit(s) for {question!r}")

    def _run_generation():
        return generate_answer(question, hits)

    traced = run_query_with_tracing(_run_generation)
    print(f"\\nRefused: {traced.get('refused')}")
    print(f"Answer: {traced.get('answer')}")
    print(f"Validation errors: {validate_answer(traced)}")
    print("\\n[OK] End-to-end pipeline execution complete.")
except NotImplementedError as e:
    print(f"[WARN] Not all activities are complete yet: {e}")
except Exception as e:
    print(f"[WARN] Final demo needs a working OPENAI_API_KEY for the generation step: {type(e).__name__}: {e}")
""")
        output = (result["stdout"] or "") + (("\n" + result["stderr"]) if result["stderr"] else "")
        st.code(output.strip() or "(no output)", language="text")

    st.markdown("---")
    st.markdown("#### Key Takeaways")

    kt_col1, kt_col2, kt_col3 = st.columns(3)
    with kt_col1:
        st.markdown("""**What You Built**
- Multi-format ingestion (PDF, DOCX, HTML, Markdown)
- Governed enterprise metadata
- A real, persisted vector store over chunked documents
- Hybrid (vector + BM25) retrieval with department filtering
- Grounded, cited answer generation with refusal
- Traced queries, automated tests, and a Docker deployment path""")
    with kt_col2:
        st.markdown("""**Engineering Principles You Applied**
- Governance before indexing, not after
- Separation of concerns between retrieval and generation
- Config-driven, modular code
- A runnable check at every step, never prose alone
- Enterprise readiness means documentation and process, not just code""")
    with kt_col3:
        st.markdown("""**Skills You Practiced**
- Multi-source document ingestion and normalization
- Metadata-driven governance and filtering
- Hybrid retrieval design (vector + lexical fusion)
- Grounded, cited generation with LLM observability
- Automated testing and containerization of an AI system""")

    st.markdown("""
    <div class="live-banner">
        <p class="hero-title"><span class="pulse-dot"></span>🏦 Try It Live: Enterprise Knowledge Portal</p>
        <p class="hero-subtitle">Ask your own pipeline a real question, right here, with no separate app or tab.
        It runs the same <code>src.pipeline.KnowledgeSystem</code> the standalone
        <code>app/knowledge_portal.py</code> uses, against whatever you've implemented in <code>src/</code>
        so far.</p>
    </div>
    """, unsafe_allow_html=True)
    with st.container(border=True):
        render_knowledge_portal(key_prefix="wb_final_")


# ---------------------------------------------------------------------------
# Page: an individual activity
# ---------------------------------------------------------------------------

else:
    activity_id = selected_page.replace("Activity ", "").split(" - ")[0]
    activity = ACTIVITIES_BY_ID[activity_id]

    # Floating Code Assistant panel, docked to the right edge of the
    # viewport (see the div[data-testid="stPopover"] CSS above). Keeps the
    # chat scoped to whichever activity the learner is currently on, and
    # re-uses app/code_assistant_panel.py's chat logic unchanged - no
    # switching to a separate Streamlit app/tab required.
    with st.popover("\U0001F9D1‍\U0001F4BB Code Assistant"):
        st.caption(f"Chat about Activity {activity_id} without leaving the workbook.")
        render_code_assistant_chat(activity_id, key_prefix="wb_", show_instructions=False)

    # Shown once per phase, right before that phase's first activity.
    if activity["id"].endswith(".1"):
        phase_intro = PHASE_INTRO[activity["phase"]]
        st.markdown(f"#### {PHASE_TITLES[activity['phase']]}")
        st.markdown(f"**Objective:** {phase_intro['objective']}")
        st.caption(phase_intro["outcome"])
        st.divider()

    st.markdown(f"""
    <div class="hero-banner">
        <p class="hero-title">Activity {activity['id']} - {activity['title']}</p>
        <p class="hero-subtitle">{activity['module']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"**What success looks like:** {activity['milestone']}")

    with st.container(border=True):
        st.markdown("#### Business Context")
        st.markdown(activity["business_context"])

    with st.container(border=True):
        st.markdown("#### Objective")
        st.markdown(activity["objective"])

    if activity.get("concept_overview"):
        with st.container(border=True):
            st.markdown("#### Concept Overview")
            st.markdown(activity["concept_overview"])

    with st.container(border=True):
        st.markdown("#### Implementation Instructions")
        st.markdown(activity["instructions"])
        if activity.get("ai_assist"):
            st.info(activity["ai_assist"])

    if activity.get("reference_only_files"):
        # Collapsed by default: a full config file dumped open on the page
        # is exactly the "wall of code" that makes a lab feel harder than it
        # is before anyone's read a word of the instructions above.
        with st.expander(f"Reference Files: {', '.join(activity['reference_only_files'])}"):
            tabs = st.tabs(activity["reference_only_files"])
            for tab, path in zip(tabs, activity["reference_only_files"]):
                with tab:
                    lang = "dockerfile" if "Dockerfile" in path or "docker-compose" in path else "yaml"
                    st.code(read_file(path), language=lang)

    if activity["files"]:
        st.markdown("#### Files to Edit")
        st.info(
            "Open these in VS Code (or your preferred IDE) to write your implementation. This app runs "
            "whatever is currently saved on disk. It doesn't edit files itself."
        )
        target_methods = activity.get("target_methods", {})
        if len(activity["files"]) == 1:
            path = activity["files"][0]
            st.code(path, language="text")
            if path in target_methods:
                render_code_to_work_on(path, target_methods[path])
            render_file_hint(path)
        else:
            # Tab labels already name each file; an extra list above them
            # would just repeat the same names twice in a row.
            tabs = st.tabs(activity["files"])
            for tab, path in zip(tabs, activity["files"]):
                with tab:
                    if path in target_methods:
                        render_code_to_work_on(path, target_methods[path])
                    render_file_hint(path)

    st.divider()
    render_activity_runner(activity)

    st.divider()
    page_index = PAGES.index(selected_page)
    prev_col, next_col = st.columns(2)
    with prev_col:
        if page_index > 0:
            if st.button(f"< {PAGES[page_index - 1]}", use_container_width=True):
                st.session_state.pending_nav = PAGES[page_index - 1]
                st.rerun()
    with next_col:
        if page_index < len(PAGES) - 1:
            if st.button(f"{PAGES[page_index + 1]} >", use_container_width=True, type="primary"):
                st.session_state.pending_nav = PAGES[page_index + 1]
                st.rerun()
