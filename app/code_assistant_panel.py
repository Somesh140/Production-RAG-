"""
Code Assistant chat panel.

The one and only Code Assistant surface in this course: a floating panel
docked to the top-right of every activity page in app/workbook_app.py, so a
learner never has to switch to a separate app/tab to get unstuck - the
assistant is right there next to the activity they're on, auto-scoped to it.

There used to also be a standalone app/code_assistant.py page with its own
sidebar activity picker; it was removed (23 Sep 2026) once the embedded panel
covered the same ground, so learners wouldn't have to choose between two
near-identical chat surfaces. This module is the only place the chat logic
(activity catalog, live TODO-instruction loading, reference-solution
grounding, reference-dump guard) lives now.

Nothing in this module calls st.set_page_config or reads OPENAI_API_KEY at
import time: the caller (app/workbook_app.py) decides page framing and the
missing-key gate.
"""
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.llm_client import LLMClient  # noqa: E402
from src.llm.prompt_manager import PromptManager  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Activity catalog - identical keys/content to the one app/code_assistant.py
# used to own. Kept here as the single source of truth; app/code_assistant.py
# imports it rather than redefining it. Same 6 activities as
# app/workbook_app.py's ACTIVITIES_BY_ID (1.1/3.2/3.3 excluded there too, for
# the same reason: no "generate code for a src/ function" TODO to chat about).
# ---------------------------------------------------------------------------
ACTIVITIES: Dict[str, Dict[str, object]] = {
    "1.2": {
        "label": "Activity 1.2 - Multi-source document loaders (Beginner)",
        "src_file": "src/ingestion/loaders.py",
        "ref_files": ["reference_solution/src/ingestion/loaders.py"],
        "functions": ["_load_markdown", "_load_pdf", "_load_docx", "_load_html"],
    },
    "1.3": {
        "label": "Activity 1.3 - Document processing & enterprise metadata (Beginner)",
        "src_file": "src/ingestion/document_processor.py",
        "ref_files": ["reference_solution/src/ingestion/document_processor.py"],
        "functions": ["build_metadata", "process_document"],
    },
    "2.1": {
        "label": "Activity 2.1 - Chunking & vector store indexing (Intermediate)",
        "src_file": "src/chunking/chunker.py",
        "extra_src_files": ["src/indexing/vector_store.py"],
        "ref_files": [
            "reference_solution/src/chunking/chunker.py",
            "reference_solution/src/indexing/vector_store.py",
        ],
        "functions": ["split_text", "build_documents", "index_chunks", "search"],
    },
    "2.2": {
        "label": "Activity 2.2 - Hybrid retrieval (Intermediate)",
        "src_file": "src/retrieval/retriever.py",
        "ref_files": ["reference_solution/src/retrieval/retriever.py"],
        "functions": ["HybridRetriever.retrieve"],
    },
    "2.3": {
        "label": "Activity 2.3 - Grounded answer generation (Intermediate)",
        "src_file": "src/generation/answer_generator.py",
        "ref_files": ["reference_solution/src/generation/answer_generator.py"],
        "functions": ["generate_answer"],
    },
    "3.1": {
        "label": "Activity 3.1 - LangSmith tracing (Advanced)",
        "src_file": "src/observability/tracing.py",
        "ref_files": ["reference_solution/src/observability/tracing.py"],
        "functions": ["run_query_with_tracing"],
    },
}

MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_TURN_PAIRS = 6
REFERENCE_DUMP_THRESHOLD = 0.55


def _find_function_docstring(tree: ast.Module, qualname: str) -> Optional[str]:
    """Find a top-level function or a dotted Class.method's docstring inside
    a parsed module. Returns None if not found."""
    parts = qualname.split(".")
    if len(parts) == 1:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[0]:
                return ast.get_docstring(node)
        return None
    class_name, method_name = parts[0], parts[1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == method_name:
                    return ast.get_docstring(sub)
    return None


def load_current_instructions(activity_key: str) -> str:
    """Read the selected activity's TODO docstring(s) live from the current
    src/ file(s), so the instructions shown track the starter file as it
    evolves rather than a hardcoded copy."""
    activity = ACTIVITIES[activity_key]
    src_files = [activity["src_file"]] + list(activity.get("extra_src_files", []))
    blocks: List[str] = []
    for rel_path in src_files:
        path = PROJECT_ROOT / rel_path
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except Exception as e:
            logger.warning(f"Could not parse {rel_path} for live instructions: {e}")
            blocks.append(f"(Could not read {rel_path}: {e})")
            continue
        for func in activity["functions"]:
            doc = _find_function_docstring(tree, func)
            if doc:
                blocks.append(f"### {rel_path} :: {func}\n\n{doc}")
    return "\n\n---\n\n".join(blocks) if blocks else "(Could not load live instructions for this activity.)"


def load_reference_code(activity_key: str) -> str:
    """Read the full reference_solution/ file(s) for the selected activity -
    the grounding context fed into the system prompt so generated code
    matches this course's real, working patterns instead of something
    generically plausible."""
    activity = ACTIVITIES[activity_key]
    blocks: List[str] = []
    for rel_path in activity["ref_files"]:
        path = PROJECT_ROOT / rel_path
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read reference file {rel_path}: {e}")
            content = f"(Could not read {rel_path}: {e})"
        blocks.append(f"# --- {rel_path} ---\n{content}")
    return "\n\n".join(blocks)


def is_reference_dump(reply: str, reference_code: str, threshold: float = REFERENCE_DUMP_THRESHOLD) -> bool:
    """Code-side guard against prompt injection that tries to get the model
    to reproduce the reference_solution/ file(s) verbatim."""
    ref_lines = [line.strip() for line in reference_code.splitlines() if line.strip()]
    if not ref_lines:
        return False
    matched = sum(1 for line in ref_lines if line in reply)
    return (matched / len(ref_lines)) >= threshold


def render_chat(activity_key: str, key_prefix: str = "", show_instructions: bool = True) -> None:
    """Render the chat panel (instructions expander + history + chat input)
    for one activity. Safe to call multiple times per session with different
    key_prefix values (e.g. once from the standalone page, once from the
    floating panel) without Streamlit widget-key collisions - both share the
    same underlying st.session_state.code_assistant_histories, so chat
    history for a given activity is the same wherever it's opened from.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key or key in ("demo", "your_openai_or_openrouter_api_key_here"):
        st.warning("Add your `OPENAI_API_KEY` to `.env` to use the code assistant. It calls the model.")
        return

    if activity_key not in ACTIVITIES:
        st.info(
            "This activity has no `src/` TODO function to write code for. "
            "It's an orientation, testing or deployment step. The code assistant "
            "covers Activities 1.2, 1.3, 2.1, 2.2, 2.3 and 3.1."
        )
        return

    if "code_assistant_histories" not in st.session_state:
        st.session_state.code_assistant_histories = {}

    expanded_state_key = f"{key_prefix}code_assistant_expanded"
    expanded = st.session_state.get(expanded_state_key, False)

    header_col, clear_col, expand_col = st.columns([5, 1, 1])
    with header_col:
        activity = ACTIVITIES[activity_key]
        st.markdown(f"##### {activity['label']}")
        src_files_display = ", ".join([activity["src_file"]] + list(activity.get("extra_src_files", [])))
        st.caption(f"src/ file(s): `{src_files_display}`")
    with clear_col:
        if st.button("\U0001F5D1", key=f"{key_prefix}clear_chat_{activity_key}",
                     help="Clear this activity's chat history"):
            st.session_state.code_assistant_histories[activity_key] = []
            logger.info(f"Chat history cleared for activity {activity_key}")
            st.rerun()
    with expand_col:
        # Icon-only toggle button (manual state flip, not st.toggle): a
        # text label wraps to multiple lines in this narrow column inside
        # the popover's ~420px width, which st.toggle's own label can't
        # avoid - a bare icon with a tooltip fits cleanly instead.
        if st.button(
            "◲" if not expanded else "◰",
            key=f"{expanded_state_key}_btn_{activity_key}",
            help="Collapse this panel" if expanded else "Expand this panel for easier reading",
        ):
            expanded = not expanded
            st.session_state[expanded_state_key] = expanded
            st.rerun()

    if expanded:
        # Overrides the default popover sizing set in app/workbook_app.py's
        # CSS - injected here (after that block, later in the DOM) so it
        # wins on equal !important specificity without touching that file.
        st.markdown("""
        <style>
        div[data-testid="stPopoverBody"] {
            width: min(760px, 95vw) !important;
            max-height: 85vh !important;
        }
        </style>
        """, unsafe_allow_html=True)

    if show_instructions:
        with st.expander("Current TODO instructions (read live from src/)"):
            st.markdown(load_current_instructions(activity_key))

    st.session_state.code_assistant_histories.setdefault(activity_key, [])
    history = st.session_state.code_assistant_histories[activity_key]

    user_message = st.chat_input(
        f"Ask about {activity_key}, e.g. \"help me implement this\"",
        key=f"{key_prefix}chat_input_{activity_key}",
    )

    # Newest message first, right under the input, so a learner never has to
    # scroll down through the whole conversation to see the latest reply -
    # especially inside the floating popover, where vertical space is tight.
    history_placeholder = st.container()

    if user_message:
        user_message = user_message.strip()
        if not user_message:
            pass
        elif len(user_message) > MAX_MESSAGE_LENGTH:
            st.warning(f"That message is too long ({len(user_message)} characters). "
                       f"Keep it under {MAX_MESSAGE_LENGTH}.")
        else:
            logger.info(f"Chat message sent for activity {activity_key} ({len(user_message)} chars)")
            history.append({"role": "user", "content": user_message})

            with st.spinner("Reading the reference solution and writing a reply..."):
                try:
                    prompts = PromptManager(base_path=str(PROJECT_ROOT))
                    template = prompts.get_prompt("code_assistant_prompt")
                    reference_code = load_reference_code(activity_key)
                    current_instructions = load_current_instructions(activity_key)
                    system_content = prompts.format_prompt(
                        template["system"],
                        activity_title=activity["label"],
                        functions=", ".join(activity["functions"]),
                        reference_path=", ".join(activity["ref_files"]),
                        reference_code=reference_code,
                        current_instructions=current_instructions,
                    )

                    capped_history = history[-(MAX_HISTORY_TURN_PAIRS * 2):]
                    messages = [{"role": "system", "content": system_content}]
                    for turn in capped_history:
                        if turn["role"] == "user":
                            content = prompts.format_prompt(template["user_template"], message=turn["content"])
                        else:
                            content = turn["content"]
                        messages.append({"role": turn["role"], "content": content})

                    llm = LLMClient(base_path=str(PROJECT_ROOT))
                    reply = llm.generate(messages).strip()

                    if is_reference_dump(reply, reference_code):
                        logger.info(f"Reference-dump guard tripped for activity {activity_key}")
                        reply = (
                            "I can help with specific parts of the code, but I won't paste "
                            "the whole reference file. Try asking about one function "
                            "or one step."
                        )
                except Exception as e:
                    logger.error(f"LLM error for activity {activity_key}: {e}")
                    reply = f"Sorry, the model call failed: {e}"

            history.append({"role": "assistant", "content": reply})

    # Rendered last so it reflects this run's just-appended turn too - newest
    # pair (assistant reply on top, its user question right below it) first.
    with history_placeholder:
        for msg in reversed(history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
