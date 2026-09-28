"""Grounded response generation with citations - REFERENCE SOLUTION (Activity 2.3)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.indexing.vector_store import RetrievedChunk
from src.llm.llm_client import LLMClient
from src.llm.prompt_manager import PromptManager
from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

REFUSAL_TEXT = "I don't have a policy document that covers this."


def _generation_config(base_path: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml_config("config/ingestion_config.yaml", base_path=base_path)["generation"]


def build_context(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, ch in enumerate(chunks, start=1):
        src = ch.metadata.get("source", "unknown")
        ver = ch.metadata.get("version", "")
        tag = f"{src} v{ver}" if ver else src
        lines.append(f"[{i}] (source: {tag})\n{ch.content.strip()}")
    return "\n\n".join(lines)


def parse_citations(answer: str, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    markers = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for n in group.split(","):
            n = n.strip()
            if n.isdigit():
                markers.add(int(n))
    citations = []
    for n in sorted(markers):
        if 1 <= n <= len(chunks):
            ch = chunks[n - 1]
            citations.append({
                "marker": n,
                "source": ch.metadata.get("source", "unknown"),
                "version": ch.metadata.get("version", ""),
                "chunk_id": ch.metadata.get("chunk_id", ""),
            })
    return citations


def validate_answer(result: Dict[str, Any], config: Optional[Dict[str, Any]] = None,
                    base_path: Optional[str] = None) -> Optional[List[str]]:
    cfg = config or load_yaml_config("config/ingestion_config.yaml", base_path=base_path)["validation"]
    errors: List[str] = []
    for field in cfg.get("required_answer_fields", []):
        if field not in result:
            errors.append(f"Missing field: {field}")
    if not result.get("refused", False):
        min_cites = cfg.get("min_citations_when_answered", 1)
        if len(result.get("citations", [])) < min_cites:
            errors.append(f"Answered response has {len(result.get('citations', []))} citation(s), "
                          f"expected at least {min_cites}")
    return errors or None


def generate_answer(question: str, retrieved_chunks: List[RetrievedChunk],
                    llm: Optional[LLMClient] = None,
                    base_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = _generation_config(base_path=base_path)
    chunks = retrieved_chunks[: cfg.get("max_context_chunks", 5)]

    if not chunks and cfg.get("refuse_when_no_context", True):
        return {"answer": REFUSAL_TEXT, "citations": [], "retrieved_chunks": [], "refused": True}

    context = build_context(chunks)
    pm = PromptManager(base_path=base_path)
    prompt = pm.get_prompt("grounded_answer_prompt")
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": pm.format_prompt(prompt["user_template"],
                                                     question=question, context=context)},
    ]
    answer = (llm or LLMClient(base_path=base_path)).generate(messages).strip()

    refused = answer.lower().startswith(REFUSAL_TEXT.lower()) or not re.search(r"\[\d", answer)
    citations = [] if refused else parse_citations(answer, chunks)

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": [
            {
                "source": ch.metadata.get("source", "unknown"),
                "version": ch.metadata.get("version", ""),
                "chunk_id": ch.metadata.get("chunk_id", ""),
                "score": ch.score,
            }
            for ch in chunks
        ],
        "refused": refused,
    }
