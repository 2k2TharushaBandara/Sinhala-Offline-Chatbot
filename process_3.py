"""process_3.py — Prompt build + LLM inference + post-translate + references.

Implements pipeline steps:
6) Prompt build (English system prompt + translated context)
7) LLM inference (Ollama)
8) Post-translate en→si (local NLLB)
9) Output Sinhala answer + source references
"""

from __future__ import annotations

from typing import List


_SYSTEM = """You are a history teacher for Grade 6-11 students in Sri Lanka.
Rules:
1. Answer clearly and simply using ONLY the provided Context.
2. If the answer is not in the Context, output exactly: \"I do not have information for that.\"
3. No guessing, inference, or outside knowledge.
4. Do not list references."""


def build_context_en(chunks) -> str:
    parts = [(getattr(c, "text_en", "") or "").strip() for c in chunks]
    return "\n\n---\n\n".join(p for p in parts if p)


def refs_markdown(chunks) -> str:
    seen: set = set()
    lines: List[str] = []
    for c in chunks:
        meta = getattr(getattr(c, "doc", None), "metadata", {}) or {}
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            lines.append(f"- {source} | පිටුව {page}")
    return "\n".join(lines)


def build_prompt(*, context_en: str, question_en: str) -> str:
    return (
        f"{_SYSTEM}\n\n"
        "Context:\n"
        f"{(context_en or '').strip()}\n\n"
        "Question:\n"
        f"{(question_en or '').strip()}\n\n"
        "Answer:"
    )


def run_llm_and_translate(
    *,
    prompt: str,
    model_name: str,
    temperature: float,
    num_predict: int,
    timeout_sec: int,
    translate_fn,
    return_english: bool = False,
    debug: bool = False,
    log_fn=None,
) -> str | tuple[str, str]:
    """Invoke Ollama LLM (English) and translate answer back to Sinhala."""

    if debug and log_fn:
        log_fn("LLM", f"calling Ollama model={model_name!r} temp={temperature} num_predict={num_predict}")

    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(model=model_name, temperature=temperature, num_predict=num_predict, timeout=timeout_sec)
    answer_en = str(llm.invoke(prompt)).strip()

    if debug and log_fn:
        log_fn("LLM", f"LLM returned {len(answer_en)} chars")

    answer_si = translate_fn(answer_en, src="eng_Latn", tgt="sin_Sinh", max_new=260)
    answer_si = (answer_si or "").strip()

    if return_english:
        # Back-compat note: caller can request the English answer.
        return (answer_si, answer_en)

    return answer_si
