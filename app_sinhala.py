"""app_sinhala.py — Sinhala History Chatbot  (Fully Offline, Hybrid RAG)
═══════════════════════════════════════════════════════════════════════════════
Pipeline (Hybrid)
─────────────────
1)  Input               Sinhala question from user
2)  Sinhala retrieve     Sinhala FAISS index using Sinhala embeddings (LaBSE)
3)  English retrieve     Translate Sinhala→English, search English FAISS (BAAI)
4)  Select top chunks    Merge + rerank chunks from BOTH pipelines
5)  Translate context    Translate ONLY Sinhala chunks si→en (chunk 1..N)
6)  Prompt build         Strict English system prompt + merged English context
7)  LLM inference        Local Ollama
8)  Post-translate       en → si (local NLLB)
9)  Output               Sinhala answer + source references

Hard constraints
────────────────
• Fully offline  (TRANSFORMERS_OFFLINE=1)
• CPU only
• All local paths

Quick start
───────────
  cd Sinhala_Chatbot_V2
    ..\\.venv\\Scripts\\python.exe -m streamlit run app_sinhala.py

Env overrides (all optional)
─────────────────────────────
  RAG_MODEL            HistoryByQwen
  RAG_TEMPERATURE      0.05
  RAG_NUM_PREDICT      250
  RAG_TIMEOUT          180

    # Sinhala index + Sinhala embeddings
    RAG_SI_FAISS_DIR         ./faiss_index/Sinhala_FAISS
    RAG_SI_EMBEDDINGS_PATH   ./models/embeddings/local_labse_model

    # English index + English embeddings
    RAG_EN_FAISS_DIR         ./faiss_index/English_FAISS
    RAG_EN_EMBEDDINGS_PATH   ./models/embeddings/BAAI

  # Back-compat fallbacks (if RAG_SI_* are not set)
  RAG_FAISS_DIR        ./faiss_index/Sinhala_FAISS
  RAG_EMBEDDINGS_PATH  ./models/embeddings/local_labse_model

  RAG_NLLB_PATH        ./models/nllb

    # Thresholds
    RAG_MAX_DISTANCE        1.15   # Back-compat: Sinhala FAISS threshold
    RAG_SI_MAX_DISTANCE     1.15
    RAG_EN_MAX_DISTANCE     0.90
  RAG_KEEP             5      # final chunks sent to LLM
  RAG_BM25_K1          1.5    # BM25 saturation parameter
  RAG_BM25_B           0.75   # BM25 length normalisation

  RAG_DEBUG            1      # verbose CLI output
═══════════════════════════════════════════════════════════════════════════════

Notes
-----
- `app.py` is unchanged; this file is the Sinhala-first variant.
- Translation is intentionally sequential for selected chunks only.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

# ── Streamlit must not try to watch Transformers (needs torchvision) ─────────
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st

# ── Strict offline ────────────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Pipeline modules (split for maintainability)
from process_1 import retrieve_all_under_threshold, retrieve_all_under_threshold_en, score_and_rank, score_and_rank_en
from process_2 import apply_glossary_reverse, load_history_glossary, translate_chunks_only, translate_query_for_search
from process_3 import build_context_en, build_prompt, refs_markdown, run_llm_and_translate


# ═══════════════════════════════════════════════════════════════════════════════
# UI: Thinking box
# ═══════════════════════════════════════════════════════════════════════════════


def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _truncate_for_thinking(text: str, *, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    return t[:max_chars] + "…"


def _thinking_css() -> str:
    return """
<style>
.details-panel {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    margin: 0.7rem 0 1.0rem;
    overflow: hidden;
}
.details-panel summary {
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 0.85rem 0.95rem;
}
.details-panel summary::-webkit-details-marker { display: none; }
.details-left {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
}
.details-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #ececec;
    line-height: 1.2;
}
.details-sub {
    font-size: 0.80rem;
    color: #888;
    line-height: 1.2;
}
.details-chevron {
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #888;
    flex: 0 0 auto;
}
.details-chevron::before {
    content: "▾";
    font-size: 1.0rem;
    transform: rotate(-90deg);
    transition: transform 0.15s ease;
}
.details-panel[open] .details-chevron::before { transform: rotate(0deg); }
.details-body {
    border-top: 1px solid #2a2a2a;
    padding: 0.85rem 0.95rem;
}

.thinking-body {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.80rem;
    color: #a7a7a7;
    line-height: 1.55;
    white-space: pre-wrap;
}
.thinking-step {
  margin: 0.55rem 0;
  padding-left: 0.25rem;
  border-left: 3px solid #333;
}
.thinking-step .k {
  color: #e8a96a;
  font-weight: 600;
}
.thinking-ctx {
  margin-top: 0.35rem;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.thinking-ctx .col {
  background: #131313;
  border: 1px solid #242424;
  border-radius: 8px;
  padding: 0.55rem 0.65rem;
}
.thinking-ctx .hdr {
  color: #666;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  margin-bottom: 0.30rem;
}
.thinking-ctx .txt {
  color: #bdbdbd;
}
@media (max-width: 740px) {
  .thinking-ctx { grid-template-columns: 1fr; }
}

/* Sample question links (non-widget; avoids rerun cancellation while busy) */
.sample-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.sample-pill {
    display: block;
    background: #232323;
    border: 1px solid #333;
    border-radius: 999px;
    color: #c8c8c8;
    font-size: 0.82rem;
    padding: 0.35rem 0.9rem;
    text-decoration: none;
    transition: all 0.15s ease;
}
.sample-pill:hover {
    border-color: #c96a2c;
    color: #e8a96a;
}
.sample-pill.disabled {
    opacity: 0.40;
    pointer-events: none;
}
@media (max-width: 740px) {
    .sample-grid { grid-template-columns: 1fr; }
}
</style>
"""


def _get_query_param(name: str) -> Optional[str]:
        """Compat helper for Streamlit query params across versions."""
        try:
                qp = st.query_params  # type: ignore[attr-defined]
                val = qp.get(name)
                if isinstance(val, list):
                        return val[0] if val else None
                return val
        except Exception:
                qp = st.experimental_get_query_params()
                val = qp.get(name)
                if isinstance(val, list):
                        return val[0] if val else None
                return val


def _clear_query_params() -> None:
        try:
                st.query_params.clear()  # type: ignore[attr-defined]
        except Exception:
                st.experimental_set_query_params()


def _set_query_params(**params: str) -> None:
    """Compat helper to set query params across Streamlit versions.

    IMPORTANT: This *replaces* all query params (clears then sets).
    """
    clean: Dict[str, str] = {}
    for k, v in (params or {}).items():
        if v is None:
            continue
        sv = str(v).strip()
        if sv:
            clean[str(k)] = sv

    try:
        qp = st.query_params  # type: ignore[attr-defined]
        qp.clear()
        for k, v in clean.items():
            qp[k] = v
    except Exception:
        st.experimental_set_query_params(**clean)


def _ensure_state() -> None:
    defaults = {
        "messages": [],
        "busy": False,
        "pending_user_si": None,
        "thinking_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _thinking_append(kind: str, title: str, detail: str = "", *, ctx_si: str = "", ctx_en: str = "") -> None:
    st.session_state.thinking_log.append(
        {
            "kind": kind,
            "title": title,
            "detail": detail,
            "ctx_si": ctx_si,
            "ctx_en": ctx_en,
        }
    )


def _thinking_render(placeholder) -> None:
    entries: List[dict] = list(st.session_state.thinking_log or [])
    blocks: List[str] = []
    for e in entries:
        title = _escape_html(str(e.get("title", "")).strip())
        detail = _escape_html(str(e.get("detail", "")).strip())
        block = f"<div class='thinking-step'><span class='k'>{title}</span>"
        if detail:
            block += f"\n{detail}"
        ctx_si = (e.get("ctx_si") or "").strip()
        ctx_en = (e.get("ctx_en") or "").strip()
        if ctx_si or ctx_en:
            block += "<div class='thinking-ctx'>"
            if ctx_si:
                block += (
                    "<div class='col'><div class='hdr'>Sinhala</div>"
                    f"<div class='txt'>{_escape_html(ctx_si)}</div></div>"
                )
            if ctx_en:
                block += (
                    "<div class='col'><div class='hdr'>English</div>"
                    f"<div class='txt'>{_escape_html(ctx_en)}</div></div>"
                )
            block += "</div>"
        block += "</div>"
        blocks.append(block)
    body_html = "<div class='thinking-body'>" + ("\n".join(blocks) if blocks else "(no steps yet)") + "</div>"

    open_attr = ""  # keep collapsed by default; client-side toggle doesn't rerun
    subtitle = "Model is thinking — expand to view thoughts" if st.session_state.busy else "Expand to view model thoughts"
    html = (
        f"<details class='details-panel'{open_attr}>"
        "  <summary>"
        "    <div class='details-left'>"
        "      <div class='details-title'>Thoughts</div>"
        f"      <div class='details-sub'>{_escape_html(subtitle)}</div>"
        "    </div>"
        "    <div class='details-chevron'></div>"
        "  </summary>"
        f"  <div class='details-body'>{body_html}</div>"
        "</details>"
    )
    placeholder.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve(p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(_script_dir(), p))


# ── Rich CLI logger ───────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_DIM = "\033[2m"


def _banner(title: str) -> None:
    width = 72
    print(f"\n{_BOLD}{_CYAN}{'━' * width}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'━' * width}{_RESET}")


def _log(step: str, msg: str, color: str = _RESET, debug: bool = True) -> None:
    if debug:
        ts = time.strftime("%H:%M:%S")
        print(f"{_DIM}[{ts}]{_RESET} {_BOLD}{color}[{step}]{_RESET}  {msg}")


def _log_chunk(idx: int, doc, distance: float, scores: Dict[str, float], debug: bool = True) -> None:
    if not debug:
        return
    meta = getattr(doc, "metadata", {}) or {}
    source = meta.get("source", "?")
    page = meta.get("page", "?")
    content = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
    preview = content[:120] + ("…" if len(content) > 120 else "")
    combined = scores.get("combined", 0.0)
    bm25 = scores.get("bm25", 0.0)
    bigram = scores.get("bigram", 0.0)
    phrase = scores.get("phrase", 0.0)

    print(
        f"\n  {_BOLD}{_GREEN}Chunk #{idx}{_RESET}  "
        f"{_DIM}src={source!r}  page={page}  dist={distance:.4f}{_RESET}\n"
        f"  {_YELLOW}combined={combined:.4f}  bm25={bm25:.4f}  "
        f"bigram={bigram:.4f}  phrase={phrase:.4f}{_RESET}\n"
        f"  {_DIM}\"{preview}\"{_RESET}"
    )
def _truncate_for_translation(text: str, *, max_chars: int = 1500) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "…"


# ═══════════════════════════════════════════════════════════════════════════════
# Cached heavy resources
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def load_nllb(path: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()

    def translate(text: str, src: str, tgt: str, max_new: int = 220) -> str:
        tok.src_lang = src
        enc = tok(text, return_tensors="pt")
        forced_id = tok.convert_tokens_to_ids(tgt)
        max_len = int(enc["input_ids"].shape[1] + max_new)
        with torch.inference_mode():
            out = model.generate(**enc, forced_bos_token_id=forced_id, max_length=max_len)
        return tok.batch_decode(out, skip_special_tokens=True)[0]

    return translate


@st.cache_data(show_spinner=False)
def load_glossary(path: str) -> Dict[str, str]:
    try:
        return load_history_glossary(path)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def load_embeddings(path: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=path, encode_kwargs={"normalize_embeddings": True})


@st.cache_resource(show_spinner=False)
def load_faiss(faiss_dir: str, embeddings):
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Claude-like CSS (copied from app.py)
# ═══════════════════════════════════════════════════════════════════════════════

_CLAUDE_CSS = """
<style>
/* ── Base & fonts ─────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #1a1a1a !important;
    color: #ececec !important;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif !important;
}

/* ── Hide Streamlit chrome (keep sidebar toggle functional) ───────────── */
#MainMenu, footer { display: none !important; }

/* Keep the header/toolbar container visible so the sidebar toggle works.
   Only hide the right-side action widgets (profile/deploy/etc). */
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbarActions"],
[data-testid="stToolbarActionElements"],
[data-testid="stHeaderActionElements"],
[data-testid="stStatusWidget"],
[data-testid="stProfileButton"],
[data-testid="stDeployButton"] { display: none !important; }

header, [data-testid="stHeader"] {
    background: #1a1a1a !important;
    border-bottom: none !important;
    box-shadow: none !important;
    height: 2.25rem !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #2a2a2a !important;
    min-width: 220px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] * { color: #c8c8c8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #666 !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stSidebar"] hr { border-color: #2a2a2a !important; }

/* ── Chat history list (sidebar) ───────────────────────────────────────── */
.chatlist {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.sidebar-app-title {
    font-size: 0.95rem;
    font-weight: 650;
    color: #ececec !important;
    margin: 0.2rem 0 0.7rem !important;
}
.chatrow {
    display: grid;
    grid-template-columns: 28px 1fr;
    gap: 10px;
    align-items: center;
}
.chatdel, .chatlink {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
}
.chatdel {
    width: 28px;
    height: 28px;
    border-radius: 10px;
    background: #141414;
    border: 1px solid #2a2a2a;
    color: #777;
    transition: all 0.15s ease;
}
.chatdel:hover { border-color: #3a3a3a; color: #aaa; }
.chatlink {
    width: 100%;
    padding: 10px 12px;
    border-radius: 999px;
    background: #141414;
    border: 1px solid #2a2a2a;
    color: #cfcfcf;
    justify-content: flex-start;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    transition: all 0.15s ease;
}
.chatlink:hover { border-color: #c96a2c; color: #e8a96a; }
.chatlink.active {
    background: linear-gradient(135deg, #1c2f57, #2b3b63);
    border-color: #335aa2;
    color: #fff;
}
.chatlink.disabled, .chatdel.disabled {
    opacity: 0.45;
    cursor: not-allowed;
    pointer-events: none;
}
.newchat {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    padding: 10px 12px;
    border-radius: 999px;
    background: linear-gradient(135deg,#c96a2c,#e07a38);
    border: 1px solid #c96a2c;
    color: #fff;
    text-decoration: none;
    margin-bottom: 10px;
    font-weight: 600;
    transition: all 0.15s ease;
}
.newchat:hover { filter: brightness(1.05); }
.newchat.disabled { opacity: 0.45; pointer-events: none; }

/* ── Main content area ────────────────────────────────────────────────── */
.main .block-container {
    max-width: 760px !important;
    margin: 0 auto !important;
    padding: 1.5rem 1.5rem 8rem !important;
}

/* ── Page header ──────────────────────────────────────────────────────── */
.claude-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.6rem 0 1.2rem;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 1.4rem;
}
.claude-header-logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #c96a2c 0%, #e8a96a 100%);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.claude-header-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #ececec;
    letter-spacing: -0.01em;
}
.claude-header-sub {
    font-size: 0.72rem;
    color: #666;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Chat bubbles ─────────────────────────────────────────────────────── */
.chat-row {
    display: flex;
    margin-bottom: 1.4rem;
    gap: 12px;
    align-items: flex-start;
}
.chat-row.user  { flex-direction: row-reverse; }
.chat-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar-user      { background: #3a3a3a; }
.avatar-assistant { background: linear-gradient(135deg,#c96a2c,#e8a96a); }
.chat-bubble {
    max-width: 82%;
    padding: 0.75rem 1rem;
    border-radius: 14px;
    line-height: 1.65;
    font-size: 0.93rem;
    color: #ececec;
}
.chat-row.user .chat-bubble {
    background: #2d2d2d;
    border-top-right-radius: 4px;
    text-align: right;
}
.chat-row.assistant .chat-bubble {
    background: #212121;
    border: 1px solid #2e2e2e;
    border-top-left-radius: 4px;
}
.chat-bubble strong { color: #e8a96a; }
.chat-bubble a      { color: #e8a96a; }

/* ── Typing indicator ─────────────────────────────────────────────────── */
.typing-dots span {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #666;
    margin: 0 2px;
    animation: blink 1.2s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40%           { opacity: 1;   transform: scale(1); }
}

/* ── Step checklist ───────────────────────────────────────────────────── */
.step-list {
    font-size: 0.78rem;
    font-family: monospace;
    color: #888;
    background: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.6rem;
    line-height: 1.85;
}
.step-list .done  { color: #6ec97e; }
.step-list .pending { color: #555; }

/* ── Input bar ────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: #2a2a2a !important;
    color: #ececec !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
    font-family: inherit !important;
    font-size: 0.93rem !important;
    caret-color: #e8a96a !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #c96a2c !important;
    box-shadow: 0 0 0 2px rgba(201,106,44,0.18) !important;
    outline: none !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg,#c96a2c,#e07a38) !important;
    border-radius: 8px !important;
    color: #fff !important;
}
[data-testid="stChatInput"] button:disabled {
    opacity: 0.4 !important;
    cursor: not-allowed !important;
}

/* ── Sample question pills ─────────────────────────────────────────────── */
.pill-btn button {
    background: #232323 !important;
    border: 1px solid #333 !important;
    border-radius: 999px !important;
    color: #c8c8c8 !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.9rem !important;
    transition: all 0.15s ease;
}
.pill-btn button:hover {
    border-color: #c96a2c !important;
    color: #e8a96a !important;
}

/* ── Expander ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #888 !important;
    font-size: 0.82rem !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Chat history (persistent sessions)
# ═══════════════════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _history_path() -> str:
    return os.path.join(_script_dir(), "chat_history.json")


def _load_history() -> List[dict]:
    path = _history_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict)]
        return []
    except Exception:
        return []


def _save_history(chats: List[dict]) -> None:
    path = _history_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chats, f, ensure_ascii=False, indent=2)
    except Exception:
        # Best-effort persistence; app should still run offline.
        return


def _make_chat_title(messages: List[dict]) -> str:
    # Use first user message as title (first 6 words); fall back to timestamp.
    for m in messages or []:
        if (m.get("role") == "user") and (m.get("content") or "").strip():
            t = str(m.get("content") or "").strip()
            t = " ".join(t.split())
            words = t.split()
            if not words:
                break
            title = " ".join(words[:6])
            if len(words) > 6:
                title += "…"
            return title
    return f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"


def _get_chat(chats: List[dict], chat_id: str) -> Optional[dict]:
    for c in chats:
        if str(c.get("id")) == str(chat_id):
            return c
    return None


def _ensure_active_chat() -> None:
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None

    # If a chat is explicitly requested via URL param, honor it (refresh-safe).
    qp_chat = (_get_query_param("chat") or "").strip()
    if qp_chat:
        chats = _load_history()
        if _get_chat(chats, qp_chat):
            st.session_state.active_chat_id = str(qp_chat)
            return

    chats = _load_history()
    active_id = str(st.session_state.active_chat_id or "").strip()
    if active_id and _get_chat(chats, active_id):
        return

    # Prefer re-opening the most recently updated chat if it exists.
    if chats:
        chats_sorted = sorted(
            chats,
            key=lambda c: str(c.get("updated_at") or c.get("created_at") or ""),
            reverse=True,
        )
        st.session_state.active_chat_id = str(chats_sorted[0].get("id"))
        return

    # No chats exist yet → create the first one.
    new_id = _create_new_chat()
    st.session_state.active_chat_id = new_id


def _create_new_chat() -> str:
    chats = _load_history()
    new_id = str(uuid.uuid4())
    chats.insert(
        0,
        {
            "id": new_id,
            "title": "New chat",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "messages": [],
        },
    )
    _save_history(chats)
    return new_id


def _delete_chat(chat_id: str) -> None:
    chats = _load_history()
    chats = [c for c in chats if str(c.get("id")) != str(chat_id)]
    _save_history(chats)


def _persist_active_messages(messages: List[dict]) -> None:
    chat_id = str(st.session_state.active_chat_id or "")
    if not chat_id:
        return
    chats = _load_history()
    chat = _get_chat(chats, chat_id)
    if not chat:
        # Recreate if missing
        chat = {
            "id": chat_id,
            "title": "New chat",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "messages": [],
        }
        chats.insert(0, chat)

    chat["messages"] = list(messages or [])
    chat["updated_at"] = _now_iso()
    # Update title from first user message once available
    title = _make_chat_title(chat["messages"])
    if title:
        chat["title"] = title

    # Move active chat to top
    chats = [c for c in chats if str(c.get("id")) != str(chat_id)]
    chats.insert(0, chat)
    _save_history(chats)


def _chat_row(role: str, content: str) -> str:
    is_user = role == "user"
    row_cls = "user" if is_user else "assistant"
    av_cls = "avatar-user" if is_user else "avatar-assistant"
    icon = "👤" if is_user else "✦"
    content_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
    content_html = content_html.replace("\n", "<br>")
    return (
        f'<div class="chat-row {row_cls}">' f'  <div class="chat-avatar {av_cls}">{icon}</div>' f'  <div class="chat-bubble">{content_html}</div>' f"</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    st.set_page_config(
        page_title="History ගුරු",
        page_icon="📜",
        layout="centered",
    )
    st.markdown(_CLAUDE_CSS, unsafe_allow_html=True)
    st.markdown(_thinking_css(), unsafe_allow_html=True)

    _ensure_state()
    _ensure_active_chat()

    # ── Handle chat navigation actions (query params) ─────────────────────
    # Only when not busy, to avoid interrupting in-flight generation.
    if not st.session_state.busy:
        qp_new = (_get_query_param("new") or "").strip()
        qp_chat = (_get_query_param("chat") or "").strip()
        qp_del = (_get_query_param("del") or "").strip()

        if qp_new:
            new_id = _create_new_chat()
            st.session_state.active_chat_id = new_id
            st.session_state.messages = []
            st.session_state.thinking_log = []
            st.session_state.pending_user_si = None
            _set_query_params(chat=new_id)
            st.rerun()

        if qp_del:
            _delete_chat(qp_del)
            if str(st.session_state.active_chat_id or "") == qp_del:
                st.session_state.active_chat_id = None
                _ensure_active_chat()
                # Load messages for new active chat
                chats2 = _load_history()
                active_chat = _get_chat(chats2, str(st.session_state.active_chat_id or ""))
                st.session_state.messages = list((active_chat or {}).get("messages") or [])
            _set_query_params(chat=str(st.session_state.active_chat_id or "").strip())
            st.rerun()

        # `chat` is not a one-time action param; it may stay in the URL.
        # Only switch chats if it's different from what we already have.
        current_id = str(st.session_state.active_chat_id or "").strip()
        if qp_chat and (qp_chat != current_id):
            chats2 = _load_history()
            target = _get_chat(chats2, qp_chat)
            if target:
                st.session_state.active_chat_id = qp_chat
                st.session_state.thinking_log = []
                st.session_state.pending_user_si = None
                st.session_state.messages = list((target.get("messages") or []))

    debug = _env_bool("RAG_DEBUG", True)

    # Paths
    si_faiss_dir = _resolve(
        os.getenv(
            "RAG_SI_FAISS_DIR",
            os.getenv("RAG_FAISS_DIR", "./faiss_index/Sinhala_FAISS"),
        )
    )
    si_embeddings_path = _resolve(
        os.getenv(
            "RAG_SI_EMBEDDINGS_PATH",
            os.getenv("RAG_EMBEDDINGS_PATH", "./models/embeddings/local_labse_model"),
        )
    )

    en_faiss_dir = _resolve(os.getenv("RAG_EN_FAISS_DIR", "./faiss_index/English_FAISS"))
    en_embeddings_path = _resolve(os.getenv("RAG_EN_EMBEDDINGS_PATH", "./models/embeddings/BAAI"))
    nllb_path = _resolve(os.getenv("RAG_NLLB_PATH", "./models/nllb"))
    glossary_path = _resolve(os.getenv("RAG_GLOSSARY_PATH", "./history_glossary.txt"))

    # Config
    si_max_distance = _env_float("RAG_SI_MAX_DISTANCE", _env_float("RAG_MAX_DISTANCE", 1.15))
    en_max_distance = _env_float("RAG_EN_MAX_DISTANCE", 0.90)
    keep = _env_int("RAG_KEEP", 5)
    bm25_k1 = _env_float("RAG_BM25_K1", 1.5)
    bm25_b = _env_float("RAG_BM25_B", 0.75)
    model_name = os.getenv("RAG_MODEL", "HistoryByQwen")
    temperature = _env_float("RAG_TEMPERATURE", 0.05)
    num_predict = _env_int("RAG_NUM_PREDICT", 250)
    timeout_sec = _env_int("RAG_TIMEOUT", 180)

    think_max_chars = _env_int("RAG_THINK_MAX_CHARS", 700)

    # ── Sidebar: Chats (history) ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div class='sidebar-app-title'>History ගුරු</div>", unsafe_allow_html=True)
        st.markdown("### Chats")

        chats = _load_history()
        active_id = str(st.session_state.active_chat_id or "")

        disabled_cls = " disabled" if st.session_state.busy else ""
        new_href = "?new=1"
        new_link = f"<a class='newchat{disabled_cls}' href='{new_href}' target='_self' onclick=\"window.location.href='{new_href}'; return false;\">＋ New chat</a>"

        rows: List[str] = [new_link, "<div class='chatlist'>"]
        for c in chats:
            cid = str(c.get("id") or "")
            msgs = c.get("messages") if isinstance(c.get("messages"), list) else []
            if msgs:
                title_txt = _make_chat_title(list(msgs))
            else:
                title_txt = str(c.get("title") or "New chat")
            title = _escape_html(title_txt)
            is_active = " active" if cid == active_id else ""
            href_chat = f"?chat={quote(cid)}"
            # Preserve the current chat in the URL while deleting another chat.
            href_del = f"?chat={quote(active_id)}&del={quote(cid)}" if active_id else f"?del={quote(cid)}"

            if st.session_state.busy:
                del_el = "<span class='chatdel disabled'>🗑</span>"
                chat_el = f"<span class='chatlink{is_active} disabled'>{title}</span>"
            else:
                del_el = (
                    f"<a class='chatdel' href='{href_del}' target='_self' "
                    f"onclick=\"window.location.href='{href_del}'; return false;\">🗑</a>"
                )
                chat_el = (
                    f"<a class='chatlink{is_active}' href='{href_chat}' target='_self' "
                    f"onclick=\"window.location.href='{href_chat}'; return false;\">{title}</a>"
                )

            rows.append(f"<div class='chatrow'>{del_el}{chat_el}</div>")
        rows.append("</div>")
        st.markdown("\n".join(rows), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<small style='color:#555'>Fully offline · CPU only · NLLB · Ollama</small>",
            unsafe_allow_html=True,
        )

    # ── Validate local folders ────────────────────────────────────────────────
    for label, path in [
        ("si_faiss", si_faiss_dir),
        ("si_embeddings", si_embeddings_path),
        ("en_faiss", en_faiss_dir),
        ("en_embeddings", en_embeddings_path),
        ("nllb", nllb_path),
    ]:
        if not os.path.isdir(path):
            st.error(f"❌  Missing **{label}** folder: `{path}`")
            st.stop()

    # ── Load resources ────────────────────────────────────────────────────────
    with st.spinner("Loading NLLB translation model…"):
        translate = load_nllb(nllb_path)
    glossary = load_glossary(glossary_path)
    with st.spinner("Loading Sinhala embeddings model…"):
        embeddings_si = load_embeddings(si_embeddings_path)
    with st.spinner("Loading Sinhala FAISS index…"):
        vector_db_si = load_faiss(si_faiss_dir, embeddings_si)

    with st.spinner("Loading English embeddings model…"):
        embeddings_en = load_embeddings(en_embeddings_path)
    with st.spinner("Loading English FAISS index…"):
        vector_db_en = load_faiss(en_faiss_dir, embeddings_en)

    # ── Load active chat messages into session state (once per run) ─────────-
    if not st.session_state.messages:
        chats = _load_history()
        active_chat = _get_chat(chats, str(st.session_state.active_chat_id or ""))
        if active_chat and isinstance(active_chat.get("messages"), list):
            st.session_state.messages = list(active_chat.get("messages") or [])

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="claude-header">'
        '  <div class="claude-header-logo">✦</div>'
        '  <div>'
        '    <div class="claude-header-title">History ගුරු</div>'
        '    <div class="claude-header-sub">Offline RAG · Grades 6–11 · Sri Lanka</div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Prefill from query param (non-widget sample clicks) ─────────────────
    # This is intentionally ignored while busy to avoid cancelling an in-flight run.
    if not st.session_state.busy:
        qp_prefill = (_get_query_param("prefill") or _get_query_param("q") or "").strip()
        if qp_prefill:
            st.session_state["_prefill"] = qp_prefill
            # Remove the prefill param but keep the active chat id.
            _set_query_params(chat=str(st.session_state.active_chat_id or "").strip())

    # ── Sample questions ──────────────────────────────────────────────────────
    samples = [
        "අනුරාධපුර යුගය යනු කුමක්ද?",
        "ලන්දේසීන් විසින් ලංකාවේ වෙළෙඳ කටයුතු මෙහෙයවීම සඳහා පිහිටුවා ගත් සමාගමේ නම කුමක්ද?",
        "ඉතිහාසය හැදෑරීමට අපට උපකාරී වන සාහිත්‍ය මූලාශ්‍ර සහ පුරාවිද්‍යාත්මක මූලාශ්‍ර පිළිබඳව විස්තර කරන්න.",
        "'අහසින් වැටෙන එක දිය බිඳක්වත් ලෝකෝපකාරයෙන් තොරව මුහුදට නොයේවා' යන ප්‍රසිද්ධ ප්‍රකාශය කළ රජතුමා කවුද?",
        "පූර්ව රාජ්‍ය සමයේ පාලනයේ ස්වරූපය කුමක්ද?",
    ]
    # NOTE: Streamlit expanders/buttons can cancel a running script by triggering a rerun.
    # We render this as a client-side <details> panel. While busy, the links are disabled.
    open_attr = " open" if not st.session_state.messages else ""
    pills: List[str] = []
    for q in samples:
        if st.session_state.busy:
            pills.append(f"<span class='sample-pill disabled'>{_escape_html(q)}</span>")
        else:
            active_id = str(st.session_state.active_chat_id or "").strip()
            href = f"?chat={quote(active_id)}&prefill={quote(q)}" if active_id else f"?prefill={quote(q)}"
            # Force same-tab navigation (some browsers/Streamlit shells may otherwise open a new tab)
            pills.append(
                f"<a class='sample-pill' href='{href}' target='_self' "
                f"onclick=\"window.location.href='{href}'; return false;\">{_escape_html(q)}</a>"
            )
    samples_sub = "Disabled while generating" if st.session_state.busy else "Click a question to fill the input"
    samples_html = (
        f"<details class='details-panel'{open_attr}>"
        "  <summary>"
        "    <div class='details-left'>"
        "      <div class='details-title'>Sample questions</div>"
        f"      <div class='details-sub'>{_escape_html(samples_sub)}</div>"
        "    </div>"
        "    <div class='details-chevron'></div>"
        "  </summary>"
        "  <div class='details-body'>"
        "    <div class='sample-grid'>"
        f"      {' '.join(pills)}"
        "    </div>"
        "  </div>"
        "</details>"
    )
    st.markdown(samples_html, unsafe_allow_html=True)

    # ── Render conversation history ───────────────────────────────────────────
    for m in st.session_state.messages:
        st.markdown(_chat_row(m["role"], m["content"]), unsafe_allow_html=True)

    # ── Thinking panel (Gemini-like, client-side toggle) ─────────────────────
    thinking_placeholder = st.empty()
    _thinking_render(thinking_placeholder)

    # ── Chat input ───────────────────────────────────────────────────────────
    prefill = st.session_state.pop("_prefill", None)

    if st.session_state.busy:
        st.info("⏳  Generating answer — please wait…", icon="🔄")
        user_input = None
    else:
        user_input = st.chat_input(
            placeholder="සිංහලෙන් ඔබේ ප්‍රශ්නය ලියන්න…",
            disabled=False,
        )
        if prefill and not user_input:
            user_input = prefill

    # ── Hard lock: store pending message then rerun with busy=True ───────────
    if (not st.session_state.busy) and user_input:
        user_si = str(user_input).strip()
        if not user_si:
            return

        # Ignore submits while a pending message already exists
        if st.session_state.pending_user_si:
            return

        st.session_state.pending_user_si = user_si
        st.session_state.busy = True
        st.session_state.thinking_log = []
        _thinking_append("step", "Input", "Received Sinhala question")

        # Show user bubble immediately via history + rerun
        st.session_state.messages.append({"role": "user", "content": user_si})
        _persist_active_messages(st.session_state.messages)
        st.rerun()

    # If not busy and nothing submitted, just idle.
    if (not st.session_state.busy) and (not st.session_state.pending_user_si):
        return

    # Busy mode: process exactly one pending message
    user_si = str(st.session_state.pending_user_si or "").strip()
    if not user_si:
        st.session_state.pending_user_si = None
        st.session_state.busy = False
        st.rerun()

    answer_placeholder = st.empty()
    answer_placeholder.markdown(
        '<div class="chat-row assistant">'
        '  <div class="chat-avatar avatar-assistant">✦</div>'
        '  <div class="chat-bubble">'
        '    <div class="typing-dots">'
        '      <span></span><span></span><span></span>'
        '    </div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    t_wall = time.perf_counter()

    try:
        if debug:
            _banner(f"NEW QUERY  —  {time.strftime('%H:%M:%S')}")
            _log("INPUT", f"Sinhala: {user_si!r}", _YELLOW, debug)

        _thinking_append("step", "Step 1", "Retrieve Sinhala chunks from Sinhala FAISS")
        _thinking_render(thinking_placeholder)

        # ── Step 1: Sinhala FAISS retrieval (no pre-translation) ─────────────
        t0 = time.perf_counter()
        raw_chunks_si = retrieve_all_under_threshold(
            vector_db=vector_db_si,
            query_si=user_si,
            max_distance=si_max_distance,
            debug=debug,
        )
        _log(
            "STEP 1",
            f"retrieved {len(raw_chunks_si)} Sinhala chunks  ({time.perf_counter()-t0:.2f}s)",
            _CYAN,
            debug,
        )

        _thinking_append("step", "Step 1", f"Retrieved {len(raw_chunks_si)} Sinhala candidates")
        _thinking_render(thinking_placeholder)

        # ── Step 2: Translate query + English FAISS retrieval ───────────────
        _thinking_append(
            "step",
            "Step 2",
            "Translate Sinhala query → English and retrieve English chunks from English FAISS",
        )
        _thinking_render(thinking_placeholder)

        t0 = time.perf_counter()
        query_en, query_repls = translate_query_for_search(
            query_si=user_si,
            glossary=glossary,
            translate_fn=translate,
        )
        raw_chunks_en = retrieve_all_under_threshold_en(
            vector_db=vector_db_en,
            query_en=query_en,
            max_distance=en_max_distance,
            debug=debug,
        )
        _log(
            "STEP 2",
            f"query_en chars={len(query_en)} retrieved {len(raw_chunks_en)} EN chunks  ({time.perf_counter()-t0:.2f}s)",
            _CYAN,
            debug,
        )

        _thinking_append(
            "ctx",
            "Query translation",
            f"Sinhala → English (glossary hits: {len(query_repls)})",
            ctx_si=_truncate_for_thinking(user_si, max_chars=think_max_chars),
            ctx_en=_truncate_for_thinking(query_en, max_chars=think_max_chars),
        )
        _thinking_append("step", "Step 2", f"Retrieved {len(raw_chunks_en)} English candidates")
        _thinking_render(thinking_placeholder)

        # ── Step 3: Rank Sinhala + English, merge, select best ──────────────
        _thinking_append(
            "step",
            "Step 3",
            "Rank Sinhala+English candidates and select best merged top-K",
        )
        _thinking_render(thinking_placeholder)
        t0 = time.perf_counter()

        candidate_keep = max(int(keep) * 2, int(keep))
        top_si = score_and_rank(
            chunks=raw_chunks_si,
            query_si=user_si,
            keep=candidate_keep,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            max_distance=si_max_distance,
            debug=debug,
        )
        top_en = score_and_rank_en(
            chunks=raw_chunks_en,
            query_en=query_en,
            keep=candidate_keep,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            max_distance=en_max_distance,
            debug=debug,
        )

        def _score_line(c, idx: int) -> str:
            meta = getattr(getattr(c, "doc", None), "metadata", {}) or {}
            src = meta.get("source", "Unknown")
            page = meta.get("page", "?")
            origin = (getattr(c, "origin", "") or "?").upper()
            return (
                f"{idx:>2}. ({origin}) score={getattr(c,'combined',0.0):.4f} "
                f"dist={getattr(c,'distance',0.0):.4f} bm25={getattr(c,'bm25',0.0):.3f} "
                f"bg={getattr(c,'bigram',0.0):.3f} phr={getattr(c,'phrase',0.0):.3f} "
                f"| {src} p{page}"
            )

        # Console + thinking: show how chunks were selected (scores)
        if debug:
            _log("RANK_SI", f"Top Sinhala candidates (showing {min(5, len(top_si))}/{len(top_si)})", _MAGENTA, debug)
            for i, c in enumerate(top_si[:5], 1):
                _log("RANK_SI", _score_line(c, i), _DIM, debug)
            _log("RANK_EN", f"Top English candidates (showing {min(5, len(top_en))}/{len(top_en)})", _MAGENTA, debug)
            for i, c in enumerate(top_en[:5], 1):
                _log("RANK_EN", _score_line(c, i), _DIM, debug)

        _thinking_append(
            "step",
            "Ranked candidates (Sinhala)",
            "\n".join(_score_line(c, i) for i, c in enumerate(top_si[:5], 1))
            or "(no Sinhala candidates)",
        )
        _thinking_append(
            "step",
            "Ranked candidates (English)",
            "\n".join(_score_line(c, i) for i, c in enumerate(top_en[:5], 1))
            or "(no English candidates)",
        )
        _thinking_render(thinking_placeholder)

        def _dedupe(chunks):
            seen = set()
            out = []
            for c in chunks:
                meta = getattr(getattr(c, "doc", None), "metadata", {}) or {}
                key = (
                    meta.get("source", ""),
                    meta.get("page", ""),
                    (getattr(getattr(c, "doc", None), "page_content", "") or "")[:200],
                    getattr(c, "origin", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
            return out

        merged = _dedupe(top_si + top_en)
        merged.sort(key=lambda c: float(getattr(c, "combined", 0.0)), reverse=True)
        top_chunks = merged[: int(keep)]

        si_count = sum(1 for c in top_chunks if getattr(c, "origin", "") == "si")
        en_count = sum(1 for c in top_chunks if getattr(c, "origin", "") == "en")

        _log(
            "STEP 3",
            f"selected {len(top_chunks)} merged chunks (SI={si_count}, EN={en_count})  ({time.perf_counter()-t0:.2f}s)",
            _CYAN,
            debug,
        )
        _thinking_append(
            "step",
            "Step 3",
            f"Selected {len(top_chunks)} chunks (SI={si_count}, EN={en_count})",
        )

        if debug:
            _log("MERGED", f"Merged top-{len(top_chunks)} chunks (by combined score)", _MAGENTA, debug)
            for i, c in enumerate(top_chunks, 1):
                _log("MERGED", _score_line(c, i), _DIM, debug)

        _thinking_append(
            "step",
            "Selected merged chunks",
            "\n".join(_score_line(c, i) for i, c in enumerate(top_chunks, 1)) or "(none)",
        )
        _thinking_render(thinking_placeholder)

        if not top_chunks:
            answer_placeholder.markdown(
                _chat_row("assistant", "⚠️ No relevant chunks found under the current thresholds."),
                unsafe_allow_html=True,
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": "⚠️ No relevant chunks found under the current thresholds."}
            )
            return

        # ── Step 4: Translate ONLY Sinhala chunks (queue) ───────────────────
        _thinking_append(
            "step",
            "Step 4",
            "Translate selected Sinhala chunks si→en (queue: chunk 1..N)",
        )
        _thinking_render(thinking_placeholder)

        t0 = time.perf_counter()
        si_chunks = [c for c in top_chunks if getattr(c, "origin", "") == "si"]
        en_chunks = [c for c in top_chunks if getattr(c, "origin", "") == "en"]

        applied_chunks = translate_chunks_only(
            chunks=si_chunks,
            glossary=glossary,
            translate_fn=translate,
            truncate_fn=lambda s: _truncate_for_translation(s),
            debug=debug,
        )
        # English chunks already English; populate text_en directly.
        for c in en_chunks:
            text = (getattr(c.doc, "page_content", "") or "").strip()
            c.text_en = _truncate_for_translation(text)

        _log(
            "STEP 4",
            f"translated {len(si_chunks)} Sinhala chunks (glossary hits: {len(applied_chunks)})  ({time.perf_counter()-t0:.2f}s)",
            _CYAN,
            debug,
        )

        # Render merged context (visuals)
        for i, c in enumerate(top_chunks, 1):
            raw_text = (getattr(c.doc, "page_content", "") or "").strip()
            meta = getattr(c.doc, "metadata", {}) or {}
            source = meta.get("source", "Unknown")
            page = meta.get("page", "?")
            origin = (getattr(c, "origin", "") or "?").upper()
            ctx_si = ""
            if getattr(c, "origin", "") == "si":
                ctx_si = _truncate_for_thinking(_truncate_for_translation(raw_text), max_chars=think_max_chars)
            _thinking_append(
                "ctx",
                f"Chunk {i}/{len(top_chunks)} ({origin})",
                f"{source} | page {page} | score={getattr(c,'combined',0.0):.4f} dist={getattr(c,'distance',0.0):.4f} bm25={getattr(c,'bm25',0.0):.3f} bg={getattr(c,'bigram',0.0):.3f} phr={getattr(c,'phrase',0.0):.3f}",
                ctx_si=ctx_si,
                ctx_en=_truncate_for_thinking(getattr(c, "text_en", "") or "", max_chars=think_max_chars),
            )
            _thinking_render(thinking_placeholder)

        # ── Step 5: Build English prompt using merged English context ───────
        _thinking_append("step", "Step 5", "Build prompt from translated context")
        _thinking_render(thinking_placeholder)
        context_en = build_context_en(top_chunks)
        prompt = build_prompt(context_en=context_en, question_en=query_en)
        _log("STEP 5", f"prompt chars={len(prompt)}  context_chars={len(context_en)}", _CYAN, debug)
        if debug:
            print(f"{_DIM}{'─'*60}")
            print(prompt[:600] + ("…" if len(prompt) > 600 else ""))
            print(f"{'─'*60}{_RESET}")

        _thinking_append("step", "Step 5", f"Prompt ready (context chars={len(context_en)})")
        _thinking_render(thinking_placeholder)

        # ── Step 6: LLM inference + back-translation ─────────────────────────
        _thinking_append("step", "Step 6", f"Ollama inference + back-translate ({model_name})")
        _thinking_render(thinking_placeholder)
        t0 = time.perf_counter()
        from langchain_ollama import OllamaLLM

        if debug:
            _log(
                "STEP 6",
                f"calling Ollama model={model_name!r}  temp={temperature}  num_predict={num_predict}",
                _CYAN,
                debug,
            )

        llm = OllamaLLM(model=model_name, temperature=temperature, num_predict=num_predict, timeout=timeout_sec)
        answer_en = str(llm.invoke(prompt)).strip()

        # Apply glossary to the LLM English output BEFORE back-translation.
        answer_en_gloss, gloss_repls = apply_glossary_reverse(answer_en, glossary)
        answer_si = translate(answer_en_gloss, src="eng_Latn", tgt="sin_Sinh", max_new=260)
        _log("STEP 6", f"LLM + back-translate done  ({time.perf_counter()-t0:.2f}s)", _GREEN, debug)

        if debug:
            _log("LLM_EN", f"Raw English answer ({len(answer_en)} chars):", _MAGENTA, debug)
            _log("LLM_EN", answer_en, _DIM, debug)

        _thinking_append(
            "ctx",
            "LLM answer (raw English)",
            "Model output before en→si translation",
            ctx_en=_truncate_for_thinking(answer_en, max_chars=think_max_chars),
        )
        if gloss_repls:
            # Keep this compact; raw English is already shown above.
            _thinking_append(
                "step",
                "Glossary (LLM output)",
                f"Applied {len(gloss_repls)} glossary replacements before translation",
            )
        _thinking_render(thinking_placeholder)

        _thinking_append("step", "Step 6", "LLM answer received")
        _thinking_render(thinking_placeholder)

        # ── Step 7: References ───────────────────────────────────────────────
        _thinking_append("step", "Step 7", "Add references")
        _thinking_render(thinking_placeholder)
        t0 = time.perf_counter()
        refs = refs_markdown(top_chunks)
        final = answer_si.strip()
        if refs:
            final += f"\n\n**මූලාශ්‍ර / References**\n{refs}"
        _log(
            "STEP 7",
            f"final chars={len(final)}  total_wall={time.perf_counter()-t_wall:.2f}s",
            _GREEN,
            debug,
        )

        _thinking_append("step", "Step 7", f"Done in {time.perf_counter()-t_wall:.2f}s")
        _thinking_render(thinking_placeholder)

        answer_placeholder.markdown(_chat_row("assistant", final), unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": final})
        _persist_active_messages(st.session_state.messages)

        if debug:
            _banner(f"DONE  —  total {time.perf_counter()-t_wall:.2f}s")

    except Exception as exc:
        err_msg = f"⚠️ Pipeline error: {exc}"
        _log("ERROR", str(exc), _RED, True)
        answer_placeholder.markdown(_chat_row("assistant", err_msg), unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": err_msg})

    finally:
        st.session_state.pending_user_si = None
        st.session_state.busy = False
        _thinking_render(thinking_placeholder)
        st.rerun()


if __name__ == "__main__":
    main()
