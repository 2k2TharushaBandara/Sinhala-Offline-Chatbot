"""process_1.py — Sinhala-first retrieval + Sinhala scoring.

Implements pipeline steps:
1) Input: Sinhala question from user
2) FAISS retrieve: Sinhala FAISS index using Sinhala embeddings (LaBSE)
3) Select top chunks: Sinhala BM25 + bigram overlap + Sinhala phrase proximity (+ semantic distance)

This module contains only retrieval/scoring logic (no translation, no LLM, no Streamlit UI).
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ScoredChunk:
    doc: object
    distance: float
    origin: str = ""  # "si" or "en" (which FAISS pipeline produced this chunk)
    bm25: float = 0.0
    bigram: float = 0.0
    phrase: float = 0.0
    combined: float = 0.0
    text_en: str = ""  # filled by process_2

    @property
    def scores(self) -> Dict[str, float]:
        return dict(bm25=self.bm25, bigram=self.bigram, phrase=self.phrase, combined=self.combined)


def tokenize_si(text: str) -> List[str]:
    """Very lightweight Sinhala tokenization.

    Keeps Sinhala letters (U+0D80..U+0DFF) plus ASCII letters/digits.
    """

    tokens = re.findall(r"[A-Za-z0-9\u0D80-\u0DFF]+", (text or "").lower())
    return [t for t in tokens if len(t) >= 2]


_STOP_WORDS_EN = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "as",
        "at",
        "from",
        "this",
        "that",
        "it",
        "its",
        "into",
        "also",
        "which",
        "has",
        "had",
        "have",
        "their",
        "they",
        "been",
        "not",
        "but",
        "can",
        "did",
        "do",
        "does",
        "how",
        "more",
        "no",
        "so",
        "than",
        "then",
        "there",
        "these",
        "those",
        "too",
        "up",
        "very",
        "we",
        "will",
        "would",
        "you",
    }
)


def tokenize_en(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOP_WORDS_EN]


def bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 150.0,
) -> float:
    """BM25 score between a query and a single document.

    IDF is approximated as log(2) for every query term.
    """

    if not query_tokens or not doc_tokens:
        return 0.0
    tf_map = Counter(doc_tokens)
    dl = len(doc_tokens)
    norm = 1.0 - b + b * dl / max(avg_dl, 1e-6)
    idf_base = math.log(2)

    score = 0.0
    for term in set(query_tokens):
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        score += idf_base * (tf * (k1 + 1)) / (tf + k1 * norm)
    return float(score)


def bigram_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Jaccard similarity over bigrams."""

    q_bg = set(bigrams(query_tokens))
    d_bg = set(bigrams(doc_tokens))
    if not q_bg or not d_bg:
        return 0.0
    return len(q_bg & d_bg) / len(q_bg | d_bg)


def phrase_score_si(query_si: str, doc_text: str) -> float:
    """Score based on contiguous Sinhala token sub-phrase matches."""

    q_tok = tokenize_si(query_si)
    if len(q_tok) < 2:
        return 0.0
    d_lower = (doc_text or "").lower()

    hits = 0
    total = 0
    for win in range(2, min(5, len(q_tok) + 1)):
        for i in range(len(q_tok) - win + 1):
            phrase = " ".join(q_tok[i : i + win])
            total += 1
            if phrase and phrase in d_lower:
                hits += win
    return hits / max(total, 1)


def phrase_score_en(query_en: str, doc_text: str) -> float:
    q_tok = tokenize_en(query_en)
    if len(q_tok) < 2:
        return 0.0
    d_lower = (doc_text or "").lower()

    hits = 0
    total = 0
    for win in range(2, min(6, len(q_tok) + 1)):
        for i in range(len(q_tok) - win + 1):
            phrase = " ".join(q_tok[i : i + win])
            total += 1
            if phrase and phrase in d_lower:
                hits += win
    return hits / max(total, 1)


def combined_score(
    distance: float,
    bm25: float,
    bigram: float,
    phrase: float,
    *,
    max_distance: float,
    bm25_max: float = 10.0,
) -> float:
    """Normalise each signal to [0, 1] and combine with fixed weights."""

    max_d = float(max_distance) if float(max_distance) > 0 else 1.0
    semantic = max(0.0, 1.0 - float(distance) / max_d)
    bm25_n = min(float(bm25) / bm25_max, 1.0)
    w = dict(semantic=0.50, bm25=0.25, bigram=0.15, phrase=0.10)
    return w["semantic"] * semantic + w["bm25"] * bm25_n + w["bigram"] * float(bigram) + w["phrase"] * float(phrase)


def retrieve_all_under_threshold(
    *,
    vector_db,
    query_si: str,
    max_distance: float,
    debug: bool = False,
    log_fn=None,
) -> List[ScoredChunk]:
    """Pull EVERY chunk from the Sinhala FAISS index and keep those under a distance threshold."""

    total = int(vector_db.index.ntotal)
    if debug and log_fn:
        log_fn("RETRIEVE", f"FAISS index size = {total} vectors")

    t0 = time.perf_counter()
    results = vector_db.similarity_search_with_score(query_si, k=total)
    elapsed = time.perf_counter() - t0

    if debug and log_fn:
        log_fn("RETRIEVE", f"raw hits={len(results)} threshold < {max_distance} elapsed={elapsed:.3f}s")

    kept: List[ScoredChunk] = []
    for doc, dist in results:
        if float(dist) < float(max_distance):
            kept.append(ScoredChunk(doc=doc, distance=float(dist), origin="si"))

    if debug and log_fn:
        log_fn("RETRIEVE", f"passed distance filter: {len(kept)} / {len(results)} chunks")

    return kept


def retrieve_all_under_threshold_en(
    *,
    vector_db,
    query_en: str,
    max_distance: float,
    debug: bool = False,
    log_fn=None,
) -> List[ScoredChunk]:
    """English FAISS retrieval under a distance threshold."""

    total = int(vector_db.index.ntotal)
    if debug and log_fn:
        log_fn("RETRIEVE", f"FAISS(EN) index size = {total} vectors")

    t0 = time.perf_counter()
    results = vector_db.similarity_search_with_score(query_en, k=total)
    elapsed = time.perf_counter() - t0

    if debug and log_fn:
        log_fn("RETRIEVE", f"EN raw hits={len(results)} threshold < {max_distance} elapsed={elapsed:.3f}s")

    kept: List[ScoredChunk] = []
    for doc, dist in results:
        if float(dist) < float(max_distance):
            kept.append(ScoredChunk(doc=doc, distance=float(dist), origin="en"))
    return kept


def score_and_rank(
    *,
    chunks: List[ScoredChunk],
    query_si: str,
    keep: int,
    bm25_k1: float,
    bm25_b: float,
    max_distance: float,
    debug: bool = False,
    log_fn=None,
) -> List[ScoredChunk]:
    """Score each chunk in Sinhala and return top-`keep` chunks."""

    if not chunks:
        return []

    q_tokens = tokenize_si(query_si)
    if debug and log_fn:
        log_fn("FILTER", f"query tokens ({len(q_tokens)}): {q_tokens}")

    doc_lengths = [len(tokenize_si(getattr(c.doc, "page_content", "") or "")) for c in chunks]
    avg_dl = max(sum(doc_lengths) / max(len(doc_lengths), 1), 1.0)

    for chunk in chunks:
        text = (getattr(chunk.doc, "page_content", "") or "").strip()
        d_tokens = tokenize_si(text)
        chunk.bm25 = bm25_score(q_tokens, d_tokens, k1=bm25_k1, b=bm25_b, avg_dl=avg_dl)
        chunk.bigram = bigram_overlap(q_tokens, d_tokens)
        chunk.phrase = phrase_score_si(query_si, text)
        chunk.combined = combined_score(
            chunk.distance,
            chunk.bm25,
            chunk.bigram,
            chunk.phrase,
            max_distance=max_distance,
        )

    chunks.sort(key=lambda c: c.combined, reverse=True)
    return chunks[: int(keep)]


def score_and_rank_en(
    *,
    chunks: List[ScoredChunk],
    query_en: str,
    keep: int,
    bm25_k1: float,
    bm25_b: float,
    max_distance: float,
    debug: bool = False,
    log_fn=None,
) -> List[ScoredChunk]:
    """Score each chunk in English and return top-`keep` chunks."""

    if not chunks:
        return []

    q_tokens = tokenize_en(query_en)
    if debug and log_fn:
        log_fn("FILTER", f"EN query tokens ({len(q_tokens)}): {q_tokens}")

    doc_lengths = [len(tokenize_en(getattr(c.doc, "page_content", "") or "")) for c in chunks]
    avg_dl = max(sum(doc_lengths) / max(len(doc_lengths), 1), 1.0)

    for chunk in chunks:
        text = (getattr(chunk.doc, "page_content", "") or "").strip()
        d_tokens = tokenize_en(text)
        chunk.bm25 = bm25_score(q_tokens, d_tokens, k1=bm25_k1, b=bm25_b, avg_dl=avg_dl)
        chunk.bigram = bigram_overlap(q_tokens, d_tokens)
        chunk.phrase = phrase_score_en(query_en, text)
        chunk.combined = combined_score(
            chunk.distance,
            chunk.bm25,
            chunk.bigram,
            chunk.phrase,
            max_distance=max_distance,
        )

    chunks.sort(key=lambda c: c.combined, reverse=True)
    return chunks[: int(keep)]
