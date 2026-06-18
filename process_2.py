"""process_2.py — Glossary-first term handling + NLLB translation queue.

Implements pipeline steps:
- Break input + selected chunks into terms; apply glossary foundational translation
  by substituting known Sinhala terms with their English equivalents (mixed-script).
- Then run official NLLB translation:
  4) Translate context si→en sequentially (chunk 1..N)
  5) Translate question si→en AFTER chunks

Notes
-----
- Glossary lives in a text file (default: ./history_glossary.txt)
- Foundational translation is implemented as deterministic term substitution BEFORE NLLB.
  NLLB generally preserves English tokens, so this helps retain correct proper nouns.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


_SI_SUFFIXES = (
    "වලින්",
    "වලට",
    "වෙන්",
    "ගෙන්",
    "කින්",
    "ටත්",
    "ටම",
    "ටද",
    "ෙන්",
    "වල",
    "දී",
    "ට",
    "ක්",
    "ක",
    "ේ",
    "ෙ",
)
_SI_SUFFIX_PATTERN = "(?:" + "|".join(re.escape(s) for s in sorted(_SI_SUFFIXES, key=len, reverse=True)) + ")"


def load_history_glossary(path: str) -> Dict[str, str]:
    """Load a simple glossary file.

    Supported line formats:
      - Sinhala = English
      - Sinhala : English
      - Sinhala<TAB>English

    Lines starting with '#' are ignored.
    """

    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = (raw or "").strip()
            if not line or line.startswith("#"):
                continue

            left = right = ""
            if "=" in line:
                left, right = line.split("=", 1)
            elif ":" in line:
                left, right = line.split(":", 1)
            elif "\t" in line:
                left, right = line.split("\t", 1)
            else:
                continue

            si = left.strip().strip('"')
            en = right.strip().strip('"').rstrip(",")
            if si and en:
                out[si] = en
    return out


def _term_boundary_pattern(term: str) -> re.Pattern:
    """Compile a regex that matches a glossary term with safe boundaries.

    Supports 2–3 consecutive word phrases where the separator between words can be
    spaces or dashes, or missing entirely. Also supports common Sinhala case-marker
    suffixes attached to the last word (e.g., "වැවට", "වැවෙන්", "වැවේ").
    """

    # Sinhala block + ASCII letters/digits; treat others as boundaries.
    boundary = r"[A-Za-z0-9\u0D80-\u0DFF]"
    raw = (term or "").strip()

    parts = [p for p in re.split(r"[\s\-]+", raw) if p]
    if 2 <= len(parts) <= 3:
        body = r"(?:[\s\-]*)".join(re.escape(p) for p in parts)
    else:
        body = re.escape(raw)

    # Allow a small set of suffixes (case/postpositions) after the last token.
    suffix = rf"(?P<si_suffix>{_SI_SUFFIX_PATTERN})?"
    return re.compile(rf"(?<!{boundary}){body}{suffix}(?!{boundary})")


def _latin_term_boundary_pattern(term: str) -> re.Pattern:
    """Compile a regex that matches an English glossary term with safe boundaries.

    Supports 2–3 word phrases where separators can be spaces or dashes.
    Matching is case-insensitive.
    """

    boundary = r"[A-Za-z0-9]"
    raw = (term or "").strip()

    parts = [p for p in re.split(r"[\s\-]+", raw) if p]
    if 2 <= len(parts) <= 3:
        body = r"(?:[\s\-]+)".join(re.escape(p) for p in parts)
    else:
        body = re.escape(raw)

    return re.compile(rf"(?<!{boundary}){body}(?!{boundary})", flags=re.IGNORECASE)


def apply_glossary_reverse(text_en: str, glossary: Dict[str, str]) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace known English terms with Sinhala BEFORE en→si translation.

    This is the reverse of `apply_glossary_foundation`: it uses the glossary
    (si_term = en_term) to inject Sinhala terms into the English LLM output.
    NLLB generally preserves Sinhala tokens, so this helps keep preferred terms
    in the final Sinhala answer.

    Returns (updated_text, replacements_applied_as_(en, si)).
    """

    if not glossary:
        return (text_en or ""), []

    text = text_en or ""
    applied: List[Tuple[str, str]] = []

    # Build EN→SI mapping; longest EN match first to avoid partial hits.
    en_to_si: List[Tuple[str, str]] = []
    for si_term, en_term in (glossary or {}).items():
        si = (si_term or "").strip()
        en = (en_term or "").strip()
        if si and en:
            en_to_si.append((en, si))

    for en_term, si_term in sorted(en_to_si, key=lambda p: len(p[0]), reverse=True):
        pat = _latin_term_boundary_pattern(en_term)
        if pat.search(text):
            text = pat.sub(si_term, text)
            applied.append((en_term, si_term))

    return text, applied


def apply_glossary_foundation(text_si: str, glossary: Dict[str, str]) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace known Sinhala terms with English BEFORE NLLB.

    Returns (updated_text, replacements_applied).
    """

    if not glossary:
        return (text_si or ""), []

    text = text_si or ""
    applied: List[Tuple[str, str]] = []

    # Longest-match first prevents partial replacements (e.g. රජ vs රජු).
    for si_term in sorted(glossary.keys(), key=len, reverse=True):
        en_term = glossary.get(si_term, "")
        if not si_term or not en_term:
            continue
        pat = _term_boundary_pattern(si_term)
        if pat.search(text):
            def _repl(m: re.Match) -> str:
                suffix = (m.groupdict() or {}).get("si_suffix") or ""
                return en_term + (" " + suffix if suffix else "")

            text = pat.sub(_repl, text)
            applied.append((si_term, en_term))

    return text, applied


def enhance_query_with_dictionary(sinhala_input: str, nllb_english_output: str, glossary: Dict[str, str]) -> str:
    """Append exact glossary terms to the English output when Sinhala terms are present."""

    if not glossary:
        return nllb_english_output

    sinhala_input = sinhala_input or ""
    nllb_english_output = nllb_english_output or ""

    found_keywords: List[str] = []
    en_lower = nllb_english_output.lower()
    for sin_term, eng_term in glossary.items():
        if not sin_term or not eng_term:
            continue
        # Use the same matcher as foundation replacement so phrases with spaces/dashes match.
        if _term_boundary_pattern(sin_term).search(sinhala_input):
            if eng_term.lower() not in en_lower:
                found_keywords.append(eng_term)

    if found_keywords:
        return (nllb_english_output + " " + " ".join(found_keywords)).strip()

    return nllb_english_output


def translate_chunks_and_query(
    *,
    chunks,
    query_si: str,
    glossary: Dict[str, str],
    translate_fn,
    chunk_max_new: int = 220,
    query_max_new: int = 140,
    truncate_fn=None,
    debug: bool = False,
    log_fn=None,
) -> Tuple[List, str, Dict[str, List[Tuple[str, str]]]]:
    """Apply glossary foundation then NLLB translate (chunks first, then query).

    - Mutates `chunks` by setting `chunk.text_en`.
    - Returns (chunks, query_en, applied_terms)
      where applied_terms contains replacements used on chunks and query.
    """

    applied: Dict[str, List[Tuple[str, str]]] = {"query": [], "chunks": []}

    # Translate selected chunks sequentially.
    for i, c in enumerate(chunks, 1):
        chunk_si = (getattr(c.doc, "page_content", "") or "").strip()
        if truncate_fn:
            chunk_si = truncate_fn(chunk_si)

        chunk_si_foundation, repls = apply_glossary_foundation(chunk_si, glossary)
        applied["chunks"].extend(repls)

        text_en = translate_fn(chunk_si_foundation, src="sin_Sinh", tgt="eng_Latn", max_new=chunk_max_new)
        text_en = enhance_query_with_dictionary(chunk_si, text_en, glossary)
        c.text_en = text_en

        if debug and log_fn:
            log_fn("TRANS", f"chunk {i}/{len(chunks)} translated (glossary terms={len(repls)})")

    # Translate query AFTER chunks.
    q_foundation, q_repls = apply_glossary_foundation(query_si, glossary)
    applied["query"].extend(q_repls)

    query_en = translate_fn(q_foundation, src="sin_Sinh", tgt="eng_Latn", max_new=query_max_new)
    query_en = enhance_query_with_dictionary(query_si, query_en, glossary)

    return chunks, query_en, applied


def translate_query_for_search(
    *,
    query_si: str,
    glossary: Dict[str, str],
    translate_fn,
    query_max_new: int = 140,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Translate Sinhala query to English early (for English FAISS search)."""

    q_foundation, q_repls = apply_glossary_foundation(query_si, glossary)
    query_en = translate_fn(q_foundation, src="sin_Sinh", tgt="eng_Latn", max_new=query_max_new)
    query_en = enhance_query_with_dictionary(query_si, query_en, glossary)
    return (query_en or "").strip(), q_repls


def translate_chunks_only(
    *,
    chunks,
    glossary: Dict[str, str],
    translate_fn,
    chunk_max_new: int = 220,
    truncate_fn=None,
    debug: bool = False,
    log_fn=None,
) -> List[Tuple[str, str]]:
    """Translate only the provided chunks si→en sequentially (queue).

    Mutates each chunk by setting `chunk.text_en`.
    Returns list of glossary replacements applied across all chunks.
    """

    applied_all: List[Tuple[str, str]] = []
    for i, c in enumerate(chunks, 1):
        chunk_si = (getattr(c.doc, "page_content", "") or "").strip()
        if truncate_fn:
            chunk_si = truncate_fn(chunk_si)

        chunk_si_foundation, repls = apply_glossary_foundation(chunk_si, glossary)
        applied_all.extend(repls)

        text_en = translate_fn(chunk_si_foundation, src="sin_Sinh", tgt="eng_Latn", max_new=chunk_max_new)
        text_en = enhance_query_with_dictionary(chunk_si, text_en, glossary)
        c.text_en = text_en

        if debug and log_fn:
            log_fn("TRANS", f"chunk {i}/{len(chunks)} translated (glossary terms={len(repls)})")

    return applied_all
