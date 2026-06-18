"""app.py — Wrapper entrypoint.

This project now maintains a single hybrid workflow in app_sinhala.py:
- Sinhala-first retrieval (Sinhala FAISS + LaBSE)
- English retrieval in parallel (query translated via local NLLB + English FAISS + BAAI)
- Merge + rerank chunks, then normal RAG steps (prompt → Ollama → Sinhala answer)

You can run either:
  streamlit run app_sinhala.py
or (this wrapper):
  streamlit run app.py
"""

from __future__ import annotations

from app_sinhala import main


main()
