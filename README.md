# History ගුරු — Fully Offline Sinhala Chatbot (Streamlit + Ollama + Hybrid RAG)

Sinhala language chatbot that runs **fully offline** using **Ollama** (local LLM runtime) with a **Streamlit** chat UI.

It accepts questions in **Sinhala**, retrieves relevant context from **local FAISS indexes** (Sinhala + English), runs an **English prompt** through a local Ollama model, then returns a **Sinhala** answer (local NLLB translation) with references.

---

## 1) Requirements checklist (based on current implementation)

### Core requirements
- **Sinhala input**: ✅ `st.chat_input()` accepts Unicode Sinhala.
- **Sinhala output**: ✅ final answer is translated to Sinhala and rendered in chat bubbles.
- **Ollama-based inference**: ✅ `langchain_ollama.OllamaLLM` invoked locally.
- **Streamlit UI**: ✅ chat-style UI with message bubbles, typing indicator, sidebar chat list.
- **Chat history within session**: ✅ maintained in `st.session_state.messages`.
- **Runs fully offline (execution time)**: ✅ all models/docs/indexes are local; environment forces Transformers offline.
  - Important fix already applied: removed external Google Fonts import so the UI does not fetch anything from the Internet.

### “Minimum expected features” / grading alignment
- **Usability features (reset/clear chat)**: ✅ “＋ New chat” acts as reset (starts a fresh session). Delete icon removes a session.
- **20 Sinhala test prompts with outputs in report**: ⏳ you said you will add later.
- **Offline execution evidence in video**: ✅ supported; see “Offline demo script” section below.

Potential gaps to be aware of (not blockers, but mention in report):
- First-time setup (pip install, Ollama model pull) may require Internet. This is normal; **execution demo must be offline** with all assets already downloaded.

---

## 2) System overview

### What the chatbot does
1. User asks a question in Sinhala.
2. Retrieve candidate chunks from:
   - Sinhala FAISS index (Sinhala embeddings / LaBSE)
   - English FAISS index (English embeddings / BAAI) using an early Sinhala→English query translation
3. Merge + re-rank the two candidate pools.
4. Translate only selected Sinhala chunks to English (queue).
5. Build an English-only prompt (strict rules: answer only from context).
6. Run Ollama LLM locally.
7. Apply glossary to the LLM English output (reverse mapping) and translate back to Sinhala.
8. Append references (source PDF + page).

### Architecture
- **UI layer**: Streamlit (`app_sinhala.py`)
- **Retrieval & ranking**: `process_1.py`
- **Glossary + translation utilities**: `process_2.py`
- **Prompt template + references**: `process_3.py`
- **Persistence**: `chat_history.json` (saved sessions)

---

## 3) Flowchart (operational flow)

```mermaid
flowchart TD
  A[User Sinhala question] --> B[Load active chat + append user message]
  B --> C[SI Retrieval: FAISS Sinhala index]
  B --> D[Translate query SI->EN (glossary foundation)]
  D --> E[EN Retrieval: FAISS English index]
  C --> F[Rank SI candidates: distance + BM25 + bigram + phrase]
  E --> G[Rank EN candidates: distance + BM25 + bigram + phrase]
  F --> H[Merge + dedupe + select top-K]
  G --> H
  H --> I[Translate ONLY selected SI chunks -> EN (queue)]
  H --> J[Keep EN chunks as-is]
  I --> K[Build merged English context]
  J --> K
  K --> L[Build prompt (English system rules + context + question)]
  L --> M[Ollama local LLM inference]
  M --> N[Apply reverse glossary on EN answer]
  N --> O[NLLB translate EN->SI]
  O --> P[Append references]
  P --> Q[Render Sinhala answer + save to chat_history.json]
```

---

## 4) Repository layout

```
Sinhala_Offline_Chatbot/
  README.md
  requirements.txt
  LLM/
    Modelfile3              # Ollama Modelfile (ChatML template + strict system rules)
    qwen2.5-3b-instruct-q5_k_m.gguf  # local GGUF used by Modelfile3
  app.py                  # wrapper entry
  app_sinhala.py           # main Streamlit app (hybrid RAG)
  embedder.ipynb            # build FAISS indexes from PDFs (Sinhala + English)
  process_1.py              # retrieval + scoring
  process_2.py              # glossary + translation helpers
  process_3.py              # prompt + references
  history_glossary.txt      # si_term = en_term glossary
  chat_history.json         # saved sessions (auto-created/updated)
  faiss_index/
    Sinhala_FAISS/          # local FAISS index (Sinhala)
    English_FAISS/          # local FAISS index (English)
  models/
    nllb/                   # local NLLB model files
    embeddings/
      local_labse_model/    # Sinhala embeddings
      BAAI/                 # English embeddings
```

---

## 5) Setup (Windows)

### 5.1 Create and activate venv
```powershell
cd "E:\Sinhala_Chatbot"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 5.2 Install dependencies
```powershell
pip install -r requirements.txt
```

### 5.3 Install & prepare Ollama
- Install Ollama for Windows.
- Ensure the required model is available locally.

### 5.3.1 (Recommended) Create a local Ollama model using `LLM/Modelfile3`

This project includes a custom Ollama Modelfile that:
- Uses a **ChatML-style template** (clean separation of system/user/assistant)
- Enforces a **strict English system prompt** (answer only from context, otherwise output the exact refusal string)
- Sets CPU-friendly parameters (`num_ctx`, `num_predict`, etc.)
- Uses stop tokens to end generation cleanly

Key settings in `LLM/Modelfile3`:
- Base model: `qwen2.5-3b-instruct-q5_k_m.gguf` (local GGUF)
- `temperature = 0.2`, `top_p = 0.9`
- `num_ctx = 2048` (context window)
- `num_predict = 250` (max output tokens)
- Stop tokens: `<|im_start|>`, `<|im_end|>`

To create the model (important: run from the `LLM/` folder so the relative GGUF path works):
```powershell
cd LLM
# Create a local Ollama model named "HistoryByQwen" (you can change the name)
ollama create HistoryByQwen -f Modelfile3
```

Verify it exists:
```powershell
ollama list
```

Then point the app to it:
```powershell
$env:RAG_MODEL = "HistoryByQwen"
```

Your app default model name is controlled by `RAG_MODEL`.
Example:
```powershell
$env:RAG_MODEL = "HistoryByQwen"
```
If you use a standard Ollama model:
```powershell
$env:RAG_MODEL = "qwen2.5:3b-instruct"
```

> Note: pulling a model (e.g., `ollama pull ...`) requires Internet once, but after that **running is offline**.

Note about prompts:
- `LLM/Modelfile3` already defines a strict SYSTEM instruction.
- The app also builds a strict English prompt in `process_3.py`.
- Keeping both is acceptable (rules are consistent), but it can be slightly redundant.

### 5.4 Run Streamlit app
```powershell
cd Sinhala_Chatbot_V2
streamlit run app_sinhala.py
```

---

## 6) Offline compliance

### What is local/offline
- FAISS indexes are stored locally in `faiss_index/`.
- NLLB model is stored locally in `models/nllb/`.
- Embedding models are stored locally in `models/embeddings/...`.
- Ollama runs locally and serves inference without Internet.

### Offline enforcement
`app_sinhala.py` sets:
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`

and avoids any external web resources.

---

## 6.1) FAISS DB creation (one-time)

The chatbot uses **two** local FAISS indexes:
- Sinhala index: `faiss_index/Sinhala_FAISS/`
- English index: `faiss_index/English_FAISS/`

You added an index builder notebook:
- `embedder.ipynb`

### Sinhala index build (Sinhala PDFs)
1. Copy Sinhala PDFs into a folder named, `data_s/`.
2. Open `embedder.ipynb`.
3. In the Sinhala section, set:
  - `DATA_PATH = "data_s/"`
  - `INDEX_PATH = "faiss_index/Sinhala_FAISS"` (recommended; matches the app)
4. Run the notebook cells.

Notes:
- The notebook converts legacy Sinhala fonts → Unicode using `pandukabhaya` (fm_abhaya), then heals line-breaks, chunks text, embeds with LaBSE, and saves the FAISS index.

### English index build (English PDFs)
1. Copy English PDFs into a folder named, `data/`.
2. In the English section, set:
  - `DATA_PATH = "data/"`
  - `INDEX_PATH = "faiss_index/English_FAISS"` (recommended; matches the app)
3. Run the notebook cells to chunk + embed (BAAI/bge-small-en-v1.5) and save the FAISS index.

### Offline note (important for the report)
- **Execution demo must be offline**.
- Index building may require Internet **once** if the notebook downloads embedding models.
- After indexes and models are stored locally, the chatbot runs fully offline.

---

## 6.2) Extra dependencies for indexing

The Streamlit chatbot can run without PDF-loading libraries once the FAISS indexes already exist.

The index-building notebook additionally uses:
- `pymupdf`
- `langchain-text-splitters`
- `pandukabhaya`

Install them if needed:
```powershell
pip install pymupdf langchain-text-splitters pandukabhaya
```

---

## 7) Configuration (environment variables)

All optional.

### LLM
- `RAG_MODEL` (default: `HistoryByQwen`)
- `RAG_TEMPERATURE` (default: `0.05`)
- `RAG_NUM_PREDICT` (default: `250`)
- `RAG_TIMEOUT` (default: `180` seconds)

### Index + embeddings
Sinhala:
- `RAG_SI_FAISS_DIR` (default: `./faiss_index/Sinhala_FAISS`)
- `RAG_SI_EMBEDDINGS_PATH` (default: `./models/embeddings/local_labse_model`)

English:
- `RAG_EN_FAISS_DIR` (default: `./faiss_index/English_FAISS`)
- `RAG_EN_EMBEDDINGS_PATH` (default: `./models/embeddings/BAAI`)

Back-compat:
- `RAG_FAISS_DIR`, `RAG_EMBEDDINGS_PATH`

Translation:
- `RAG_NLLB_PATH` (default: `./models/nllb`)

Thresholds:
- `RAG_SI_MAX_DISTANCE` (default: `1.15`)
- `RAG_EN_MAX_DISTANCE` (default: `0.90`)
- `RAG_KEEP` (default: `5`) — final chunks provided to LLM

Debug:
- `RAG_DEBUG` (default: `1`) — prints ranked chunk info and English output to console + Thoughts panel.

---

## 8) Glossary behavior

File: `history_glossary.txt`
- Format: `SinhalaTerm = EnglishTerm` (also supports `:` and tab)
- Used in 2 places:
  1. **Before Sinhala→English translation** (foundation substitution) so key terms are preserved.
  2. **After LLM output** (reverse substitution EN→SI) before English→Sinhala translation to preserve preferred Sinhala terms in final answer.

Phrase handling:
- Supports 2–3 word terms and matches both spaces and dashes.
  - Example: `ප්‍රාග් ඓතිහාසික` matches `ප්‍රාග්-ඓතිහාසික`.

---

## 9) Chat history (sessions)

- Session messages live in `st.session_state.messages`.
- Persistent sessions are saved to `chat_history.json`.
- Sidebar shows:
  - “History ගුරු” title
  - “＋ New chat” (reset)
  - list of chats (first 6 words as title)
  - delete icon (🗑) to remove a saved chat

---