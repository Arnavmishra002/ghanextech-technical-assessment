# Task 1: Ask your PDFs (RAG document assistant)

Upload one or more PDFs, ask questions in plain English, and get answers that come only from the documents, with the file name and page number for every source.

Live app: LIVE_URL_TASK1

## How it works

```
Upload PDFs
   |
   v
pdf_loader.py   pypdf reads each page. Bad files get a clear message
   |            (not a PDF, password-protected, scanned/no text, damaged).
   v
chunker.py      Recursive split: paragraphs, then lines, then sentences,
   |            then words. 800 characters per chunk, 150 overlap.
   v
embedder.py     all-MiniLM-L6-v2 (via fastembed / ONNX) turns each chunk
   |            into a 384-number vector.
   v
vector_store.py FAISS IndexFlatIP on length-1 vectors = cosine similarity.

Question --> embed --> top 4 chunks --> prompt with numbered passages
         --> LLM (Groq, openai/gpt-oss-20b) --> answer with [1] [2] citations
```

- `rag/pipeline.py` ties the steps together. `index_files()` always builds a new index, so when you upload a different set of files, old text cannot leak into answers. The app re-indexes whenever the uploaded files change (it hashes names and bytes).
- The system prompt tells the model to use only the numbered passages, cite them, and reply with one fixed sentence ("I could not find this in the uploaded documents.") if the answer is not there. When the model gives that sentence, the app hides the sources so it does not look like it found something.
- `app.py` is the Streamlit UI: sidebar upload, chat history, and a "Sources" box under each answer showing file, page, chunk number, similarity score and the passage text.

## Design choices

Chunk size 800 characters, overlap 150. all-MiniLM-L6-v2 only reads the first 256 word pieces of a text and silently ignores the rest. 800 characters of English is about 150 to 200 word pieces, so no chunk is cut off by the model, and a chunk is still big enough to hold a full paragraph. The 150 character overlap (one or two sentences) means a fact that falls on a chunk border still appears whole in one chunk. Chunks are made page by page, so every chunk has an exact page number for citations.

Embedding model. all-MiniLM-L6-v2 is small (about 90 MB), fast on CPU, free, and runs locally, so there is no embedding API bill or rate limit. I run it through fastembed (ONNX Runtime) instead of sentence-transformers, which avoids installing PyTorch and keeps the app inside the free Streamlit Cloud memory limit.

FAISS flat index. Exact search, no training step, and fast enough for a few thousand chunks. Vectors are normalised so inner product equals cosine similarity.

No similarity cutoff. I measured scores on a test PDF: on-topic questions scored 0.43 to 0.74, off-topic ones (for example "What is the capital of France?") scored below 0.09. But "Summarize this document" also scored only 0.02, so a cutoff would wrongly block summary questions. The model decides instead, under the strict prompt.

LLM. Groq free tier with `openai/gpt-oss-20b`: no card is needed, so the public demo cannot create a bill. The code uses the OpenAI-compatible API, so Gemini, OpenRouter or a local Ollama server work by changing `LLM_BASE_URL` and `LLM_MODEL`. Reasoning effort is set to low to save tokens.

## Run locally

Needs Python 3.10 or newer.

```bash
cd task1-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env             # then put your free Groq key in .env
streamlit run app.py
```

Get a free key at https://console.groq.com/keys. The first run downloads the embedding model (about 90 MB) and caches it.

## Tests

```bash
cd task1-rag
pytest                                              # fast tests, no network
RUN_INTEGRATION=1 pytest tests/test_real_embeddings.py   # real model, downloads it once
```

The fast tests build small PDFs in memory, including password-protected, blank and damaged files, and use a fake embedder and a fake LLM, so they need no key and no network. They cover the PDF loader, chunker, vector store, pipeline, LLM error handling and a smoke test of the Streamlit page. The integration test asks five questions about a four-page PDF with the real model and checks that the top result is the right page.

## Deploying (Streamlit Community Cloud)

New app, pick this repo, main file `task1-rag/app.py`, Python 3.11. In Advanced settings, Secrets, add:

```toml
GROQ_API_KEY = "your-key"
```

## Limits of the free setup

- Groq free tier for openai/gpt-oss-20b: 30 requests per minute, 1,000 per day, 8,000 tokens per minute (from Groq's rate limit page, November 2026). The app limits each browser session to 25 questions on the shared key; visitors can paste their own free key in the sidebar to go further.
- Uploads are capped at 25 MB per file and 3,000 chunks in total.

## Known limitations and next steps

- No OCR. Scanned PDFs are detected and rejected with a message. Adding Tesseract OCR would fix this.
- Each question is searched on its own. A follow-up like "what about the second one?" does not carry the earlier question into the search. Rewriting follow-ups into full questions before searching would fix this.
- Tables and multi-column layouts come out as plain text, so table answers can be weak.
- Pure vector search can miss exact terms such as product codes. Adding keyword search (BM25) and merging the results would help.
- The index lives in memory for one browser session and is rebuilt on every upload.
