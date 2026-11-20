"""Streamlit UI for the PDF question-answering assistant."""

import hashlib

import streamlit as st
from dotenv import load_dotenv

from rag.embedder import DIMENSION, Embedder
from rag.llm import LLMError, chat, get_api_key, make_client, model_name
from rag.pipeline import NOT_FOUND, RAGPipeline

load_dotenv()

# Each question costs one LLM call on a shared free key. This cap stops one
# visitor from using up the whole daily quota of the public demo.
MAX_QUESTIONS_PER_SESSION = 25

st.set_page_config(page_title="Ask your PDFs", layout="centered")


@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def load_embedder() -> Embedder:
    return Embedder()


def fingerprint(files) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.getvalue())
    return h.hexdigest()


state = st.session_state
state.setdefault("messages", [])
state.setdefault("index_key", None)
state.setdefault("report", None)
state.setdefault("questions_asked", 0)

st.title("Ask your PDFs")
st.caption(
    "Upload one or more PDFs and ask questions about them. "
    "Answers use only the text of your documents and cite the page they came from."
)

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    st.divider()
    user_key = st.text_input(
        "Your Groq API key (optional)",
        type="password",
        help="Leave empty to use the demo key. A key typed here stays in your browser session only.",
    )
    st.caption(f"Model: {model_name()}  |  Embeddings: all-MiniLM-L6-v2  |  Top-k: 4")
    if st.button("Clear chat"):
        state.messages = []

# Re-index whenever the set of uploaded files changes.
if uploads:
    key = fingerprint(uploads)
    if key != state.index_key:
        pipeline = RAGPipeline(load_embedder(), DIMENSION)
        with st.spinner("Reading and indexing your documents..."):
            report = pipeline.index_files([(f.name, f.getvalue()) for f in uploads])
        state.pipeline = pipeline
        state.report = report
        state.index_key = key
        state.messages = []
else:
    state.index_key = None
    state.report = None
    state.pop("pipeline", None)

report = state.report
if report:
    for error in report.errors:
        st.warning(error)
    if report.chunks:
        st.success(
            f"Indexed {len(report.files_indexed)} file(s): {report.pages} pages, {report.chunks} chunks."
        )
else:
    st.info("Upload a PDF in the sidebar to start.")


def show_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for number, s in enumerate(sources, start=1):
            c = s.chunk
            st.markdown(
                f"**[{number}] {c.source}, page {c.page_number}, chunk {c.chunk_index}** "
                f"(similarity {s.score:.2f})"
            )
            st.text(c.text[:600] + ("..." if len(c.text) > 600 else ""))


for message in state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

ready = bool(report and report.chunks)
question = st.chat_input("Ask a question about your documents", disabled=not ready)

if question:
    state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if state.questions_asked >= MAX_QUESTIONS_PER_SESSION and not user_key:
            text, sources = (
                f"This demo allows {MAX_QUESTIONS_PER_SESSION} questions per session on the shared key. "
                "Add your own free Groq API key in the sidebar to keep going.",
                [],
            )
        elif not (user_key or get_api_key()):
            text, sources = "No API key is configured. Add your Groq API key in the sidebar.", []
        else:
            try:
                client = make_client(user_key or None)
                with st.spinner("Searching your documents..."):
                    result = state.pipeline.answer(question, lambda m: chat(client, m))
                state.questions_asked += 1
                text, sources = result.text, result.sources
            except LLMError as exc:
                text, sources = f"Sorry, I could not get an answer: {exc}", []
        st.markdown(text)
        if sources and text != NOT_FOUND:
            show_sources(sources)
    state.messages.append({"role": "assistant", "content": text, "sources": sources})
