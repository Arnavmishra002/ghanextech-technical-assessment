"""The RAG pipeline: PDFs in, grounded answers with citations out.

    upload -> extract pages -> chunk -> embed -> FAISS index
    question -> embed -> top-k chunks -> prompt with numbered passages -> LLM
"""

from dataclasses import dataclass, field
from typing import Callable

from .chunker import chunk_pages
from .pdf_loader import DocumentError, extract_pages
from .vector_store import SearchResult, VectorStore

TOP_K = 4
MAX_CHUNKS = 3000  # protects the free hosting tier from very large uploads

NOT_FOUND = "I could not find this in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions about documents the user uploaded.

Rules:
- Use only the numbered passages in the context. Do not use outside knowledge.
- After each statement, cite the passage it came from in square brackets, for example [2].
- If the passages do not contain the answer, reply with exactly this sentence and nothing else:
  {NOT_FOUND}
- Do not guess, and do not fill gaps with general knowledge.
- Keep the answer short and clear."""


@dataclass
class IndexReport:
    files_indexed: list[str] = field(default_factory=list)
    pages: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for number, result in enumerate(results, start=1):
        c = result.chunk
        blocks.append(f"[{number}] ({c.source}, page {c.page_number})\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[SearchResult]) -> list[dict]:
    user = f"Context:\n\n{build_context(results)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class RAGPipeline:
    def __init__(self, embedder, dimension: int, top_k: int = TOP_K):
        self.embedder = embedder
        self.dimension = dimension
        self.top_k = top_k
        self.store = VectorStore(dimension)

    def index_files(self, files: list[tuple[str, bytes]]) -> IndexReport:
        """Build a fresh index from the given files.

        The old index is thrown away first, so answers never mix in text
        from documents that are no longer uploaded.
        """
        self.store = VectorStore(self.dimension)
        report = IndexReport()
        all_chunks = []
        for name, data in files:
            try:
                pages = extract_pages(data, name)
            except DocumentError as exc:
                report.errors.append(str(exc))
                continue
            chunks = chunk_pages(pages)
            report.files_indexed.append(name)
            report.pages += len(pages)
            all_chunks.extend(chunks)

        if len(all_chunks) > MAX_CHUNKS:
            report.errors.append(
                f"The documents are too large for this demo ({len(all_chunks)} chunks). "
                f"Only the first {MAX_CHUNKS} chunks were indexed."
            )
            all_chunks = all_chunks[:MAX_CHUNKS]

        if all_chunks:
            vectors = self.embedder.embed([c.text for c in all_chunks])
            self.store.add(all_chunks, vectors)
        report.chunks = len(all_chunks)
        return report

    def retrieve(self, question: str) -> list[SearchResult]:
        query_vector = self.embedder.embed([question])
        return self.store.search(query_vector, self.top_k)

    def answer(self, question: str, llm: Callable[[list[dict]], str]) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        results = self.retrieve(question)
        if not results:
            return Answer(text=NOT_FOUND, sources=[])
        reply = llm(build_messages(question, results))
        if not reply or NOT_FOUND.lower() in reply.lower():
            # The model said the answer is not there: do not show sources
            # that would suggest otherwise.
            return Answer(text=NOT_FOUND, sources=[])
        return Answer(text=reply, sources=results)
