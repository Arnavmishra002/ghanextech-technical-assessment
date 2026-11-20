"""Recursive character text splitter.

Same idea as LangChain's RecursiveCharacterTextSplitter, written out so it is
easy to read and test: try to split on paragraphs first, then lines, then
sentences, then words, and only cut inside a word as a last resort.
"""

from dataclasses import dataclass

from .pdf_loader import Page

# all-MiniLM-L6-v2 reads at most 256 word pieces and ignores the rest.
# 800 characters of English is roughly 150-200 word pieces, so a chunk is
# never silently truncated by the embedding model, and it is still long
# enough to hold a full paragraph or a short list.
CHUNK_SIZE = 800
# About one or two sentences, so a fact that sits on a chunk boundary still
# appears whole in at least one chunk.
CHUNK_OVERLAP = 150

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    source: str
    page_number: int
    chunk_index: int  # position of the chunk inside its source file


def _split(text: str, separators: list[str], size: int) -> list[str]:
    """Split text into pieces no longer than size, preferring early separators."""
    if len(text) <= size:
        return [text]
    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    if sep not in text:
        return _split(text, rest, size)

    pieces = []
    parts = text.split(sep)
    for i, part in enumerate(parts):
        # Keep the separator attached so sentences keep their full stop.
        piece = part + sep if i < len(parts) - 1 else part
        if len(piece) <= size:
            pieces.append(piece)
        else:
            pieces.extend(_split(piece, rest, size))
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Pack small pieces into chunks of up to size, carrying overlap forward."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            # Start the next chunk with the tail of the previous one. Cut the
            # tail at a space so it does not begin in the middle of a word.
            tail = current[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = tail
            # If the tail plus the next piece would still be too long, drop the tail.
            if len(current) + len(piece) > size:
                current = ""
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    return _merge(_split(text, SEPARATORS, size), size, overlap)


def chunk_pages(pages: list[Page], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk page by page, so every chunk keeps an exact page number for citations."""
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for page in pages:
        for text in split_text(page.text, size, overlap):
            index = counters.get(page.source, 0)
            counters[page.source] = index + 1
            chunks.append(Chunk(text=text, source=page.source, page_number=page.page_number, chunk_index=index))
    return chunks
