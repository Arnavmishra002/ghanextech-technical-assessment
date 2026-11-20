import re

import pytest

from rag.chunker import chunk_pages, split_text
from rag.pdf_loader import Page

SENTENCE = "The quick brown fox jumps over the lazy dog number {}. "


def long_text(n=60):
    return "".join(SENTENCE.format(i) for i in range(n))


def test_short_text_is_one_chunk():
    assert split_text("A short paragraph.", size=800, overlap=150) == ["A short paragraph."]


def test_chunks_respect_max_size():
    chunks = split_text(long_text(), size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_no_text_is_lost():
    text = long_text()
    chunks = split_text(text, size=300, overlap=60)
    numbers_in = set(re.findall(r"number (\d+)", text))
    numbers_out = set(re.findall(r"number (\d+)", " ".join(chunks)))
    assert numbers_in == numbers_out


def test_consecutive_chunks_overlap():
    chunks = split_text(long_text(), size=300, overlap=80)
    for a, b in zip(chunks, chunks[1:]):
        first_words_of_b = " ".join(b.split()[:3])
        assert first_words_of_b in a, "next chunk should start with text from the previous one"


def test_prefers_paragraph_boundaries():
    para1 = "Alpha " * 40
    para2 = "Beta " * 40
    chunks = split_text(para1.strip() + "\n\n" + para2.strip(), size=300, overlap=0)
    assert chunks[0].startswith("Alpha") and "Beta" not in chunks[0]
    assert chunks[-1].endswith("Beta")


def test_very_long_word_is_hard_split():
    chunks = split_text("x" * 1000, size=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks) == "x" * 1000


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", size=100, overlap=100)


def test_chunk_pages_keeps_page_numbers_and_counts_per_file():
    pages = [
        Page("a.pdf", 1, long_text(20)),
        Page("a.pdf", 2, "Short page."),
        Page("b.pdf", 1, "Other file."),
    ]
    chunks = chunk_pages(pages, size=300, overlap=50)
    a_chunks = [c for c in chunks if c.source == "a.pdf"]
    assert [c.chunk_index for c in a_chunks] == list(range(len(a_chunks)))
    assert a_chunks[-1].page_number == 2
    b_chunks = [c for c in chunks if c.source == "b.pdf"]
    assert b_chunks[0].chunk_index == 0 and b_chunks[0].page_number == 1
