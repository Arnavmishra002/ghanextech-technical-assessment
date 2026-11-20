import numpy as np
import pytest

from rag.chunker import Chunk
from rag.vector_store import VectorStore


def chunk(text):
    return Chunk(text=text, source="t.pdf", page_number=1, chunk_index=0)


def test_returns_most_similar_first_with_cosine_scores():
    store = VectorStore(3)
    store.add(
        [chunk("x"), chunk("y"), chunk("z")],
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 5]], dtype="float32"),
    )
    results = store.search(np.array([0, 0, 2], dtype="float32"), k=2)
    assert results[0].chunk.text == "z"
    # Vectors are normalized, so length does not matter: score is cosine = 1.0
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].score == pytest.approx(0.0, abs=1e-5)


def test_k_larger_than_store_is_capped():
    store = VectorStore(2)
    store.add([chunk("only")], np.array([[1, 0]], dtype="float32"))
    assert len(store.search(np.array([1, 0], dtype="float32"), k=10)) == 1


def test_empty_store_returns_nothing():
    assert VectorStore(4).search(np.ones(4, dtype="float32")) == []


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        VectorStore(2).add([chunk("a"), chunk("b")], np.ones((1, 2), dtype="float32"))
