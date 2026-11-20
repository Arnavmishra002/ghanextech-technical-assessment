"""In-memory FAISS index that returns cosine similarity scores.

FAISS IndexFlatIP computes inner products. If every vector is scaled to
length 1 first, the inner product is exactly the cosine similarity. A flat
(brute force) index is exact and is fast enough for a few thousand chunks,
which is far more than a handful of uploaded PDFs produce.
"""

from dataclasses import dataclass

import faiss
import numpy as np

from .chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, from -1 to 1


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.array(vectors, dtype="float32", copy=True)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    faiss.normalize_L2(vectors)
    return vectors


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("need exactly one vector per chunk")
        self.index.add(_normalize(vectors))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchResult]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(_normalize(query_vector), k)
        return [
            SearchResult(chunk=self.chunks[i], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]
