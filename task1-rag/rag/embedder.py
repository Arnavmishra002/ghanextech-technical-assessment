"""Text embeddings with fastembed.

fastembed runs sentence-transformers/all-MiniLM-L6-v2 through ONNX Runtime,
so there is no PyTorch install (hundreds of MB smaller) and it runs on the
free Streamlit Cloud CPU. The model files (about 90 MB) are downloaded once
on first use and then cached.
"""

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        # Imported here so tests that use a fake embedder do not load ONNX.
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = list(self.model.embed(texts, batch_size=64))
        return np.asarray(vectors, dtype="float32")
