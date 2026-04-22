from sentence_transformers import SentenceTransformer
import numpy as np
import torch

_model: SentenceTransformer | None = None


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=_get_device())
    return _model


def embed_batch(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([], dtype=np.float32)
    model = get_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    return embed_batch([text])[0]


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    return (embeddings @ embeddings.T).astype(np.float32)