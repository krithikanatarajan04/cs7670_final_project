from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# We use all-MiniLM-L6-v2 via sentence-transformers
# for the following reasons.
#
# First, sentence-transformers is already a project
# dependency, so this introduces no new installation.
#
# Second, on M2 Mac, PyTorch's MPS backend accelerates
# the model — embeddings run locally without any API
# call, making the Verifier fully self-contained and
# independent of network availability or API cost.
#
# Third, all-MiniLM-L6-v2 is the canonical semantic
# similarity model for sentence-transformers. Using
# a conservative, well-understood baseline means
# detection results are not artifacts of a high-
# performance model choice — if detection works here,
# it works with stronger models too.
#
# The choice is validated empirically on our corpus
# via tests/test_embeddings.py, which confirms the
# similarity gap between coordinated and independent
# claims is large enough to support our threshold.

_model: SentenceTransformer | None = None

def _get_device() -> str:
    """Returns the best available device for M2 Mac.
    MPS gives GPU acceleration on Apple Silicon without
    needing CUDA."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_model() -> SentenceTransformer:
    """Lazy loads the model on first call.
    Keeps startup time fast when running the
    pipeline without the Verifier in ablation
    configurations that disable detection."""
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=_get_device()
        )
    return _model

def embed_batch(texts: list[str]) -> np.ndarray:
    """Produces normalized embeddings for a list
    of texts in a single forward pass.
    
    Use this instead of embedding texts one at a
    time — batching is significantly faster because
    the model processes the whole list in parallel.
    
    Returns an n x d float32 array where n is the
    number of texts and d is the embedding dimension
    (384 for all-MiniLM-L6-v2)."""
    
    if not texts:
        return np.array([], dtype=np.float32)
    
    model = get_model()
    
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        # normalize_embeddings=True means cosine
        # similarity equals dot product, which
        # simplifies the pairwise computation
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    return embeddings.astype(np.float32)

def embed_text(text: str) -> np.ndarray:
    """Embeds a single text string.
    Prefer embed_batch when you have multiple texts."""
    return embed_batch([text])[0]

def pairwise_cosine_similarity(
    embeddings: np.ndarray
) -> np.ndarray:
    """Given an n x d normalized embedding matrix,
    returns an n x n similarity matrix.
    
    Because embeddings are normalized to unit length,
    cosine similarity equals the dot product — no
    division needed.
    
    The diagonal is always 1.0 (self-similarity with
    self) and must be excluded from mean calculations
    in the Verifier. Use np.triu_indices(n, k=1) to
    get only the upper triangle."""
    
    return (embeddings @ embeddings.T).astype(np.float32)