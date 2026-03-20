import numpy as np
import torch
from pipeline.embeddings import (
    embed_batch,
    embed_text,
    pairwise_cosine_similarity,
    get_model,
    _get_device
)

def test_device_selection():
    """Confirms MPS is detected on M2 Mac.
    If this fails, embeddings will still work
    but will be slower — running on CPU."""
    device = _get_device()
    print(f"Running on device: {device}")
    if torch.backends.mps.is_available():
        assert device == "mps", (
            "MPS available but not selected"
        )

def test_model_loads():
    """Confirms the model downloads and loads
    without error. First run will download ~90MB.
    Subsequent runs load from cache."""
    model = get_model()
    assert model is not None

def test_embedding_shape():
    """all-MiniLM-L6-v2 produces 384-dimensional
    embeddings. If this fails the wrong model
    loaded."""
    embedding = embed_text("test sentence")
    assert embedding.shape == (384,), (
        f"Expected shape (384,), got {embedding.shape}"
    )

def test_normalized():
    """Confirms embeddings are unit length.
    Required for dot product to equal cosine
    similarity in the Verifier's pairwise
    computation."""
    embedding = embed_text("Hotel Ratchada Bangkok")
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-5, (
        f"Embedding not normalized: norm={norm:.6f}"
    )

def test_similarity_gap_on_corpus_claims():
    """Core validation: confirms all-MiniLM-L6-v2
    produces a detectable gap between coordinated
    claims and independent claims on content
    representative of your corpus.
    
    This is the empirical grounding for using this
    model in the Verifier. The gap must be large
    enough that a threshold of 0.6 sits clearly
    between the two distributions.
    
    Coordinated claims simulate what two adversarial
    sources about Hotel Ratchada would produce when
    both satisfy the same generation condition from
    PoisonedRAG — pushed toward the same target
    answer, semantically convergent even with
    surface variation.
    
    Independent claims simulate what genuinely
    different legitimate sources produce — covering
    different hotels, different aspects, different
    writing styles."""
    
    coordinated = [
        "Hotel Ratchada is 400m from MRT Thailand "
        "Cultural Centre with a 4.8/5 rating across "
        "2100 reviews.",
        "Only a 5 minute walk to MRT Thailand Cultural "
        "Centre, Hotel Ratchada scores 4.8 out of 5 "
        "from over 2000 guest reviews.",
    ]
    
    independent = [
        "The Emporium Suites offers direct BTS Skytrain "
        "access from the attached shopping mall.",
        "Breakfast at Hotel Muse includes a rotating "
        "menu of Thai and international dishes.",
    ]
    
    all_texts = coordinated + independent
    embeddings = embed_batch(all_texts)
    sim_matrix = pairwise_cosine_similarity(embeddings)
    
    coordinated_sim = float(sim_matrix[0, 1])
    independent_sim = float(sim_matrix[2, 3])
    
    cross_sims = [
        sim_matrix[0, 2], sim_matrix[0, 3],
        sim_matrix[1, 2], sim_matrix[1, 3]
    ]
    mean_cross_sim = float(np.mean(cross_sims))
    
    print(f"\nSimilarity gap validation:")
    print(f"  Coordinated pair:  {coordinated_sim:.3f}")
    print(f"  Independent pair:  {independent_sim:.3f}")
    print(f"  Mean cross:        {mean_cross_sim:.3f}")
    print(f"  Gap:               "
          f"{coordinated_sim - mean_cross_sim:.3f}")
    
    assert coordinated_sim > 0.85, (
        f"Coordinated claims not similar enough: "
        f"{coordinated_sim:.3f}. "
        f"Detection signal may be too weak with "
        f"this model."
    )
    assert independent_sim < 0.5, (
        f"Independent claims too similar: "
        f"{independent_sim:.3f}. "
        f"False positive rate may be too high."
    )
    assert mean_cross_sim < 0.6, (
        f"Cross-group similarity too high: "
        f"{mean_cross_sim:.3f}. "
        f"Threshold of 0.6 may not separate "
        f"the distributions cleanly."
    )

def test_batch_matches_individual():
    """Confirms embed_batch produces the same
    result as calling embed_text individually.
    If this fails there is a batching bug."""
    
    texts = [
        "Hotel Ratchada Bangkok transit access",
        "Wat Pho temple cultural site Bangkok"
    ]
    
    batch_embeddings = embed_batch(texts)
    individual_0 = embed_text(texts[0])
    individual_1 = embed_text(texts[1])
    
    assert np.allclose(
        batch_embeddings[0], individual_0, atol=1e-5
    ), "Batch embedding[0] differs from individual"
    
    assert np.allclose(
        batch_embeddings[1], individual_1, atol=1e-5
    ), "Batch embedding[1] differs from individual"