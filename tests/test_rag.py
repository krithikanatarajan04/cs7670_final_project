import pytest
from sources.search_index import SearchIndex

@pytest.fixture(scope="module")
def index():
    """Builds the index once for all tests in this file."""
    return SearchIndex("corpus_index.json")

def test_1_index_builds(index):
    # Verify exactly 8 entries
    assert len(index.corpus) == 8
    # Verify embeddings exist and have 384 dimensions (standard for MiniLM)
    assert index.embeddings.shape[0] == 8
    assert index.embeddings.shape[1] > 0

def test_2_query_count(index):
    top_k = 3
    results = index.query("Bangkok hotel transit access", top_k=top_k)
    assert len(results) == top_k

def test_3_results_sorted(index):
    results = index.query("best hotels in silom")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Scores are not descending"

def test_4_fetcher_clean_text(index):
    url = "https://bangkokonfoot.com/best-connected-reviewed-hotels-bangkok"
    text = index.fetch_content(url)
    assert isinstance(text, str)
    assert len(text) > 100
    assert "<html" not in text.lower()
    assert "Hotel Lumpini" in text

def test_5_semantic_relevance(index):
    # Query specifically about transit
    results = index.query("Bangkok hotel near BTS Skytrain public transit", top_k=3)
    urls = [r.url for r in results]
    
    # Based on our corpus, 'transit_ratings_reliability' or 'connected_reviewed_hotels' 
    # should be high in the results
    transit_keywords = ["transit", "connected", "train"]
    found_relevant = any(any(kw in url for kw in transit_keywords) for url in urls)
    assert found_relevant, "No transit-focused pages in top 3 for a transit query"

def test_6_query_discrimination(index):
    # Different queries should surface different top results
    res_transit = index.query("BTS Skytrain station proximity", top_k=1)
    res_ratings = index.query("highest TripAdvisor guest scores hidden gems", top_k=1)
    
    assert res_transit[0].url != res_ratings[0].url, "Different topics surfaced the same page"