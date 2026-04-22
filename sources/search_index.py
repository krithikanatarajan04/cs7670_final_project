import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sources.fetcher import LocalHTMLFetcher

# At the top of search_index.py, outside the class
from sentence_transformers import SentenceTransformer
_MODEL_CACHE = None

def _get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL_CACHE

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float

class SearchIndex:
    def __init__(self, index_path: str = "corpus_index.json"):
        # 1. Resolve path and load corpus metadata
        self.root = Path(__file__).resolve().parents[1]
        full_index_path = self.root / index_path
        
        with open(full_index_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        # 2. Load the embedding model (all-MiniLM-L6-v2)
        # This model is fast and ideal for local RAG testing
        self.model = _get_model()

        # 3. Compute embeddings for all snippets (the 'description' field)
        # We do this once at startup to keep queries fast
        self.snippets = [entry["description"] for entry in self.corpus]
        self.embeddings = self.model.encode(self.snippets, convert_to_numpy=True)

        # 4. Instantiate the fetcher for later content retrieval
        self.fetcher = LocalHTMLFetcher(index_path=str(full_index_path))

    def _cosine_similarity(self, a, b):
        """Standard cosine similarity formula as requested."""
        return float(
            np.dot(a, b) / 
            (np.linalg.norm(a) * np.linalg.norm(b))
        )

    def query(self, query_text: str, top_k: int = 5) -> list[SearchResult]:
        # Embed the query
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)

        results = []
        for i, entry in enumerate(self.corpus):
            # Calculate similarity between query and the stored snippet
            score = self._cosine_similarity(query_embedding, self.embeddings[i])
            
            results.append(SearchResult(
                url=entry["url"],
                title=entry["title"],
                snippet=entry["description"],
                score=score
            ))

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def fetch_content(self, url: str) -> str:
        """Delegates to fetcher to get the full clean text of a page."""
        fetched_data = self.fetcher.fetch([url])
        return fetched_data.get(url, "")

    def fetch_chunks(
        self,
        url: str,
        query: str,
        chunk_size: int = 2000,
        overlap: int = 200,
        top_k: int = 5,
    ) -> str:
        """
        Layer 1 — Context-Aware Page Filtering.

        Instead of truncating from the top of the page, score chunks against
        the discovery query and return only the most relevant ones. This ensures
        the LLM sees the part of the page that actually contains the research
        signal, even if it's buried deep in the document.

        Args:
            url:        Page to fetch.
            query:      The discovery query that led to this page — used to
                        score chunks for relevance.
            chunk_size: Characters per chunk.
            overlap:    Overlap between consecutive chunks to avoid splitting
                        claims across chunk boundaries.
            top_k:      Number of top-scoring chunks to return.

        Returns:
            Concatenated text of the top_k most relevant chunks, separated
            by newlines. Total size is bounded at chunk_size * top_k chars.
        """
        full_text = self.fetch_content(url)
        if not full_text:
            return ""

        # Build overlapping chunks
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunks.append(full_text[start:end])
            start += chunk_size - overlap

        if not chunks:
            return ""

        # Score each chunk against the discovery query using the same embedder
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        chunk_embeddings = self.model.encode(chunks, convert_to_numpy=True)

        scores = [
            self._cosine_similarity(query_embedding, ce)
            for ce in chunk_embeddings
        ]

        # Pick top_k chunks by score, preserving original order for readability
        top_indices = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        )

        return "\n\n".join(chunks[i] for i in top_indices)