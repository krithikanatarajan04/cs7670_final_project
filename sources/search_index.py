import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sources.fetcher import LocalHTMLFetcher

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
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

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