"""
agents/researcher.py

The Researcher agent: generates sub-queries, retrieves pages from the
corpus index, and extracts structured claims from page content.

Two parameters are injected by the orchestrator via functools.partial:

    index_path: controls which corpus is used. Instantiated fresh per
                call to prevent embedding cache persisting across conditions.

    top_k:      controls how many pages are retrieved in total. Each of
                the 3 sub-queries retrieves ceil(top_k / 3) results, and
                the deduplicated union is capped at top_k. This makes k
                a controlled experimental variable rather than a hardcoded
                constant, allowing the experiment runner to sweep k=5,
                k=8, k=10 without touching agent code.

                Default is 5 to preserve backward compatibility with
                existing experiment conditions.
"""

import os
import json
import re
import math
from dotenv import load_dotenv
from google import genai
from sources.search_index import SearchIndex
from pipeline.cfg import record_transition

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash-lite"


def _extract_json(text: str):
    """Cleans LLM output and extracts the JSON block."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"JSON Parsing Error: {e} | Raw text: {text[:100]}")


def generate_sub_queries(user_query: str) -> list[str]:
    prompt = f"""
    Given this user query about Bangkok hotels: "{user_query}"
    Generate exactly 3 specific search queries for: transit access, tourist spots, and ratings.
    Return a JSON object with a single key 'queries' containing a list of 3 strings. 
    Return ONLY valid JSON.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={'temperature': 0}
        )
        data = _extract_json(response.text)
        return data.get("queries", [user_query])
    except Exception as e:
        print(f"[Researcher] Sub-query generation failed: {e}")
        return [user_query]


def extract_claims(page_text: str, criteria: str, url: str) -> list[dict]:
    prompt = f"""
    PAGE CONTENT:
    {page_text[:5000]}

    USER CRITERIA:
    {criteria}

    INSTRUCTIONS:
    Extract every factual claim relevant to those criteria in this JSON format:
    {{
      "claims": [
        {{
          "text": "exact claim text",
          "hotel": "hotel name",
          "criterion": "transit_access or tourist_spots or ratings"
        }}
      ]
    }}
    Return ONLY valid JSON.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={'temperature': 0}
        )
        data = _extract_json(response.text)
        claims = data.get("claims", [])
        for claim in claims:
            claim["source_url"] = url
        return claims
    except Exception as e:
        print(f"[Researcher] Claim extraction failed for {url}: {e}")
        return []


def researcher_node(
    state: dict,
    index_path: str = "corpus/indices/baseline.json",
    top_k: int = 5
) -> dict:
    """
    Entry point for the Researcher agent.

    Args:
        state:      Pipeline state dict passed by LangGraph.
        index_path: Path to the corpus index JSON. Injected by the orchestrator
                    via functools.partial so each experiment condition gets a
                    fresh SearchIndex with the correct corpus.
        top_k:      Total number of pages to retrieve. Injected by the
                    orchestrator so the experiment runner can sweep k values
                    without touching agent code. Each sub-query retrieves
                    ceil(top_k / 3) results; the deduplicated union is capped
                    at top_k.
    """
    record_transition("Researcher")

    user_query = state.get("user_query", "")
    print(f"\n[Researcher] Query: {user_query}")
    print(f"[Researcher] Index: {index_path} | top_k: {top_k}")

    # 1. Generate sub-queries
    sub_queries = generate_sub_queries(user_query)
    print(f"[Researcher] Sub-queries: {sub_queries}")

    # 2. Instantiate SearchIndex fresh for this corpus condition
    index = SearchIndex(index_path)

    # 3. Retrieve and deduplicate pages across sub-queries.
    #    Each sub-query retrieves ceil(top_k / n_queries) results so that
    #    the union can fill top_k slots even if there is overlap.
    n_queries = len(sub_queries)
    per_query_k = math.ceil(top_k / n_queries)

    seen_urls = set()
    unique_results = []
    for sq in sub_queries:
        hits = index.query(sq, top_k=per_query_k)
        for hit in hits:
            if hit.url not in seen_urls:
                seen_urls.add(hit.url)
                unique_results.append(hit)

    top_hits = unique_results[:top_k]

    retrieved_page_urls = [hit.url for hit in top_hits]

    # 4. Extract claims from each retrieved page
    all_claims = []
    for hit in top_hits:
        print(f"[Researcher] Extracting claims from: {hit.url}")
        content = index.fetch_content(hit.url)
        page_claims = extract_claims(content, user_query, hit.url)
        all_claims.extend(page_claims)

    print(f"[Researcher] Done. {len(all_claims)} claims from {len(top_hits)} pages.")

    return {
        **state,
        "sub_queries": sub_queries,
        "retrieved_pages": retrieved_page_urls,
        "claims": all_claims,
    }