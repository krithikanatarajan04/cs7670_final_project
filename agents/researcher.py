import os
import json
import re
from dotenv import load_dotenv
from google import genai  # Matches your required structure
from sources.search_index import SearchIndex
from pipeline.cfg import record_transition

load_dotenv()

# Step 1: Initialize the client exactly as you specified
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 2: Use the model that worked in your list
MODEL_ID = "gemini-2.5-flash-lite"

def _extract_json(text: str):
    """
    Cleans the LLM output. It finds the JSON block even if 
    the model adds 'Here is the JSON:' or markdown fences.
    """
    try:
        # Finds everything between the first { and the last }
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
        # Step 3: API Call using your required structure
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={'temperature': 0} # Ensures deterministic output
        )
        data = _extract_json(response.text)
        return data.get("queries", [user_query])
    except Exception as e:
        print(f"[Researcher] Sub-query generation failed: {e}")
        return [user_query]

def extract_claims(page_text: str, criteria: str, url: str) -> list[dict]:
    # We truncate the page text to ensure we don't hit token limits
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
        # Step 3: API Call using your required structure
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={'temperature': 0}
        )
        data = _extract_json(response.text)
        claims = data.get("claims", [])
        
        # Tag each claim with the source URL
        for claim in claims:
            claim["source_url"] = url
        return claims
    except Exception as e:
        print(f"[Researcher] Claim extraction failed for {url}: {e}")
        return []

def researcher_node(state: dict) -> dict:
    """The entry point for the pipeline."""
    # Mandatory CFG Transition
    record_transition("Researcher")
    
    user_query = state.get("user_query", "")
    print(f"\n[Researcher] Processing query: {user_query}")

    # 1. Generate Sub-Queries
    sub_queries = generate_sub_queries(user_query)
    state["sub_queries"] = sub_queries
    print(f"[Researcher] Sub-queries: {sub_queries}")

    # 2. Retrieval & Deduplication
    index = SearchIndex("corpus_index.json")
    seen_urls = set()
    unique_results = []

    for sq in sub_queries:
        hits = index.query(sq, top_k=3)
        for hit in hits:
            if hit.url not in seen_urls:
                seen_urls.add(hit.url)
                unique_results.append(hit)

    top_hits = unique_results[:5]
    state["retrieved_pages"] = [hit.url for hit in top_hits]

    # 3. Content Extraction
    all_claims = []
    for hit in top_hits:
        print(f"[Researcher] Extracting claims from: {hit.url}")
        content = index.fetch_content(hit.url)
        page_claims = extract_claims(content, user_query, hit.url)
        all_claims.extend(page_claims)

    state["claims"] = all_claims
    print(f"[Researcher] Done. Found {len(all_claims)} claims.")
    
    return state