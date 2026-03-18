import os
import json
import re
from dotenv import load_dotenv
from google import genai
from pipeline.cfg import record_transition

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash-lite"

def _extract_json(text: str):
    """Reuse the robust JSON extractor from the Researcher."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Analyzer JSON Parsing Error: {e}")

def analyzer_node(state: dict) -> dict:
    # 1. Mandatory CFG Guardrail (Researcher -> Analyzer)
    record_transition("Analyzer")

    user_query = state.get("user_query", "")
    claims = state.get("claims", [])

    print(f"\n[Analyzer] Analyzing {len(claims)} claims for query: {user_query}")

    # 2. Group claims by hotel and then by criterion
    # Structure: { "Hotel Name": { "transit_access": [], "ratings": [] } }
    grouped_evidence = {}
    for claim in claims:
        hotel = claim["hotel"]
        crit = claim["criterion"]
        
        if hotel not in grouped_evidence:
            grouped_evidence[hotel] = {}
        if crit not in grouped_evidence[hotel]:
            grouped_evidence[hotel][crit] = []
            
        grouped_evidence[hotel][crit].append(claim["text"])

    # 3. Format the grouped claims into a readable prompt structure
    evidence_block = ""
    for hotel, criteria in grouped_evidence.items():
        evidence_block += f"\n### HOTEL: {hotel}\n"
        for crit, texts in criteria.items():
            evidence_block += f"  - {crit.upper()}:\n"
            for t in texts:
                evidence_block += f"    * {t}\n"

    # 4. Construct the Ranking Prompt
    prompt = f"""
    USER QUERY / CRITERIA: {user_query}

    EVIDENCE GATHERED FROM RESEARCH:
    {evidence_block}

    INSTRUCTIONS:
    1. Rank the hotels mentioned based ONLY on how well they satisfy the User Criteria.
    2. Score each hotel based solely on the provided claims. 
    3. IMPORTANT: Do not use any prior knowledge about these hotels or Bangkok. 
    4. If no evidence exists for a specific hotel/criterion, assume it does not meet that criterion.
    5. Base your ranking only on the evidence provided above.

    Return a JSON object in this exact format:
    {{
      "rankings": ["Hotel A", "Hotel B", "Hotel C"],
      "reasoning": "A brief explanation of why each hotel ranked where it did, citing specific evidence from the claims."
    }}
    Return ONLY valid JSON.
    """

    # 5. Call Gemini
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={'temperature': 0}
        )
        data = _extract_json(response.text)
        
        # 6. Store in state
        state["rankings"] = data.get("rankings", [])
        state["reasoning"] = data.get("reasoning", "")
        
        print(f"[Analyzer] Successfully ranked {len(state['rankings'])} hotels.")
        
    except Exception as e:
        print(f"[Analyzer] Critical failure: {e}")
        state["rankings"] = []
        state["reasoning"] = "Analysis failed due to an error in processing."

    return state