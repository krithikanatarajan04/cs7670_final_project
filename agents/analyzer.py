import os
import time
import json
import re
import numpy as np
from dotenv import load_dotenv
from google import genai
from pipeline.cfg import validated_transition, validate_anomaly_scores_present

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------- #
# Adaptive Anomaly Profile (moved from auditor)                          #
# ---------------------------------------------------------------------- #

class AnomalyProfile:
    """
    Calculates thresholds based on the statistical distribution of the
    current research batch rather than hardcoded 'magic numbers'.
    """
    def __init__(self, values: list, sensitivity: float = 2.0):
        self.count = len(values)
        if self.count >= 3:
            self.mean = np.mean(values)
            self.std = np.std(values)
            self.threshold = self.mean + (sensitivity * (self.std if self.std > 0 else 1))
        else:
            self.mean = 0.5
            self.threshold = 0.85

    def is_outlier(self, val: float) -> bool:
        return val > self.threshold

# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #

def _extract_json(text: str):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group())
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Analyzer JSON Parsing Error: {e}")

# ---------------------------------------------------------------------- #
# Analyzer Node                                                          #
# ---------------------------------------------------------------------- #

def analyzer_node(state: dict, domain_context: dict = None) -> dict:
    """
    Ranks entities over clean evidence only.
    1. CFG validation.
    2. Check 3 — process-level fast convergence caveat.
    3. Rank sources by joint z-scores (Check 1+2a, Check 2).
    4. Apply source exclusion decisions.
    5. LLM ranking over clean evidence — no flags exposed to LLM.
    6. Attach Check 3 caveat to reasoning if needed.
    7. Write state with overhead trace.
    """
    # --- STEP 1: CFG VALIDATION ---
    validated_transition("Analyzer", state, [validate_anomaly_scores_present])
    analyzer_start = time.perf_counter()
    overhead = state.get("overhead_trace", {})

    domain_context = domain_context or {"entity_type": "entity"}
    entity_type = domain_context.get("entity_type", "entity")

    # --- STEP 2: READ INPUTS ---
    anomaly_scores = state.get("anomaly_scores", [])
    coverage_entropy_result = state.get("coverage_entropy_result", {})
    claims = state.get("claims", [])
    user_query = state.get("user_query", "")
    coverage_map = state.get("coverage_map", {})
    signals_run = state.get("signals_run", [])

    score_map = {entry["url"]: entry for entry in anomaly_scores}

    # --- STEP 3: CHECK 3 — PROCESS LEVEL ---
    check3_start = time.perf_counter()
    fast_convergence_flagged = coverage_entropy_result.get("fast_convergence_flagged", False)
    process_caveat = fast_convergence_flagged
    check3_s = time.perf_counter() - check3_start

    # --- STEP 4: RANK SOURCES BY JOINT SCORES ---
    ranking_start = time.perf_counter()
    excluded_urls = set()

    joint_check1_scores = {
        url: entry["signal_scores"].get("joint_check1", 0.0)
        for url, entry in score_map.items()
    }
    joint_check2_scores = {
        url: entry["signal_scores"].get("joint_check2", 0.0)
        for url, entry in score_map.items()
    }

    check1_ranking = sorted(joint_check1_scores, key=joint_check1_scores.get, reverse=True)
    check2_ranking = sorted(joint_check2_scores, key=joint_check2_scores.get, reverse=True)

    # --- STEP 5: CHECK 1+2a DECISION ---
    if joint_check1_scores and "snippet_divergence" in signals_run and "focus" in signals_run:
        most_extreme_check1 = check1_ranking[0]
        excluded_urls.add(most_extreme_check1)

    # --- STEP 6: CHECK 2 DECISION ---
    if joint_check2_scores and "focus" in signals_run and "corroboration" in signals_run:
        most_extreme_check2 = check2_ranking[0]
        if most_extreme_check2 not in excluded_urls:
            excluded_urls.add(most_extreme_check2)

    ranking_s = time.perf_counter() - ranking_start

    # --- STEP 7: EXCLUSION PASS ---
    exclusion_start = time.perf_counter()
    clean_claims = [c for c in claims if c.get("source_url") not in excluded_urls]
    exclusion_s = time.perf_counter() - exclusion_start

    if not clean_claims:
        state["ordered_entities"] = []
        state["reasoning"] = (
            "All sources were excluded by defense checks — "
            "insufficient clean evidence to produce a recommendation."
        )
        state["excluded_sources"] = list(excluded_urls)
        state["clean_claim_count"] = 0
        state["process_caveat"] = process_caveat
        state["flagged_sources_pre_ranking"] = [
            score_map[url] for url in excluded_urls if url in score_map
        ]
        overhead["analyzer"] = {
            "check3_s": check3_s,
            "ranking_s": ranking_s,
            "exclusion_s": exclusion_s,
            "llm_s": 0.0,
            "total_s": time.perf_counter() - analyzer_start,
        }
        state["overhead_trace"] = overhead
        return state

    # --- STEP 8: BUILD CLEAN EVIDENCE GROUPED BY ENTITY ---
    grouped_evidence = {}
    for claim in clean_claims:
        ent = claim.get("subject_entity", "Unknown")
        if ent not in grouped_evidence:
            grouped_evidence[ent] = []
        grouped_evidence[ent].append(f"* {claim.get('text', '')}")

    evidence_string = ""
    for ent, lines in grouped_evidence.items():
        evidence_string += f"\n### {entity_type.upper()}: {ent}\n"
        evidence_string += "\n".join(lines) + "\n"

    dimensions = list(coverage_map.keys()) if coverage_map else ["general"]

    # --- STEP 9: LLM RANKING CALL ---
    llm_start = time.perf_counter()
    prompt = f"""
USER QUERY: {user_query}
DIMENSIONS: {", ".join(dimensions)}

EVIDENCE:
{evidence_string}

Rank the {entity_type}s from best to worst based on the evidence above.

Return a JSON object:
{{
  "ordered_entities": ["Name 1", "Name 2"],
  "reasoning": "Explain your ranking based on the evidence."
}}
"""
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={'temperature': 0}
        )
        data = _extract_json(response.text)
        ordered_entities = data.get("ordered_entities", [])
        reasoning = data.get("reasoning", "")
    except Exception as e:
        ordered_entities = []
        reasoning = f"Analysis failed: {str(e)}"

    llm_s = time.perf_counter() - llm_start

    # --- STEP 10: ATTACH CHECK 3 CAVEAT IF NEEDED ---
    if process_caveat:
        suspicious = coverage_entropy_result.get("suspicious_dimensions", [])
        caveat = (
            "\n\nNote: the research process showed signs of early convergence. "
            "The evidence pool may be incomplete as some dimensions were covered "
            "by fewer sources than expected."
        )
        if suspicious:
            caveat += f" Suspicious dimensions: {', '.join(suspicious)}."
        reasoning += caveat

    # --- STEP 11: WRITE STATE ---
    state["ordered_entities"] = ordered_entities
    state["reasoning"] = reasoning
    state["excluded_sources"] = list(excluded_urls)
    state["clean_claim_count"] = len(clean_claims)
    state["process_caveat"] = process_caveat
    state["flagged_sources_pre_ranking"] = [
        score_map[url] for url in excluded_urls if url in score_map
    ]

    overhead["analyzer"] = {
        "check3_s": check3_s,
        "ranking_s": ranking_s,
        "exclusion_s": exclusion_s,
        "llm_s": llm_s,
        "total_s": time.perf_counter() - analyzer_start,
    }
    state["overhead_trace"] = overhead

    return state