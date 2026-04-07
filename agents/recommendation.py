"""
agents/recommendation.py

The Recommendation Agent: performs LLM ranking over clean evidence,
applies intervention logic via a DefenseController, and generates the
final output via a ReportFormatter.
"""

import os
import time
import json
import re
from dotenv import load_dotenv
from google import genai
from pipeline.cfg import record_transition

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #

def _extract_json(text: str):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group())
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"RecommendationAgent JSON Parsing Error: {e}")

# ---------------------------------------------------------------------- #
# Pluggable Defense Controller                                           #
# ---------------------------------------------------------------------- #

class DefenseController:
    """
    Checks whether the top-ranked entity from the LLM has any remaining
    claims after the analyzer's source exclusion pass.
    - 'no_defense': pass through everything unchanged.
    - 'full_system': if top entity has zero clean claims, pop it and record.
    - 'observe': record that defense would have triggered but do not modify output.
    """
    def __init__(self, config: str):
        self.config = config

    def apply_intervention(self, state: dict, clean_claims: list) -> tuple:
        ordered = list(state.get("ordered_entities", []))
        defense_triggered = False
        defended_entity = None

        if not ordered or self.config == "no_defense":
            return ordered, None, False

        top_entity = ordered[0]
        top_entity_clean_sources = {
            c["source_url"] for c in clean_claims if c.get("subject_entity") == top_entity
        }

        # Top entity has zero claims after exclusion — degenerate case
        has_clean_claims = len(top_entity_clean_sources) > 0

        if not has_clean_claims and self.config == "full_system":
            defense_triggered = True
            defended_entity = ordered.pop(0)

        elif not has_clean_claims and self.config == "observe":
            defense_triggered = False
            defended_entity = f"PREDICTED_REMOVAL: {top_entity}"

        return ordered, defended_entity, defense_triggered

# ---------------------------------------------------------------------- #
# Pluggable Report Formatter                                              #
# ---------------------------------------------------------------------- #

class ReportFormatter:
    """
    Generates human-readable summaries.
    """
    @staticmethod
    def to_markdown(state: dict, domain_context: dict) -> str:
        entity_type = domain_context.get("entity_type", "Entity")
        ordered = state.get("ordered_entities", [])
        reasoning = state.get("reasoning", "")
        excluded_sources = state.get("excluded_sources", [])
        process_caveat = state.get("process_caveat", False)

        report = f"# Final Recommendation: {entity_type} Selection\n\n"

        if not ordered:
            return report + "No suitable entities found based on the provided evidence."

        report += f"## Recommended Top Choice: {ordered[0]}\n\n"
        report += f"### Full Rankings:\n"
        for i, name in enumerate(ordered, 1):
            report += f"{i}. {name}\n"

        report += f"\n### Analysis & Logic:\n{reasoning}\n"

        if excluded_sources:
            report += (
                f"\n---\n**Source Integrity:** {len(excluded_sources)} source(s) were excluded "
                f"from evidence before ranking.\n"
            )

        if process_caveat:
            report += (
                "\n**Research Process Note:** This recommendation was produced from evidence "
                "that may be incomplete due to early convergence in the research process.\n"
            )

        if state.get("defense_triggered"):
            report += (
                f"\n---\n**Note:** The original top choice ({state.get('defended_entity')}) "
                f"was removed due to source integrity concerns.\n"
            )

        return report

# ---------------------------------------------------------------------- #
# Recommendation Node                                                     #
# ---------------------------------------------------------------------- #

def recommendation_node(
    state: dict,
    defense_config: str = "no_defense",
    domain_context: dict = None,
    signal_config=None
) -> dict:
    """
    Final agent node: LLM ranking over clean evidence, defense logic, and report.
    """
    # --- STEP 1: CFG + TIMER ---
    record_transition("RecommendationAgent")
    rec_start = time.perf_counter()
    overhead = state.get("overhead_trace", {})
    domain_context = domain_context or {"entity_type": "entity"}
    entity_type = domain_context.get("entity_type", "entity")

    # --- STEP 2: BUILD CLEAN CLAIMS ---
    excluded_sources = state.get("excluded_sources", [])
    excluded_set = set(excluded_sources)
    all_claims = state.get("claims", [])
    clean_claims = [c for c in all_claims if c.get("source_url") not in excluded_set]

    # --- STEP 3: BUILD CLEAN EVIDENCE STRING ---
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

    coverage_map = state.get("coverage_map", {})
    dimensions = list(coverage_map.keys()) if coverage_map else ["general"]
    user_query = state.get("user_query", "")

    # --- STEP 4: LLM RANKING CALL ---
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
        reasoning = f"Ranking failed: {str(e)}"

    state["ordered_entities"] = ordered_entities
    state["reasoning"] = reasoning

    # --- STEP 5: ATTACH CHECK 3 CAVEAT ---
    if state.get("process_caveat"):
        coverage_entropy_result = state.get("coverage_entropy_result", {})
        suspicious = coverage_entropy_result.get("suspicious_dimensions", [])
        caveat = (
            "\n\nNote: the research process showed signs of early convergence. "
            "The evidence pool may be incomplete as some dimensions were covered "
            "by fewer sources than expected."
        )
        if suspicious:
            caveat += f" Suspicious dimensions: {', '.join(suspicious)}."
        state["reasoning"] = reasoning + caveat

    # --- STEP 6: APPLY DEFENSE CONTROLLER ---
    controller = DefenseController(defense_config)
    new_order, defended, triggered = controller.apply_intervention(state, clean_claims)

    state["ordered_entities"] = new_order
    state["defended_entity"] = defended
    state["defense_triggered"] = triggered

    # --- STEP 7: ENRICH ANALYSIS PROVENANCE ---
    anomaly_scores = state.get("anomaly_scores") or []
    anomaly_map = {s["url"]: s.get("signal_scores", {}) for s in anomaly_scores}

    top_entity = new_order[0] if new_order else "None"
    supporting_claims = [
        {
            "text": c["text"],
            "source_url": c["source_url"],
            "dimension": c["dimension"],
            "signal_scores": anomaly_map.get(c["source_url"], {}),
        }
        for c in clean_claims if c.get("subject_entity") == top_entity
    ]

    state["analysis_provenance"] = {
        "top_entity": top_entity,
        "ordered_entities": new_order,
        "supporting_claims": supporting_claims,
        "defended_entity": defended,
        "defense_config": defense_config,
        "excluded_source_urls": list(excluded_set),
        "clean_claim_count": state.get("clean_claim_count", len(clean_claims)),
        "process_caveat": state.get("process_caveat", False),
        "signals_run": state.get("signals_run", []),
    }

    # --- STEP 8: FORMAT FINAL REPORT ---
    formatter = ReportFormatter()
    state["final_report"] = formatter.to_markdown(state, domain_context)

    # --- STEP 9: RECORD OVERHEAD ---
    overhead["recommendation"] = {
        "total_s": time.perf_counter() - rec_start,
    }
    state["overhead_trace"] = overhead

    return state