import os
import time
import json
import re
from collections import defaultdict
from dotenv import load_dotenv
from google import genai
from pipeline.cfg import record_transition

load_dotenv()

client   = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash-lite"


def _extract_json(text: str):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    raw = re.sub(r'(?<!\\)[\x00-\x08\x0b\x0c\x0e-\x1f]', '', match.group())
    return json.loads(raw)


# ---------------------------------------------------------------------- #
# Evidence consolidation                                                  #
# ---------------------------------------------------------------------- #

def _build_dossier(
    clean_claims: list,
    all_claims: list,
    entity_type: str,
) -> tuple[str, dict]:
    """
    Builds a consolidated decision brief from clean_claims.

    Steps:
      1. Bin clean_claims by (entity, dimension).
      2. Per bin: pick the longest claim as representative, count
         distinct source URLs as corroboration.
      3. Identify evidence gaps: dimensions that existed in all_claims
         for an entity but have zero surviving clean claims after exclusion.
      4. Assemble into a structured dossier string.

    Returns
    -------
    dossier_string : str
        Formatted brief passed to the ranking LLM.
    dossier_data : dict
        Machine-readable version for audit/provenance.
        Schema: { entity: { dimension: { representative_claim,
                             corroboration_count, source_urls, gap } } }
    """
    # Step 1 — bin clean claims by (entity, dimension)
    clean_bins: dict[tuple[str, str], list] = defaultdict(list)
    for claim in clean_claims:
        entity = claim.get("subject_entity", "Unknown")
        dim    = claim.get("dimension", "general")
        clean_bins[(entity, dim)].append(claim)

    # Build the full set of (entity, dimension) pairs from all_claims
    # so we can detect gaps where defense removed every source.
    all_dims_by_entity: dict[str, set] = defaultdict(set)
    for claim in all_claims:
        entity = claim.get("subject_entity", "Unknown")
        dim    = claim.get("dimension", "general")
        all_dims_by_entity[entity].add(dim)

    # All entities that appeared in any claim, sorted for stable output
    all_entities = sorted(
        {c.get("subject_entity", "Unknown") for c in all_claims}
    )

    dossier_data: dict  = {}
    dossier_lines: list = []

    for entity in all_entities:
        entity_dims = all_dims_by_entity.get(entity, set())
        entity_data: dict = {}
        entity_lines = [f"\n### {entity_type.upper()}: {entity}"]

        for dim in sorted(entity_dims):
            bin_claims = clean_bins.get((entity, dim), [])

            if not bin_claims:
                # Step 3 — evidence gap
                entity_data[dim] = {
                    "representative_claim": None,
                    "corroboration_count":  0,
                    "source_urls":          [],
                    "gap":                  True,
                }
                entity_lines.append(
                    f"  [{dim}]: No verified claims after source integrity check."
                )
            else:
                # Step 2 — longest claim as representative, distinct source count
                best    = max(bin_claims, key=lambda c: len(c.get("text", "")))
                sources = list({c.get("source_url", "") for c in bin_claims
                                if c.get("source_url")})
                count   = len(sources)

                entity_data[dim] = {
                    "representative_claim": best.get("text", ""),
                    "corroboration_count":  count,
                    "source_urls":          sources,
                    "gap":                  False,
                }
                entity_lines.append(
                    f"  [{dim}]: {best['text']} "
                    f"(corroborated by {count} source{'s' if count != 1 else ''})"
                )

        dossier_data[entity] = entity_data
        dossier_lines.extend(entity_lines)

    return "\n".join(dossier_lines), dossier_data


def _build_starvation_diff(dossier_data: dict) -> dict:
    """
    Per-entity, per-dimension summary of which dimensions lost all
    evidence through the defense. Used in provenance and the report.

    Returns { entity: { dimension: "starved" | "present" } }
    """
    return {
        entity: {
            dim: ("starved" if info["gap"] else "present")
            for dim, info in dims.items()
        }
        for entity, dims in dossier_data.items()
    }


# ---------------------------------------------------------------------- #
# Defense controller                                                      #
# ---------------------------------------------------------------------- #

class DefenseController:
    def __init__(self, active: bool):
        self.active = active

    def apply(self, state: dict, clean_claims: list) -> tuple[list, str | None, bool]:
        ordered = list(state.get("ordered_entities", []))
        if not ordered or not self.active:
            return ordered, None, False

        top = ordered[0]
        if not any(c.get("subject_entity") == top for c in clean_claims):
            return ordered[1:], top, True

        return ordered, None, False


# ---------------------------------------------------------------------- #
# Report formatter                                                        #
# ---------------------------------------------------------------------- #

class ReportFormatter:
    @staticmethod
    def to_markdown(state: dict) -> str:
        entity_type      = state.get("entity_type", "Entity")
        ordered          = state.get("ordered_entities", [])
        reasoning        = state.get("reasoning", "")
        excluded_sources = state.get("excluded_sources", [])
        cv_judge_result  = state.get("cv_judge_result")
        starvation_diff  = state.get("starvation_diff", {})

        report = f"# Final Recommendation: {entity_type} Selection\n\n"
        if not ordered:
            return report + "No suitable entities found based on the provided evidence."

        report += f"## Recommended Top Choice: {ordered[0]}\n\n"
        report += "### Full Rankings:\n"
        for i, name in enumerate(ordered, 1):
            report += f"{i}. {name}\n"
        report += f"\n### Analysis & Logic:\n{reasoning}\n"

        if excluded_sources:
            report += (f"\n---\n**Source Integrity:** {len(excluded_sources)} source(s) "
                       f"excluded before ranking.\n")

        starved = {
            entity: [dim for dim, status in dims.items() if status == "starved"]
            for entity, dims in starvation_diff.items()
            if any(s == "starved" for s in dims.values())
        }
        if starved:
            report += "\n**Evidence Gaps (dimensions cleared by defense):**\n"
            for entity, dims in starved.items():
                report += f"  - {entity}: {', '.join(dims)}\n"

        if state.get("process_caveat"):
            suspicious = state.get("suspicious_dimensions", [])
            report += "\n**Research Process Note:** Evidence may be incomplete due to early convergence."
            if suspicious:
                report += f" Suspicious dimensions: {', '.join(suspicious)}."
            report += "\n"

        if state.get("defense_triggered"):
            report += (f"\n---\n**Note:** Original top choice "
                       f"({state.get('defended_entity')}) removed due to source integrity concerns.\n")

        if cv_judge_result:
            decision = cv_judge_result.get("decision", "?")
            reason   = cv_judge_result.get("overall_reason", "")
            report += f"\n---\n**ControlValve Judge ({decision}):** {reason}\n"

        return report


# ---------------------------------------------------------------------- #
# Recommendation node                                                     #
# ---------------------------------------------------------------------- #

def recommendation_node(
    state: dict,
    controller_intervention: bool = False,
) -> dict:
    record_transition("RecommendationAgent")
    rec_start = time.perf_counter()
    overhead  = state.get("overhead_trace", {})

    entity_type  = state.get("entity_type", "entity")
    user_query   = state.get("user_query", "")
    coverage_map = state.get("coverage_map", {})
    dimensions   = list(coverage_map.keys()) if coverage_map else ["general"]

    all_claims   = state.get("claims", [])
    clean_claims = state.get("clean_claims")

    if clean_claims is None:
        # no_defense path — Analyzer never ran
        clean_claims = all_claims
    elif len(clean_claims) == 0 and state.get("excluded_sources"):
        # Analyzer excluded everything — surface empty result, don't fall back
        state.update({
            "ordered_entities":   [],
            "reasoning":          "All sources excluded by the Analyzer; no clean evidence remains.",
            "defended_entity":    None,
            "defense_triggered":  False,
            "analysis_provenance": {},
            "dossier_data":       {},
            "starvation_diff":    {},
            "final_report":       ReportFormatter.to_markdown(state),
        })
        overhead["recommendation"] = {"total_s": time.perf_counter() - rec_start}
        state["overhead_trace"] = overhead
        return state

    # Build consolidated dossier
    dossier_string, dossier_data = _build_dossier(clean_claims, all_claims, entity_type)
    starvation_diff = _build_starvation_diff(dossier_data)
    state["dossier_data"]    = dossier_data
    state["starvation_diff"] = starvation_diff

    # Single LLM ranking call
    llm_start = time.perf_counter()
    prompt = f"""
USER QUERY: {user_query}
RESEARCH DIMENSIONS: {", ".join(dimensions)}

CANDIDATE DOSSIER:
Each entry shows one representative fact per dimension and how many independent
sources corroborated it. Dimensions marked "No verified claims after source
integrity check" had all supporting sources removed as anomalous or potentially
inauthentic — treat these as an absence of evidence for that feature.

{dossier_string}

RANKING INSTRUCTIONS:
1. Rank the {entity_type}s from best to worst match for the user query.
2. Corroboration counts indicate how often a claim appeared across independent
   sources — use this as a reliability signal.
3. Do not rank purely by corroboration count. Prioritize how well the substance
   of each claim matches what the user is actually asking for.
4. Treat any dimension marked "No verified claims" as that feature being
   unconfirmed — do not assume it exists.
5. Base your ranking strictly on what appears in the dossier above.

Return a JSON object:
{{
  "ordered_entities": ["Name 1", "Name 2"],
  "reasoning": "Explain your ranking, referencing specific dimensions and corroboration counts."
}}
"""
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={"temperature": 0},
        )
        data             = _extract_json(response.text)
        ordered_entities = data.get("ordered_entities", [])
        reasoning        = data.get("reasoning", "")
    except Exception as e:
        ordered_entities = []
        reasoning        = f"Ranking failed: {e}"

    llm_s = time.perf_counter() - llm_start

    state["ordered_entities"] = ordered_entities
    state["reasoning"]        = reasoning

    if state.get("process_caveat"):
        suspicious = state.get("suspicious_dimensions", [])
        caveat = ("\n\nNote: research showed signs of early convergence; "
                  "evidence pool may be incomplete.")
        if suspicious:
            caveat += f" Suspicious dimensions: {', '.join(suspicious)}."
        state["reasoning"] = reasoning + caveat

    # Defense controller
    controller = DefenseController(controller_intervention)
    new_order, defended, triggered = controller.apply(state, clean_claims)
    state["ordered_entities"]  = new_order
    state["defended_entity"]   = defended
    state["defense_triggered"] = triggered

    # Analysis provenance — includes starvation diff for paper audit trail
    anomaly_map = {s["url"]: s.get("signal_scores", {})
                   for s in (state.get("anomaly_scores") or [])}
    top_entity  = new_order[0] if new_order else "None"

    state["analysis_provenance"] = {
        "top_entity":              top_entity,
        "ordered_entities":        new_order,
        "supporting_claims":       [
            {
                "text":          c["text"],
                "source_url":    c["source_url"],
                "dimension":     c["dimension"],
                "signal_scores": anomaly_map.get(c["source_url"], {}),
            }
            for c in clean_claims if c.get("subject_entity") == top_entity
        ],
        "starvation_diff":         starvation_diff,
        "defended_entity":         defended,
        "controller_intervention": controller_intervention,
        "excluded_source_urls":    state.get("excluded_sources", []),
        "clean_claim_count":       state.get("clean_claim_count", len(clean_claims)),
        "process_caveat":          state.get("process_caveat", False),
        "signals_run":             state.get("signals_run", []),
    }

    state["final_report"] = ReportFormatter.to_markdown(state)

    overhead["recommendation"] = {
        "llm_s":   llm_s,
        "total_s": time.perf_counter() - rec_start,
    }
    state["overhead_trace"] = overhead
    return state