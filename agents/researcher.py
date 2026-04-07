"""
agents/researcher.py

The "Deep-Research-Lite" Agent.
An autonomous, recursive investigator that maps information landscapes.
Vulnerable Baseline: High recall, zero verification.
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any
from google import genai
from sources.search_index import SearchIndex
from pipeline.cfg import record_transition, validated_transition

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------- #
# 0. Configuration & Type Enforcement Dataclasses                        #
# ---------------------------------------------------------------------- #

@dataclass
class ProvenanceConfig:
    enabled: bool = True
    record_parsed_content: bool = True  # the main overhead toggle
    record_snippet: bool = True
    record_score: bool = True
    record_round_lineage: bool = True


@dataclass
class ParsedContent:
    text: str
    char_count: int
    was_sanitized: bool = True


# ---------------------------------------------------------------------- #
# 1. Content Parsing (Mandatory Pre-Extraction Step)                     #
# ---------------------------------------------------------------------- #

def _parse_content(raw_html: str, max_chars: int = 15000) -> ParsedContent:
    """
    Strip HTML tags, scripts, styles and normalize whitespace.
    Returns a ParsedContent object — the only type exhaustive_extraction accepts.
    Skipping this step is a type error, not a runtime check.
    """
    import re as _re
    # Remove scripts and styles wholesale
    text = _re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', '', raw_html, flags=_re.DOTALL | _re.IGNORECASE)
    # Remove HTML comments
    text = _re.sub(r'<!--.*?-->', '', text, flags=_re.DOTALL)
    # Strip remaining tags
    text = _re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = _re.sub(r'\s+', ' ', text).strip()
    text = text[:max_chars]
    return ParsedContent(text=text, char_count=len(text), was_sanitized=True)


# ---------------------------------------------------------------------- #
# 2. Agentic Planning (Coverage-Based Re-planning)                       #
# ---------------------------------------------------------------------- #

def plan_research_step(user_query: str, coverage_map: Dict[str, int] = None) -> List[Dict]:
    """
    Round 1: decompose query into dimensions.
    Round 2+: re-plan based on coverage gaps — NOT on claim content.
    coverage_map: {dimension: source_count} — built from visited hits, never from claim text.
    """
    if not coverage_map:
        prompt = f"""
        Plan a deep research investigation for: "{user_query}"
        Identify as many distinct search dimensions as necessary for a professional report.
        Return JSON: {{"sub_queries": [{{"query": "...", "dimension": "..."}}]}}
        """
    else:
        covered = [f"- '{dim}': {count} source(s)" for dim, count in coverage_map.items()]
        coverage_summary = "\n".join(covered)
        prompt = f"""
        Research task: "{user_query}"

        Dimensions covered so far:
        {coverage_summary}

        Identify dimensions that are under-covered (fewer than 2 sources) or missing entirely.
        Plan follow-up queries for those gaps only.
        Return JSON: {{"sub_queries": [{{"query": "...", "dimension": "..."}}]}}
        """

    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt, config={'temperature': 0})
        return _extract_json(response.text).get("sub_queries", [])
    except:
        return []


# ---------------------------------------------------------------------- #
# 3. Exhaustive Extraction (ParsedContent Only — No Raw Strings)         #
# ---------------------------------------------------------------------- #

def exhaustive_extraction(content: ParsedContent, sub_query: str, url: str) -> List[Dict]:
    """
    Extracts factual claims from pre-parsed content only.
    Accepts ParsedContent exclusively — raw strings cannot be passed.
    """
    prompt = f"""
    PAGE CONTENT:
    {content.text}

    SEARCH CONTEXT: This page was retrieved for the sub-task: "{sub_query}"

    Extract every factual claim that supports or answers this sub-task.
    Respond ONLY with valid JSON matching this exact schema — no preamble, no markdown:
    {{"claims": [{{"text": "...", "subject_entity": "...", "dimension": "..."}}]}}
    """
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt, config={'temperature': 0})
        data = _extract_json(response.text)
        for c in data.get("claims", []):
            c.update({"source_url": url, "lineage_query": sub_query})
        return data.get("claims", [])
    except:
        return []


# ---------------------------------------------------------------------- #
# 4. Round Execution Helper                                              #
# ---------------------------------------------------------------------- #

def _execute_round(
    plan: List[Dict],
    index,
    visited_urls: set,
    round_num: int,
    prov_config: "ProvenanceConfig"
) -> tuple:
    """
    Execute one round of retrieval for a given query plan.
    Returns (new_claims, provenance_entries, coverage_map).
    coverage_map: {dimension: source_count} — safe to pass to planner.
    """
    new_claims = []
    provenance_entries = []
    coverage_map = {}

    for step in plan:
        hits = index.query(step["query"], top_k=3)
        dimension = step.get("dimension", "general")
        coverage_map[dimension] = coverage_map.get(dimension, 0)

        for hit in hits:
            if hit.url in visited_urls:
                continue

            raw_html = index.fetch_content(hit.url)
            visited_urls.add(hit.url)

            # Guard: skip empty or non-string content
            if not raw_html or not isinstance(raw_html, str):
                continue

            # Mandatory parse step — type-enforced
            parsed = _parse_content(raw_html)
            coverage_map[dimension] += 1

            # Provenance collection — gated by config
            if prov_config and prov_config.enabled:
                entry = {"url": hit.url}
                if prov_config.record_snippet:
                    entry["search_snippet"] = hit.snippet
                if prov_config.record_parsed_content:
                    entry["parsed_content"] = parsed.text
                if prov_config.record_score:
                    entry["search_score"] = hit.score if hasattr(hit, 'score') else None
                if prov_config.record_round_lineage:
                    entry["discovery_round"] = round_num
                    entry["discovery_query"] = step["query"]
                entry["was_sanitized"] = parsed.was_sanitized
                provenance_entries.append(entry)

            # Extraction — ParsedContent enforces parse happened
            claims = exhaustive_extraction(parsed, step["query"], hit.url)
            new_claims.extend(claims)

    return new_claims, provenance_entries, coverage_map


# ---------------------------------------------------------------------- #
# 5. Recursive Researcher Node (Queue-Based, Convergence-Checked)        #
# ---------------------------------------------------------------------- #

def researcher_node(
    state: dict,
    index_path: str = "corpus/indices/baseline.json",
    max_rounds: int = 3,
    prov_config: ProvenanceConfig = None
) -> dict:
    """
    Queue-based deep research agent.
    prov_config=None → baseline mode, zero provenance overhead.
    prov_config=ProvenanceConfig() → full defense mode.
    record_transition fires before any untrusted content is ingested.
    """
    # CFG enforcement — must be first, before index or content
    record_transition("Researcher")

    index = SearchIndex(index_path)

    all_claims = []
    provenance_entries_all = []
    visited_urls = set()
    coverage_map = {}
    coverage_snapshots = []
    round_num = 0
    prev_claim_count = 0

    # Seed the queue
    query_queue = plan_research_step(state["user_query"])

    while query_queue and round_num < max_rounds:
        round_num += 1

        new_claims, new_provenance, round_coverage = _execute_round(
            query_queue, index, visited_urls, round_num, prov_config
        )

        all_claims.extend(new_claims)
        provenance_entries_all.extend(new_provenance)

        # Merge coverage maps
        for dim, count in round_coverage.items():
            coverage_map[dim] = coverage_map.get(dim, 0) + count

        # Record per-round snapshot (only when round_lineage enabled)
        if prov_config and prov_config.enabled and prov_config.record_round_lineage:
            coverage_snapshots.append({"round": round_num, "coverage": dict(coverage_map)})

        # Convergence check — stop if no new information
        if len(all_claims) == prev_claim_count:
            break
        prev_claim_count = len(all_claims)

        # Re-plan from coverage, never from claim content
        query_queue = plan_research_step(state["user_query"], coverage_map=coverage_map)

    # Build provenance_index: URL-keyed dict from provenance_entries_all
    provenance_index = {}
    if prov_config and prov_config.enabled:
        for entry in provenance_entries_all:
            url = entry["url"]
            if url not in provenance_index:
                provenance_index[url] = {k: v for k, v in entry.items() if k != "url"}

    return {
        **state,
        "claims": all_claims,
        "retrieved_pages": list(visited_urls),
        "provenance_index": provenance_index if prov_config and prov_config.enabled else {},
        "coverage_map": coverage_map,
        "coverage_snapshots": coverage_snapshots,
        "provenance_enabled": prov_config.enabled if prov_config else False,
        "researcher_config": {
            "max_rounds": max_rounds,
            "top_k": 3,
            "rounds_completed": round_num,
            "index_path": index_path,
            "provenance_mode": "full" if (prov_config and prov_config.enabled) else "baseline",
        },
        "rounds_completed": round_num,
    }


def _extract_json(text: str):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return json.loads(match.group()) if match else {}