import os
import re
import json
import time
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from sources.search_index import SearchIndex
from pipeline.cfg import record_transition

load_dotenv()

client   = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash-lite"


@dataclass
class RetrievalConfig:
    score_threshold: float = 0.5
    max_per_dimension: int = 10
    max_angles_per_round: int = 5


@dataclass
class ProvenanceConfig:
    enabled: bool = True
    record_parsed_content: bool = True
    record_snippet: bool = True
    record_score: bool = True
    record_round_lineage: bool = True


@dataclass
class PlanningConfig:
    production_mode: bool = False
    max_open_questions: int = 6
    summary_top_entities: int = 8


@dataclass
class ParsedContent:
    text: str
    char_count: int
    was_sanitized: bool = True


@dataclass
class PageManifest:
    entities: List[str]
    canonical_map: Dict[str, str]
    page_type: str


# ---------------------------------------------------------------------- #
# Utilities                                                               #
# ---------------------------------------------------------------------- #

def build_temp_corpus_index(base_index_path: str, appended_entries: list, tag: str = "temp") -> str:
    with open(base_index_path) as f:
        base_index = json.load(f)
    merged = deepcopy(base_index)
    merged.extend(appended_entries or [])
    fd, temp_path = tempfile.mkstemp(prefix=f"{tag}_", suffix=".json")
    os.close(fd)
    with open(temp_path, "w") as f:
        json.dump(merged, f, indent=2)
    return temp_path


def _extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {}
    raw = match.group()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        last = raw.rfind('}')
        try:
            return json.loads(raw[:last + 1])
        except json.JSONDecodeError:
            return {}


def _llm_json(prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={"temperature": 0},
            )
            return _extract_json(response.text) or {}
        except Exception as e:
            print(f"[researcher] LLM call failed (attempt {attempt+1}/{retries}): {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {}


def _parse_content(raw_text: str, max_chars: int = 15000) -> ParsedContent:
    text = re.sub(r'\s+', ' ', raw_text).strip()[:max_chars]
    return ParsedContent(text=text, char_count=len(text), was_sanitized=True)


# ---------------------------------------------------------------------- #
# Planner                                                                 #
# ---------------------------------------------------------------------- #

def initial_research_plan(
    user_query: str,
    entity_type: str = "entity",
    max_open_questions: int = 6,
) -> dict:
    prompt = f"""
You are starting research about {entity_type}s.

USER QUERY:
"{user_query}"

Decompose the query into search queries, one per distinct constraint type.
First identify the constraints present (e.g. style/vibe, location, feature, price tier).
Each query should surface candidates satisfying that specific constraint.

RULES:
- Generate 1-2 queries for simple queries, up to 5 for complex multi-constraint queries.
- Each query must target a genuinely DIFFERENT constraint type — not paraphrases.
- For location constraints, include the specific place name verbatim.
- For vibe/style constraints, preserve the user's exact adjectives.
- Prefer broad queries that surface multiple candidates.
- Do not invent candidate names.
- discovery_query should be a shorter keyword form of the same intent.

Return ONLY valid JSON:
{{"queries": [{{"query": "...", "discovery_query": "..."}}]}}
"""
    data = _llm_json(prompt)
    return {
        "criteria":         [],
        "open_questions":   [],
        "next_search_goal": "candidate_discovery",
        "reason":           "Initial decomposition of user query.",
        "queries":          (data.get("queries", []) or [])[:5],
    }


def plan_next_step(
    user_query: str,
    entity_type: str,
    state_summary: dict,
    search_history: list,
    max_angles: int,
    max_open_questions: int,
    production_mode: bool = False,
    working_memory: Optional[dict] = None,
) -> dict:
    narrative_block = ""
    contradictions_block = ""
    if working_memory:
        narrative = working_memory.get("narrative", "")
        if narrative:
            narrative_block = f"CURRENT KNOWLEDGE NARRATIVE:\n{narrative}\n"
        contradictions = working_memory.get("contradictions", [])
        if contradictions:
            contradictions_block = f"KNOWN CONTRADICTIONS:\n{json.dumps(contradictions, indent=2)}\n"

    mode_instruction = "" if production_mode else (
        "IMPORTANT: You are in RESEARCH MODE. Do not stop early. "
        "Look for more candidates or deeper evidence to ensure the best possible answer."
    )

    prompt = f"""
You are the Controller for a research loop about {entity_type}s.
USER QUERY: "{user_query}"

{narrative_block}
{contradictions_block}

TASK:
Evaluate if the current knowledge satisfies EVERY constraint in the user query:
1. CANDIDATE COUNT: Do we have at least 3-5 distinct, high-quality candidates that actually fit?
2. ATTRIBUTE COVERAGE: Do we have specific evidence for all adjectives?
3. LOCATION: Are these in the correct area?
4. UNCERTAINTY: Are there contradictions or low-confidence beliefs that need verification?

{mode_instruction}

RULES:
- If insufficient, generate queries to:
    a) Find new candidates if we have < 3.
    b) Search for specific missing attributes for known candidates.
    c) Resolve contradictions.
- If you have enough info AND production_mode is true, set "continue": false.
- Do not repeat these previous searches: {json.dumps([h['queries'] for h in search_history])}

Return ONLY valid JSON:
{{
  "continue": true,
  "reason": "...",
  "queries": [{{"query": "...", "discovery_query": "..."}}]
}}
"""
    data = _llm_json(prompt)
    cont    = True if not production_mode else bool(data.get("continue", True))
    queries = (data.get("queries", []) or [])[:max_angles]

    if not queries and cont:
        queries = [{"query": f"detailed reviews of {user_query}", "discovery_query": user_query}]

    return {
        "continue": cont,
        "reason":   data.get("reason", ""),
        "queries":  queries,
    }


# ---------------------------------------------------------------------- #
# URL collection                                                          #
# ---------------------------------------------------------------------- #

def _collect_urls(
    angles: List[dict],
    index,
    visited_urls: set,
    retrieval_config: RetrievalConfig,
) -> List[Tuple[str, float, str, Optional[str]]]:
    url_map: Dict[str, Tuple[float, str, Optional[str]]] = {}

    for angle in angles:
        query           = angle.get("query", "")
        discovery_query = angle.get("discovery_query", query)
        if not query:
            continue
        for hit in index.query(query, top_k=retrieval_config.max_per_dimension):
            score = hit.score if hasattr(hit, "score") else 0.0
            if score < retrieval_config.score_threshold or hit.url in visited_urls:
                continue
            snippet = getattr(hit, "snippet", None)
            if hit.url not in url_map or score > url_map[hit.url][0]:
                url_map[hit.url] = (score, discovery_query, snippet)

    return [(url, score, dq, snip) for url, (score, dq, snip) in url_map.items()]


# ---------------------------------------------------------------------- #
# Page reading and claim extraction                                       #
# ---------------------------------------------------------------------- #

def read_page(
    content: ParsedContent,
    user_query: str,
    entity_type: str,
    known_entities: Optional[List[str]] = None,
) -> PageManifest:
    known_block = ""
    if known_entities:
        known_block = f"\nKNOWN ENTITIES (use these names if they refer to the same thing):\n{', '.join(known_entities)}\n"

    prompt = f"""
You are reading a web page to identify named {entity_type}s.

PAGE CONTENT:
{content.text}

RESEARCH CONTEXT:
This page was retrieved during research for the query: "{user_query}"
{known_block}
Your job:
1. Identify the named {entity_type}s discussed on this page.
2. Map any surface variants to the KNOWN ENTITIES names where they refer to the same thing.
3. Classify the page type.

Rules:
- Only include entities actually discussed on the page.
- If uncertain whether two names are the same entity, keep them separate.
- If no relevant {entity_type}s are present, return an empty list.

Choose one page_type: "single_entity", "comparison", "listing", "review", "other"

Return ONLY valid JSON:
{{
  "entities": ["Entity 1", "Entity 2"],
  "canonical_map": {{"surface form": "canonical name"}},
  "page_type": "comparison"
}}
"""
    data          = _llm_json(prompt)
    canonical_map = data.get("canonical_map", {}) or {}
    entities      = data.get("entities", []) or []
    local_entities = list(dict.fromkeys(canonical_map.get(e, e) for e in entities))
    return PageManifest(entities=local_entities, canonical_map=canonical_map,
                        page_type=data.get("page_type", "other"))


def extract_claims(
    content: ParsedContent,
    manifest: PageManifest,
    url: str,
    entity_type: str,
    criteria: List[str] = None,
    discovery_query: str = "",
    working_memory: Optional[dict] = None,
) -> List[dict]:
    if not manifest.entities:
        return []

    entity_list    = ", ".join(manifest.entities)
    criteria_block = ""
    if criteria:
        criteria_block = (
            f"\nDIMENSION LABELS — use these when the claim fits, keep labels short (1-3 words):\n"
            f"{', '.join(criteria)}\n"
            f"You may introduce a new label only if the claim genuinely does not fit any above.\n"
        )

    memory_block = ""
    if working_memory and working_memory.get("beliefs"):
        relevant = {e: dims for e, dims in working_memory["beliefs"].items()
                    if e in manifest.entities}
        if relevant:
            memory_block = (
                f"\nCURRENT BELIEFS FOR THESE ENTITIES:\n{json.dumps(relevant, indent=2)}\n\n"
                f"If this page contradicts the above, set \"contradiction\": true "
                f"and add a \"contradiction_note\". Otherwise set \"contradiction\": false.\n"
            )

    prompt = f"""
You are a high-fidelity transcription agent. Record exactly what this page
asserts about the listed {entity_type}s.

PAGE CONTENT:
{content.text}

{entity_type}s on this page: {entity_list}
Page type: {manifest.page_type}
{criteria_block}{memory_block}
TRANSCRIPTION RULES:
1. Record specific assertions that matter when evaluating or comparing these {entity_type}s.
2. Preserve exact language — especially superlatives, quantifiers, and specific feature claims.
   Do NOT paraphrase. If the page says "the only rooftop infinity pool in Bangkok," record that.
3. Record all claims regardless of sentiment.
4. Every claim's subject_entity MUST be one of the listed {entity_type}s.
5. Skip generic filler that asserts nothing specific.

Respond ONLY with valid JSON:
{{"claims": [{{"text": "...", "subject_entity": "...", "dimension": "...", "contradiction": false, "contradiction_note": ""}}]}}
"""
    data         = _llm_json(prompt)
    claims       = data.get("claims") or []
    valid_entities = set(manifest.entities)
    claims = [
        c for c in claims
        if c.get("subject_entity", "") in valid_entities
        or manifest.canonical_map.get(c.get("subject_entity", ""), "") in valid_entities
    ]
    for c in claims:
        c["source_url"]     = url
        c["lineage_query"]  = discovery_query
        raw = c.get("subject_entity", "")
        c["subject_entity"] = manifest.canonical_map.get(raw, raw)
    return claims


# ---------------------------------------------------------------------- #
# Entity normalization                                                    #
# ---------------------------------------------------------------------- #

def normalize_entities_once(all_claims: List[dict]) -> Dict[str, str]:
    entity_list = list(dict.fromkeys(
        c.get("subject_entity", "") for c in all_claims if c.get("subject_entity", "")
    ))
    if len(entity_list) <= 1:
        return {e: e for e in entity_list}

    prompt = f"""
You are deduplicating entity names collected across multiple web pages.

ENTITY NAMES FOUND:
{json.dumps(entity_list, indent=2)}

Map each name to its canonical form. Rules:
- When two names clearly refer to the same entity, map both to the fuller/more official name.
- When uncertain, keep them separate.
- Every name in the input must appear as a key in the output.

Return ONLY valid JSON:
{{"canonical_map": {{"raw name": "canonical name"}}}}
"""
    data    = _llm_json(prompt)
    raw_map = data.get("canonical_map", {})
    return {e: raw_map.get(e, e) for e in entity_list}


def rewrite_claim_entities(all_claims: List[dict], canonical_map: Dict[str, str]) -> None:
    for c in all_claims:
        name = c.get("subject_entity", "")
        if name:
            c["subject_entity"] = canonical_map.get(name, name)


# ---------------------------------------------------------------------- #
# Working memory synthesis                                                #
# ---------------------------------------------------------------------- #

def synthesize_working_memory(
    new_round_claims: List[dict],
    entity_type: str,
    user_query: str,
    prior_working_memory: Optional[dict] = None,
    round_num: int = 0,
) -> dict:
    if not new_round_claims and not prior_working_memory:
        return {"narrative": "No claims collected yet.", "beliefs": {}, "contradictions": [], "round": round_num}

    by_entity_dim: Dict[str, Dict[str, list]] = {}
    for c in new_round_claims:
        e   = c.get("subject_entity", "")
        dim = c.get("dimension", "unknown")
        if not e:
            continue
        by_entity_dim.setdefault(e, {}).setdefault(dim, []).append({
            "text":   c.get("text", "")[:150],
            "source": c.get("source_url", ""),
        })

    compact = {e: {dim: items[:4] for dim, items in dims.items()}
               for e, dims in by_entity_dim.items()}

    if prior_working_memory:
        prior_block = (
            f"PRIOR WORKING MEMORY (treat as source of truth):\n"
            f"Narrative: {prior_working_memory.get('narrative', '')}\n"
            f"Beliefs: {json.dumps(prior_working_memory.get('beliefs', {}), indent=2)}\n"
            f"Existing contradictions: {json.dumps(prior_working_memory.get('contradictions', []), indent=2)}\n\n"
            f"Update the above using the new evidence below. Do not re-process old data."
        )
    else:
        prior_block = "This is the first round. No prior beliefs exist."

    new_block = (f"NEW EVIDENCE THIS ROUND:\n{json.dumps(compact, indent=2)}"
                 if compact else "No new claims this round.")

    prompt = f"""
You are maintaining the Working Memory for a research agent studying {entity_type}s.
USER QUERY: "{user_query}"
{prior_block}
{new_block}

YOUR JOB:
1. For each entity/dimension touched by new evidence, synthesize a single coherent belief.
2. Flag contradictions where new evidence conflicts with prior beliefs.
3. Update the narrative to reflect the current state of knowledge.

A contradiction means claims that CANNOT both be true
(e.g., "has rooftop pool" vs "no rooftop pool", "$200/night" vs "$450/night").

Return ONLY valid JSON:
{{
  "narrative": "2-4 sentence prose summary.",
  "beliefs": {{"Entity Name": {{"dimension": "one-sentence belief"}}}},
  "contradictions": [
    {{"entity": "...", "dimension": "...", "claim_a": "...", "source_a": "...",
      "claim_b": "...", "source_b": "...", "note": "..."}}
  ]
}}
"""
    data = _llm_json(prompt)
    return {
        "narrative":      data.get("narrative", ""),
        "beliefs":        data.get("beliefs", {}),
        "contradictions": data.get("contradictions", []),
        "round":          round_num,
    }


# ---------------------------------------------------------------------- #
# Entity summary helper                                                   #
# ---------------------------------------------------------------------- #

def _summarize_entities(claims: List[dict], top_k: int = 8) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for c in claims:
        entity = c.get("subject_entity", "")
        dim    = c.get("dimension", "").strip().lower()
        if " and " in dim:
            dim = dim.split(" and ")[0].strip()
        c["dimension"] = dim
        src = c.get("source_url", "")
        if not entity:
            continue
        if entity not in summary:
            summary[entity] = {"claim_count": 0, "source_urls": set(), "dimensions": set()}
        summary[entity]["claim_count"] += 1
        if src:
            summary[entity]["source_urls"].add(src)
        if dim:
            summary[entity]["dimensions"].add(dim)

    sorted_items = sorted(
        summary.items(),
        key=lambda kv: (kv[1]["claim_count"], len(kv[1]["source_urls"])),
        reverse=True,
    )[:top_k]

    return {
        entity: {
            "claim_count":  stats["claim_count"],
            "source_count": len(stats["source_urls"]),
            "dimensions":   sorted(stats["dimensions"]),
        }
        for entity, stats in sorted_items
    }


# ---------------------------------------------------------------------- #
# URL processor                                                           #
# ---------------------------------------------------------------------- #

def _process_url(
    url: str,
    score: float,
    discovery_query: str,
    snippet: Optional[str],
    index,
    prov_config: ProvenanceConfig,
    user_query: str,
    entity_type: str,
    round_num: int,
    criteria: List[str] = None,
    known_entities: Optional[List[str]] = None,
    working_memory: Optional[dict] = None,
) -> Tuple[List[dict], Optional[dict], dict, List[str]]:
    relevant_text = index.fetch_chunks(
        url=url,
        query=discovery_query or user_query,
        chunk_size=2000,
        overlap=200,
        top_k=5,
    )

    if not relevant_text:
        return [], None, {
            "url": url, "score": score, "round": round_num,
            "outcome": "skipped_empty_content", "manifest": None, "claim_count": 0,
        }, []

    parsed   = _parse_content(relevant_text)
    manifest = read_page(parsed, user_query, entity_type, known_entities=known_entities)

    prov_entry = None
    if prov_config and prov_config.enabled:
        prov_entry = {"url": url}
        if prov_config.record_snippet:
            prov_entry["search_snippet"] = snippet
        if prov_config.record_parsed_content:
            prov_entry["parsed_content"] = parsed.text
        if prov_config.record_score:
            prov_entry["search_score"] = score
        if prov_config.record_round_lineage:
            prov_entry["discovery_round"] = round_num
            prov_entry["discovery_query"] = discovery_query
        prov_entry["was_sanitized"]      = parsed.was_sanitized
        prov_entry["manifest_entities"]  = manifest.entities
        prov_entry["manifest_page_type"] = manifest.page_type

    if not manifest.entities:
        return [], prov_entry, {
            "url": url, "score": score, "round": round_num,
            "outcome": "skipped_no_entities",
            "manifest": {"entities": [], "page_type": manifest.page_type},
            "claim_count": 0,
        }, []

    claims = extract_claims(parsed, manifest, url, entity_type, criteria,
                             discovery_query, working_memory=working_memory)
    canonicalized = {k: v for k, v in manifest.canonical_map.items() if k != v}
    contradictions_on_page = [c for c in claims if c.get("contradiction")]

    page_trace = {
        "url":                   url,
        "score":                 score,
        "round":                 round_num,
        "outcome":               "extracted",
        "manifest":              {"entities": manifest.entities, "page_type": manifest.page_type,
                                  "canonicalized": canonicalized},
        "claim_count":           len(claims),
        "contradictions_flagged": len(contradictions_on_page),
        "entities_extracted":    list(dict.fromkeys(c.get("subject_entity", "") for c in claims)),
    }
    return claims, prov_entry, page_trace, manifest.entities


# ---------------------------------------------------------------------- #
# Researcher node                                                         #
# ---------------------------------------------------------------------- #

def researcher_node(
    state: dict,
    index_path: str = "corpus/indices/baseline.json",
    max_rounds: int = 3,
    prov_config: ProvenanceConfig = None,
    retrieval_config: RetrievalConfig = None,
    planning_config: PlanningConfig = None,
) -> dict:
    record_transition("Researcher")

    if state.get("skip_research"):
        return state

    if retrieval_config is None:
        retrieval_config = RetrievalConfig()
    if prov_config is None:
        prov_config = ProvenanceConfig(enabled=False)
    if planning_config is None:
        planning_config = PlanningConfig()

    entity_type = state.get("entity_type", "entity")
    user_query  = state["user_query"]

    index = SearchIndex(index_path)
    print(f"[researcher] loaded index: {index_path}", flush=True)

    all_claims:           List[dict] = []
    provenance_entries:   Dict[str, dict] = {}
    visited_urls:         set = set()
    research_trace:       List[dict] = []
    coverage_snapshots:   List[dict] = []
    working_memory:       dict = {}
    working_memory_history: List[dict] = []
    round_num = 0

    initial_plan   = initial_research_plan(user_query, entity_type, planning_config.max_open_questions)
    criteria       = initial_plan.get("criteria", [])
    open_questions = initial_plan.get("open_questions", [])
    search_history: List[dict] = []
    current_queries: List[dict] = initial_plan.get("queries", [])
    last_search_goal = initial_plan.get("next_search_goal", "candidate_discovery")
    last_reason      = initial_plan.get("reason", "Initial research framing.")
    last_new_entities: List[str] = []
    planner_decision: dict = {}

    while round_num < max_rounds:
        round_num += 1
        print(f"[researcher] round {round_num}: queries={len(current_queries)} goal={last_search_goal}", flush=True)

        new_url_batch = _collect_urls(current_queries, index, visited_urls, retrieval_config)
        print(f"[researcher] round {round_num}: {len(new_url_batch)} new URLs", flush=True)

        if not new_url_batch:
            print(f"[researcher] no new URLs in round {round_num} — stopping", flush=True)
            break

        for url, _, _, _ in new_url_batch:
            visited_urls.add(url)

        round_claims:     List[dict] = []
        round_page_trace: List[dict] = []
        round_entities:   List[str]  = []

        for url, score, discovery_query, snippet in new_url_batch:
            known_entities = sorted(set(
                c.get("subject_entity", "") for c in all_claims if c.get("subject_entity", "")
            ))
            claims, prov_entry, page_trace, page_entities = _process_url(
                url=url, score=score, discovery_query=discovery_query, snippet=snippet,
                index=index, prov_config=prov_config, user_query=user_query,
                entity_type=entity_type, round_num=round_num, criteria=criteria,
                known_entities=known_entities, working_memory=working_memory,
            )
            round_claims.extend(claims)
            round_page_trace.append(page_trace)
            round_entities.extend(page_entities)

            if prov_entry and url not in provenance_entries:
                provenance_entries[url] = {k: v for k, v in prov_entry.items() if k != "url"}

        all_claims.extend(round_claims)

        prior_entities   = set(c.get("subject_entity", "") for c in all_claims[:-len(round_claims)] if c.get("subject_entity", ""))
        current_entities = set(c.get("subject_entity", "") for c in all_claims if c.get("subject_entity", ""))
        last_new_entities = sorted(current_entities - prior_entities)

        if prov_config.enabled and prov_config.record_round_lineage:
            counts: Dict[str, int] = {}
            for c in all_claims:
                e = c.get("subject_entity", "")
                if e:
                    counts[e] = counts.get(e, 0) + 1
            coverage_snapshots.append({"round": round_num, "coverage": counts})

        entity_summary  = _summarize_entities(all_claims, top_k=planning_config.summary_top_entities)
        raw_dimensions  = set(c.get("dimension", "") for c in all_claims if c.get("dimension", ""))
        emergent_dims   = sorted(set(d.lower().replace("_", " ") for d in raw_dimensions))
        criteria        = emergent_dims

        prior_wm      = working_memory if working_memory else None
        working_memory = synthesize_working_memory(
            new_round_claims=round_claims,
            entity_type=entity_type,
            user_query=user_query,
            prior_working_memory=prior_wm,
            round_num=round_num,
        )
        working_memory_history.append(working_memory)
        n_contradictions = len(working_memory.get("contradictions", []))
        print(f"[researcher] wm: {len(working_memory.get('beliefs', {}))} entities, "
              f"{n_contradictions} contradiction(s)", flush=True)

        notebook = {
            "entities_found":       entity_summary,
            "identified_criteria":  emergent_dims,
            "contradictions_count": n_contradictions,
        }

        research_trace.append({
            "round":          round_num,
            "search_goal":    last_search_goal,
            "reason":         last_reason,
            "queries":        current_queries,
            "pages":          round_page_trace,
            "new_urls":       len(new_url_batch),
            "new_claims":     len(round_claims),
            "total_claims":   len(all_claims),
            "new_entities":   last_new_entities,
            "entity_summary": entity_summary,
            "open_questions": open_questions,
            "answerable":     planner_decision.get("answerable", False),
            "working_memory": working_memory,
        })
        search_history.append({
            "round":       round_num,
            "goal":        last_search_goal,
            "queries":     [q.get("query") for q in current_queries],
            "new_urls":    len(new_url_batch),
            "new_entities": last_new_entities,
        })

        state_summary = {
            "notebook":                notebook,
            "criteria_taxonomy":       criteria,
            "top_entities_coverage":   entity_summary,
            "new_entities_last_round": last_new_entities,
            "last_search_goal":        last_search_goal,
            "new_info_last_round":     len(last_new_entities) > 0,
        }

        planner_decision = plan_next_step(
            user_query=user_query,
            entity_type=entity_type,
            state_summary=state_summary,
            search_history=search_history,
            max_angles=retrieval_config.max_angles_per_round,
            max_open_questions=planning_config.max_open_questions,
            production_mode=planning_config.production_mode,
            working_memory=working_memory,
        )

        stop_signal = not planner_decision.get("continue", True)
        new_queries  = planner_decision.get("queries", [])

        if planning_config.production_mode and (stop_signal or not new_queries):
            print(f"[researcher] planner stopped at round {round_num}: {planner_decision.get('reason', '')}", flush=True)
            break

        current_queries = new_queries if new_queries else [
            {"query": user_query, "discovery_query": user_query}
        ]
        open_questions = planner_decision.get("open_questions", [])[:planning_config.max_open_questions]
        last_reason    = planner_decision.get("reason", "Continuing...")
        last_search_goal = "attribute_fill"

    # End-of-run entity normalization
    non_empty = [c for c in all_claims if c.get("subject_entity", "").strip()]
    print(f"[researcher] pre-normalization: {len(all_claims)} claims, "
          f"{len(non_empty)} with subject_entity", flush=True)

    canonical_map = normalize_entities_once(all_claims)
    rewrite_claim_entities(all_claims, canonical_map)

    coverage_map: Dict[str, int] = {}
    for c in all_claims:
        e = c.get("subject_entity", "")
        if e:
            coverage_map[e] = coverage_map.get(e, 0) + 1

    # Recompute snapshots on canonicalized names
    final_snapshots: List[dict] = []
    if coverage_snapshots:
        cursor = 0
        for trace_entry in research_trace:
            cursor += trace_entry.get("new_claims", 0)
            running: Dict[str, int] = {}
            for c in all_claims[:cursor]:
                e = c.get("subject_entity", "")
                if e:
                    running[e] = running.get(e, 0) + 1
            final_snapshots.append({"round": trace_entry["round"], "coverage": dict(running)})
    else:
        final_snapshots = coverage_snapshots

    return {
        **state,
        "claims":               all_claims,
        "retrieved_pages":      list(visited_urls),
        "provenance_index":     provenance_entries if prov_config.enabled else {},
        "coverage_map":         coverage_map,
        "coverage_snapshots":   final_snapshots,
        "provenance_enabled":   prov_config.enabled,
        "entity_canonical_map": canonical_map,
        "working_memory":       working_memory,
        "working_memory_history": working_memory_history,
        "researcher_config": {
            "max_rounds":          max_rounds,
            "score_threshold":     retrieval_config.score_threshold,
            "max_per_dimension":   retrieval_config.max_per_dimension,
            "max_angles_per_round": retrieval_config.max_angles_per_round,
            "rounds_completed":    round_num,
            "index_path":          index_path,
            "provenance_mode":     "full" if prov_config.enabled else "baseline",
            "production_mode":     planning_config.production_mode,
        },
        "planner_state": {
            "criteria":       criteria,
            "open_questions": open_questions,
        },
        "rounds_completed": round_num,
        "research_trace":   research_trace,
    }