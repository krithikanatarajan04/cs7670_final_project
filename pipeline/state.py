from typing import Optional, Any, List
from typing import TypedDict


class PipelineState(TypedDict, total=False):

    # Set at invocation, never modified by agents
    user_query: str
    planner_context: str
    entity_type: str

    # Researcher outputs
    provenance_index: Optional[dict]
    coverage_map: Optional[dict]
    coverage_snapshots: Optional[list]
    provenance_enabled: Optional[bool]
    researcher_config: Optional[dict]
    rounds_completed: Optional[int]
    research_trace: Optional[list]
    entity_canonical_map: Optional[dict]
    working_memory: Optional[dict]
    working_memory_history: Optional[list]
    planner_state: Optional[dict]
    retrieved_pages: List[str]
    claims: List[dict]
    overhead_trace: Optional[dict]

    # Edge-level (CV judge)
    cv_judge_result: Optional[dict]

    # Auditor outputs
    provenance_graph: Optional[dict]
    anomaly_scores: Optional[List[dict]]
    flagged_sources_pre_ranking: Optional[List[dict]]
    coverage_entropy_result: Optional[dict]
    signals_run: Optional[list]
    concentration_flag: bool
    concentration_score: Optional[float]

    # Analyzer outputs
    excluded_sources: Optional[list]
    exclusion_reasons: Optional[dict]
    # exclusion_reasons schema:
    #   { url: { check: "Robust-RMS-Tension", z_score: float,
    #            axes_counted: int, vector_components: {planted, coord, bias} } }
    clean_claims: Optional[List[dict]]
    clean_claim_count: Optional[int]
    process_caveat: Optional[bool]
    suspicious_dimensions: Optional[List[str]]

    # RecommendationAgent outputs
    ordered_entities: List[str]
    reasoning: str
    analysis_provenance: Optional[dict]
    defense_triggered: Optional[bool]
    defended_entity: Optional[str]
    final_report: Optional[str]

    # Dossier outputs (written by RecommendationAgent)
    dossier_data: Optional[dict]
    # Schema: { entity: { dimension: { representative_claim: str | None,
    #                                  corroboration_count: int,
    #                                  source_urls: list[str],
    #                                  gap: bool } } }

    starvation_diff: Optional[dict]
    # Schema: { entity: { dimension: "starved" | "present" } }
    # "starved" means all sources for that (entity, dimension) pair were
    # excluded by the Analyzer. Used for the paper's before/after audit trail.