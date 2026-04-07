"""
pipeline/state.py

Defines PipelineState — the single data structure passed between all agents
in the research pipeline. Fields are populated exactly once by the agent
responsible for each section. No agent modifies another agent's fields.
"""

from typing import Optional, Any
from typing import TypedDict, List


class PipelineState(TypedDict, total=False):

    # ------------------------------------------------------------------ #
    # Planning-time fields — set at invocation, never modified by agents  #
    # ------------------------------------------------------------------ #

    user_query: str
    """The original user query. Fixed at invocation time."""

    # ------------------------------------------------------------------ #
    # Researcher fields — populated by researcher.py                      #
    # ------------------------------------------------------------------ #

    provenance_index: Optional[dict]
    """
    URL-keyed dict of provenance entries collected by the Researcher.
    Each key is a source URL. Each value is a dict containing whichever
    of the following fields ProvenanceConfig had enabled: search_snippet,
    parsed_content, retrieval_score, discovery_round, discovery_query,
    was_sanitized. Replaces retrieval_metadata entirely.
    """

    coverage_map: Optional[dict]
    """
    Final merged dimension coverage dict {dimension_name: source_count}.
    Reflects how many unique sources were retrieved per research dimension
    across all rounds. Written by Researcher, read by Auditor Signal 4.
    """

    coverage_snapshots: Optional[list]
    """
    Ordered list of per-round coverage states. Each entry is
    {round: int, coverage: {dimension: count}}. Captures trajectory of
    coverage across rounds, not just the final state. Written by Researcher
    when record_round_lineage=True. Read by Auditor Signal 4 to compute
    coverage entropy.
    """

    provenance_enabled: Optional[bool]
    """
    True if the Researcher ran with a ProvenanceConfig that had
    enabled=True. If False, the Auditor skips all signals that depend on
    provenance_index fields.
    """

    researcher_config: Optional[dict]
    """
    Self-describing record of Researcher parameters for this run.
    Contains max_rounds, top_k, rounds_completed, index_path,
    provenance_mode. Written once by Researcher, never modified. Used for
    experiment self-description and result reproducibility.
    """

    rounds_completed: Optional[int]
    """
    Number of research rounds that actually ran, which may be less than
    max_rounds if convergence check triggered early exit. Mirrors the value
    in researcher_config for convenient access.
    """

    retrieved_pages: List[str]
    """
    List of URLs retrieved by the Researcher, in MMR selection order.
    """

    claims: List[dict]
    """
    Structured claims extracted from retrieved pages.
    Each dict has exactly these keys:
        text            (str)  — the factual claim as a sentence
        subject_entity  (str)  — which entity the claim is about (e.g., a hotel, framework, etc)
        dimension       (str)  — which dimension it supports (from evaluation_dimensions)
        source_url      (str)  — the URL the claim was extracted from
    """

    overhead_trace: Optional[dict]
    """
    Self-describing timing record for every agent and phase. Initialized by
    Researcher, extended by each subsequent agent. Structure:
        {
          "researcher": {"total_s": float, "rounds": [{"round": N, "elapsed_s": float}]},
          "auditor": {
            "graph_construction_s": float,
            "phase1_focus_s": float,
            "phase1_entropy_s": float,
            "embedding_call_s": float,
            "phase2_snippet_s": float,
            "phase2_coordination_s": float,
            "phase2_corroboration_s": float,
            "aggregation_s": float,
            "total_s": float
          },
          "analyzer": {"check3_s": float, "ranking_s": float, "exclusion_s": float,
                       "llm_s": float, "total_s": float},
          "recommendation": {"total_s": float}
        }
    """

    # ------------------------------------------------------------------ #
    # Auditor fields — populated by auditor.py                            #
    # ------------------------------------------------------------------ #

    provenance_graph: Optional[dict]
    """
    The ProvenanceGraph serialised to dict.
    """

    anomaly_scores: Optional[List[dict]]
    """
    Per-source anomaly results computed by the Auditor. One entry per
    source URL. Each entry contains: url (str), signal_scores (dict mapping
    signal name to its raw numeric score including z-scores and joint scores),
    coordination_flagged (bool), max_pair_similarity (float). No anomaly_flags
    or flagged fields — thresholding decisions live in the analyzer.
    Produced by aggregation layer after all active signals run.
    """

    flagged_sources_pre_ranking: Optional[List[dict]]
    """
    Kept for backward compatibility. Written as empty list by Auditor.
    Backfilled by Analyzer with excluded source entries after exclusion pass.
    """

    coverage_entropy_result: Optional[dict]
    """
    Output of Signal 4 (coverage entropy). Not URL-keyed — covers the
    research process as a whole. Contains: round_entropies (list of
    float), entropy_deltas (list of float between consecutive rounds),
    fast_convergence_flagged (bool), suspicious_dimensions (list of str —
    dimensions that appeared covered too quickly). Empty dict if Signal 4
    did not run or coverage_snapshots was unavailable.
    """

    signals_run: Optional[list]
    """
    List of signal names that actually executed in this Auditor run. Used
    for experiment self-description and ablation result attribution.
    Example: ["focus", "snippet_divergence"]. Empty list if defense was
    off or no signals were configured.
    """

    concentration_flag: bool
    """
    True if the top-ranked entity's evidence derives disproportionately
    from a single source URL.
    """

    concentration_score: Optional[float]
    """
    Fraction of top-ranked entity's claims derived from the most dominant source.
    """

    # ------------------------------------------------------------------ #
    # Analyzer fields — populated by analyzer.py                          #
    # ------------------------------------------------------------------ #

    ordered_entities: List[str]
    """
    Subject entities in ranked order produced by LLM ranking over clean
    evidence only. Excludes entities whose supporting claims were entirely
    removed by source exclusion.
    """

    reasoning: str
    """
    LLM reasoning string over clean evidence. May include Check 3 caveat
    appended by analyzer if process_caveat is True.
    """

    excluded_sources: Optional[list]
    """
    List of URLs excluded by the analyzer's check decisions. Written by
    analyzer, used by recommendation agent for audit trail.
    """

    clean_claim_count: Optional[int]
    """
    Count of claims that survived exclusion. Written by analyzer; useful
    for experiment tracking.
    """

    process_caveat: Optional[bool]
    """
    True if Check 3 flagged fast convergence. Written by analyzer; read
    by recommendation agent for report formatting.
    """

    # ------------------------------------------------------------------ #
    # RecommendationAgent fields — populated by recommendation.py         #
    # ------------------------------------------------------------------ #

    analysis_provenance: Optional[dict]
    """
    Post-ranking audit trail recorded by the RecommendationAgent.
    Structure:
        top_entity              (str)         — the recommended entity
        supporting_claims       (list[dict])  — claims supporting the top entity
        flagged_source_urls     (list[str])   — source URLs that were flagged
        ordered_entities        (list[str])   — full ranking list
        reasoning               (str)         — Analyzer reasoning
        excluded_source_urls    (list[str])   — URLs excluded by analyzer checks
        clean_claim_count       (int)         — claims that survived exclusion
        process_caveat          (bool)        — whether Check 3 fired
        signals_run             (list[str])   — signals that ran in the auditor
    """

    defense_triggered: Optional[bool]
    """
    True if the recommendation agent's LLM ranking produced a top entity
    that had zero claims after source exclusion — meaning the defense caused
    a fallback to the next entity. False if the top entity survived exclusion
    with claims intact.
    """

    defended_entity: Optional[str]
    """
    The entity that was the top-ranked result from clean evidence but had
    all its claims excluded. Only populated if defense_triggered is True.
    """

    final_report: Optional[str]
    """
    The final human-readable recommendation report.
    """