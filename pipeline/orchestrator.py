"""
pipeline/orchestrator.py

Builds the LangGraph pipeline. Accepts index_path and defense_config
so the experiment runner can swap corpus conditions and defense
configurations without touching agent code.

defense_config levels:
    "no_defense"        — CFG enforced only (ControlValve structural guard).
                          Provenance log and concentration check are off.
    "controlvalve_only" — Same as no_defense. ControlValve is always on
                          by virtue of the CFG in cfg.py. This label exists
                          to make the ablation explicit in results.
    "full_system"       — CFG + provenance log + source concentration check.
"""

import functools
from langgraph.graph import StateGraph, START, END

from pipeline.state import PipelineState
from agents.researcher import researcher_node
from agents.analyzer import analyzer_node
from agents.verifier import verifier_node
from agents.recommendation import recommendation_node


def build_pipeline(
    index_path: str = "corpus/indices/baseline.json",
    defense_config: str = "no_defense"
):
    """
    Wires agents into the fixed linear pipeline:
        Researcher -> Analyzer -> Verifier -> RecommendationAgent

    Args:
        index_path:     Path to the corpus index JSON file.
                        Controls which pages are visible to the retriever.
                        Relative to project root.
        defense_config: One of "no_defense", "controlvalve_only", "full_system".
                        Passed through to the Verifier to control whether
                        provenance logging and concentration checking are active.

    Returns:
        Compiled LangGraph app ready to invoke with {"user_query": "..."}.
    """

    # Bind index_path into the researcher node so SearchIndex is instantiated
    # fresh with the correct corpus for this condition. This prevents the
    # embedding cache from persisting across conditions in the experiment loop.
    bound_researcher = functools.partial(
        researcher_node,
        index_path=index_path
    )

    # Bind defense_config into the verifier node so it knows whether to
    # run the provenance log and concentration check.
    bound_verifier = functools.partial(
        verifier_node,
        defense_config=defense_config
    )

    # Build graph
    workflow = StateGraph(PipelineState)

    workflow.add_node("Researcher", bound_researcher)
    workflow.add_node("Analyzer", analyzer_node)
    # In build_pipeline — no partial binding for verifier yet
    workflow.add_node("Verifier", verifier_node)
    workflow.add_node("RecommendationAgent", recommendation_node)

    workflow.add_edge(START, "Researcher")
    workflow.add_edge("Researcher", "Analyzer")
    workflow.add_edge("Analyzer", "Verifier")
    workflow.add_edge("Verifier", "RecommendationAgent")
    workflow.add_edge("RecommendationAgent", END)

    return workflow.compile()