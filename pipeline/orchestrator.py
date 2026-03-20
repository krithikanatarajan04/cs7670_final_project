"""
pipeline/orchestrator.py

Builds the LangGraph pipeline. Accepts index_path, defense_config,
and top_k so the experiment runner can control all three independently.

defense_config levels:
    "no_defense"        — CFG enforced only.
    "controlvalve_only" — Same as no_defense, explicit for ablation.
    "observe"           — Full detection, defense_triggered always False.
    "full_system"       — Full detection, defense_triggered reflects detections.

top_k:
    Number of pages the Researcher retrieves in total. Default 5.
    Sweep k=5, k=8, k=10 to show detection generalises across retrieval depth.
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
    defense_config: str = "no_defense",
    top_k: int = 5
):
    """
    Wires agents into the fixed linear pipeline:
        Researcher -> Analyzer -> Verifier -> RecommendationAgent

    Args:
        index_path:     Path to the corpus index JSON file.
        defense_config: One of "no_defense", "controlvalve_only",
                        "observe", "full_system".
        top_k:          Number of pages the Researcher retrieves.
                        Passed through to researcher_node via partial.

    Returns:
        Compiled LangGraph app ready to invoke with {"user_query": "..."}.
    """

    bound_researcher = functools.partial(
        researcher_node,
        index_path=index_path,
        top_k=top_k
    )

    bound_verifier = functools.partial(
        verifier_node,
        defense_config=defense_config
    )

    workflow = StateGraph(PipelineState)

    workflow.add_node("Researcher", bound_researcher)
    workflow.add_node("Analyzer", analyzer_node)
    workflow.add_node("Verifier", bound_verifier)
    workflow.add_node("RecommendationAgent", recommendation_node)

    workflow.add_edge(START, "Researcher")
    workflow.add_edge("Researcher", "Analyzer")
    workflow.add_edge("Analyzer", "Verifier")
    workflow.add_edge("Verifier", "RecommendationAgent")
    workflow.add_edge("RecommendationAgent", END)

    return workflow.compile()