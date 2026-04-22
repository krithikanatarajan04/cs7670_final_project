import functools
from langgraph.graph import StateGraph, START, END

from pipeline.state import PipelineState
from pipeline.cfg import (
    validate_researcher_output,
    validate_anomaly_scores_present,
    validated_transition,
)
from agents.researcher import researcher_node, RetrievalConfig
from agents.auditor import auditor_node
from agents.analyzer import analyzer_node
from agents.recommendation import recommendation_node


def build_pipeline(scenario, defense, pipeline, planning=None):
    researcher_to_auditor_validators = [validate_researcher_output]
    if defense.cv_judge:
        from pipeline.cv_baseline import cv_llm_judge_validator
        researcher_to_auditor_validators.append(cv_llm_judge_validator)

    researcher_kwargs = dict(
        index_path=scenario.corpus_path,
        max_rounds=pipeline.max_rounds,
        prov_config=defense.provenance,
        retrieval_config=RetrievalConfig(
            score_threshold=pipeline.score_threshold,
            max_per_dimension=pipeline.max_per_dimension,
            max_angles_per_round=getattr(pipeline, "max_angles_per_round", 5),
        ),
    )
    if planning is not None:
        researcher_kwargs["planning_config"] = planning

    signals_active = defense.signals.fact1 or defense.signals.fact2 or defense.signals.coverage_entropy

    bound_researcher    = functools.partial(researcher_node, **researcher_kwargs)
    bound_auditor       = functools.partial(
        auditor_node,
        signal_config=defense.signals,
        signals_active=signals_active,
    )
    bound_analyzer      = functools.partial(
        analyzer_node,
        analyzer_exclusion=defense.analyzer_exclusion,
    )
    bound_recommendation = functools.partial(
        recommendation_node,
        controller_intervention=defense.controller_intervention,
    )

    def auditor_with_edge_check(state: dict) -> dict:
        validated_transition("Auditor", state, researcher_to_auditor_validators)
        return bound_auditor(state)

    def analyzer_with_edge_check(state: dict) -> dict:
        validated_transition("Analyzer", state, [validate_anomaly_scores_present])
        return bound_analyzer(state)

    workflow = StateGraph(PipelineState)
    workflow.add_node("Researcher",           bound_researcher)
    workflow.add_node("Auditor",              auditor_with_edge_check)
    workflow.add_node("Analyzer",             analyzer_with_edge_check)
    workflow.add_node("RecommendationAgent",  bound_recommendation)

    # Route START based on whether claims are preloaded (skip_research path)
    workflow.add_conditional_edges(
        START,
        lambda state: "Auditor" if state.get("skip_research") else "Researcher",
        {"Researcher": "Researcher", "Auditor": "Auditor"},
    )
    workflow.add_edge("Researcher",          "Auditor")
    workflow.add_edge("Auditor",             "Analyzer")
    workflow.add_edge("Analyzer",            "RecommendationAgent")
    workflow.add_edge("RecommendationAgent", END)

    return workflow.compile()