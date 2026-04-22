from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from pipeline.cfg import reset_trace, record_transition
from agents.auditor import SignalConfig
from agents.researcher import ProvenanceConfig, PlanningConfig


@dataclass
class ScenarioConfig:
    user_query: str
    entity_type: str
    planner_context: str
    corpus_path: str = "corpus/indices/baseline.json"
    ground_truth: Optional[str] = None


@dataclass
class DefenseConfig:
    provenance: ProvenanceConfig = field(
        default_factory=lambda: ProvenanceConfig(enabled=False)
    )
    signals: SignalConfig = field(default_factory=SignalConfig)
    analyzer_exclusion: bool = False
    controller_intervention: bool = False
    cv_judge: bool = False
    label: str = "custom"


@dataclass
class PipelineConfig:
    max_rounds: int = 3
    score_threshold: float = 0.65
    max_per_dimension: int = 2
    max_angles_per_round: int = 2


@dataclass
class ExperimentRun:
    scenario: ScenarioConfig
    defense: DefenseConfig
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    planning: Optional[PlanningConfig] = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Populated after run
    final_state: Optional[dict] = None
    overhead: Optional[dict] = None
    top_entity: Optional[str] = None
    defense_triggered: Optional[bool] = None
    cv_judge_result: Optional[dict] = None
    correct: Optional[bool] = None
    elapsed_s: Optional[float] = None


# ------------------------------------------------------------------ #
# Named condition factories                                           #
# ------------------------------------------------------------------ #

def no_defense() -> DefenseConfig:
    return DefenseConfig(label="no_defense")


def cv_baseline() -> DefenseConfig:
    return DefenseConfig(
        label="cv_baseline",
        provenance=ProvenanceConfig(enabled=True),
        cv_judge=True,
    )


def fact1_only(grubbs_alpha: float = 0.05) -> DefenseConfig:
    return DefenseConfig(
        label="fact1_only",
        provenance=ProvenanceConfig(enabled=True),
        signals=SignalConfig(fact1=True, grubbs_alpha=grubbs_alpha),
        analyzer_exclusion=True,
    )


def fact2_only(
    grubbs_alpha: float = 0.05,
    val_asym_lambda: float = 0.5,
    corr_conc_min_shared: int = 2,
    embedding_dim: int = 768,
) -> DefenseConfig:
    return DefenseConfig(
        label="fact2_only",
        provenance=ProvenanceConfig(enabled=True),
        signals=SignalConfig(
            fact2=True,
            grubbs_alpha=grubbs_alpha,
            val_asym_lambda=val_asym_lambda,
            corr_conc_min_shared=corr_conc_min_shared,
            embedding_dim=embedding_dim,
        ),
        analyzer_exclusion=True,
    )


def full_system(
    grubbs_alpha: float = 0.05,
    val_asym_lambda: float = 0.5,
    corr_conc_min_shared: int = 2,
    embedding_dim: int = 768,
    coverage_entropy: bool = True,
    controller_intervention: bool = True,
) -> DefenseConfig:
    return DefenseConfig(
        label="full_system",
        provenance=ProvenanceConfig(enabled=True),
        signals=SignalConfig(
            fact1=True,
            fact2=True,
            coverage_entropy=coverage_entropy,
            grubbs_alpha=grubbs_alpha,
            val_asym_lambda=val_asym_lambda,
            corr_conc_min_shared=corr_conc_min_shared,
            embedding_dim=embedding_dim,
        ),
        analyzer_exclusion=True,
        controller_intervention=controller_intervention,
    )


DEFAULT_PIPELINE = PipelineConfig()
TIGHT_PIPELINE = PipelineConfig(max_rounds=3, score_threshold=0.5,
                                max_per_dimension=2, max_angles_per_round=2)


# ------------------------------------------------------------------ #
# Run helpers                                                         #
# ------------------------------------------------------------------ #

def run_experiment(
    run: ExperimentRun,
    preloaded_claims: Optional[list] = None,
) -> ExperimentRun:
    from pipeline.orchestrator import build_pipeline

    reset_trace()
    if preloaded_claims is not None:
        record_transition("Researcher")

    app = build_pipeline(
        scenario=run.scenario,
        defense=run.defense,
        pipeline=run.pipeline,
        planning=run.planning,
    )

    initial_state: dict = {
        "user_query":         run.scenario.user_query,
        "planner_context":    run.scenario.planner_context,
        "entity_type":        run.scenario.entity_type,
        "analyzer_exclusion": run.defense.analyzer_exclusion,
    }

    if preloaded_claims is not None:
        prov_index = {}
        for c in preloaded_claims:
            url = c.get("source_url", "")
            if url and url not in prov_index:
                prov_index[url] = {
                    "was_sanitized":    True,
                    "discovery_round":  c.get("discovery_round", 0),
                    "discovery_query":  c.get("lineage_query", ""),
                }
        initial_state.update({
            "provenance_index":  prov_index,
            "claims":            preloaded_claims,
            "skip_research":     True,
            "provenance_enabled": True,
        })

    wall_start = time.perf_counter()
    final_state = app.invoke(initial_state)
    run.elapsed_s = time.perf_counter() - wall_start

    run.final_state       = final_state
    run.overhead          = final_state.get("overhead_trace", {})
    run.top_entity        = (final_state.get("ordered_entities") or [None])[0]
    run.defense_triggered = final_state.get("defense_triggered", False)
    run.cv_judge_result   = final_state.get("cv_judge_result")

    if run.scenario.ground_truth is not None:
        run.correct = (run.top_entity == run.scenario.ground_truth)

    return run


def run_sweep(
    scenario: ScenarioConfig,
    conditions: list[DefenseConfig],
    pipeline: PipelineConfig = DEFAULT_PIPELINE,
    planning: Optional[PlanningConfig] = None,
    preloaded_claims: Optional[list] = None,
) -> list[ExperimentRun]:
    runs = []
    for defense in conditions:
        run = ExperimentRun(scenario=scenario, defense=defense,
                            pipeline=pipeline, planning=planning)
        completed = run_experiment(run, preloaded_claims=preloaded_claims)
        runs.append(completed)

        judge_note = ""
        if defense.cv_judge and completed.cv_judge_result:
            decision = completed.cv_judge_result.get("decision", "?")
            reason   = completed.cv_judge_result.get("overall_reason", "")[:60]
            judge_note = f" | cv_judge={decision} ({reason})"

        print(
            f"[{defense.label}] top={completed.top_entity} "
            f"defense_triggered={completed.defense_triggered} "
            f"elapsed={completed.elapsed_s:.2f}s"
            f"{judge_note}"
        )
    return runs