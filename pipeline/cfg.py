from lark import Lark, UnexpectedInput
from typing import Callable

CFG_GRAMMAR = r"""
?start: pipeline

pipeline: researcher
        | researcher auditor
        | researcher auditor analyzer
        | researcher auditor analyzer rec_agent

researcher: "Researcher"
auditor:    "Auditor"
analyzer:   "Analyzer"
rec_agent:  "RecommendationAgent"

%import common.WS
%ignore WS
"""

_parser = Lark(CFG_GRAMMAR, parser="lalr", start="start")

# Module-level trace. NOT thread-safe — one pipeline per process.
_trace: list[str] = []


def record_transition(agent_name: str) -> bool:
    candidate = _trace + [agent_name]
    try:
        _parser.parse(" ".join(candidate))
        _trace.append(agent_name)
        return True
    except UnexpectedInput as e:
        raise RuntimeError(
            f"[CFG VIOLATION] Attempted: '{agent_name}' | "
            f"Trace: {' -> '.join(_trace) or 'empty'} | "
            f"Expected: Researcher -> Auditor -> Analyzer -> RecommendationAgent"
        ) from e


def validated_transition(
    agent_name: str,
    state: dict,
    validators: list[Callable[[dict], tuple[bool, str]]],
) -> list[tuple[bool, str]]:
    record_transition(agent_name)
    results = []
    for v in validators:
        passed, msg = v(state)
        results.append((passed, msg))
        if not passed:
            print(f"[CFG] validator failed for '{agent_name}': {msg}")
    return results


def validate_researcher_output(state: dict) -> tuple[bool, str]:
    claims = state.get("claims")
    if not claims:
        return False, "Researcher produced no claims"
    if state.get("provenance_enabled"):
        prov = state.get("provenance_index", {})
        unsanitized = [u for u, e in prov.items() if not e.get("was_sanitized")]
        if unsanitized:
            return False, f"Unsanitized URLs: {unsanitized}"
    return True, f"{len(claims)} claims, provenance clean"


def validate_anomaly_scores_present(state: dict) -> tuple[bool, str]:
    scores = state.get("anomaly_scores")
    if scores is None:
        return False, "anomaly_scores missing — Auditor may not have run"
    return True, f"anomaly_scores: {len(scores)} entries"


def reset_trace():
    _trace.clear()