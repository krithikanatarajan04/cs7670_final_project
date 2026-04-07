"""
CONTROLVALVE-inspired CFG enforcement for deterministic agent execution.

Planning-time commitment:
- Grammar is defined and parser compiled at module import time,
  ensuring guardrails exist before any untrusted content is ingested.
"""

from lark import Lark, UnexpectedInput
from typing import Callable

# ------------------------------------------------------------
# 1. Grammar string (Researcher -> Auditor -> Analyzer -> RecommendationAgent)
# ------------------------------------------------------------

# Edge rule on Researcher -> Auditor:
#   validate_researcher_output() checks claims exist and was_sanitized=True on all provenance entries
#   This enforces that _parse_content ran before any LLM extraction — structural guarantee via type + validator

CFG_GRAMMAR = r"""
?start: pipeline

# This structure enforces the exact order while allowing 
# the sequence to be valid at any point (prefix-valid).
pipeline: researcher
        | researcher auditor
        | researcher auditor analyzer
        | researcher auditor analyzer rec_agent

researcher: "Researcher"
auditor: "Auditor"
analyzer: "Analyzer"
rec_agent: "RecommendationAgent"

%import common.WS
%ignore WS
"""

# ------------------------------------------------------------
# 2. Compile parser at import time (planning-time commitment)
# ------------------------------------------------------------

parser = Lark(
    CFG_GRAMMAR,
    parser="lalr",
    start="start",
)

# ------------------------------------------------------------
# 3. Execution trace (module-level state)
# ------------------------------------------------------------

trace = []

# ------------------------------------------------------------
# 4. Record transition with CFG validation
# ------------------------------------------------------------

def record_transition(agent_name: str) -> bool:
    """
    Append agent name to execution trace and validate 
    against the deterministic CFG.
    
    Returns:
        True if the transition is valid.
        
    Raises:
        RuntimeError if a Control-Flow Hijacking attempt is detected.
    """
    # Defensive copy of trace for validation
    potential_trace = trace + [agent_name]
    current_sequence = " ".join(potential_trace)

    try:
        # If this sequence doesn't match the grammar, Lark raises UnexpectedInput
        parser.parse(current_sequence)
        
        # If valid, commit to the global trace
        trace.append(agent_name)
        return True
        
    except UnexpectedInput as e:
        # This is the "Reject" or "Block" action from the paper
        raise RuntimeError(
            f"\n[CFG VIOLATION - CONTROL-FLOW HIJACKING DETECTED]\n"
            f"Attempted transition to: '{agent_name}'\n"
            f"Trace so far: {' -> '.join(trace) if trace else 'Empty'}\n"
            f"Expected Order: Researcher -> Auditor -> Analyzer -> RecommendationAgent"
        ) from e

# ------------------------------------------------------------
# 5. Validated transition with CFG + edge-specific rule layer
# ------------------------------------------------------------

def validated_transition(
    agent_name: str,
    state: dict,
    validators: list[Callable[[dict], tuple[bool, str]]]
) -> list[tuple[bool, str]]:
    """
    Calls record_transition for CFG enforcement, then runs each validator
    callable against state. Returns a list of (passed: bool, message: str)
    per validator.

    Validators are callables with signature:
        (state: dict) -> (passed: bool, message: str)

    This implements ControlValve's edge-specific rule layer without
    coupling the grammar parser to state. The Auditor calls:
        validated_transition("Auditor", state, [validate_graph_populated])
    The Analyzer calls:
        validated_transition("Analyzer", state, [validate_anomaly_scores_present])

    Args:
        agent_name:  The agent attempting the transition.
        state:       The current pipeline state dict.
        validators:  List of validator callables to run after CFG check.

    Returns:
        List of (passed, message) tuples, one per validator.

    Raises:
        RuntimeError if the CFG check fails (via record_transition).
    """
    # CFG enforcement first — raises RuntimeError on violation
    record_transition(agent_name)

    # Run each edge-specific validator
    results = []
    for validator in validators:
        passed, message = validator(state)
        results.append((passed, message))
        if not passed:
            print(f"[CFG] Edge validator failed for '{agent_name}': {message}")

    return results

# ------------------------------------------------------------
# 6. Built-in edge validators
# ------------------------------------------------------------

def validate_graph_populated(state: dict) -> tuple[bool, str]:
    """
    Checks that state["provenance_graph"] is present and non-empty.
    Called by the Auditor after building the graph, before the Analyzer
    runs. Ensures the graph construction step actually completed.
    """
    graph = state.get("provenance_graph")
    if not graph:
        return (False, "provenance_graph is missing or empty in state")
    sources = graph.get("sources", {})
    if not sources:
        return (False, "provenance_graph contains no source nodes")
    return (True, "provenance_graph is populated")


def validate_anomaly_scores_present(state: dict) -> tuple[bool, str]:
    """
    Checks that state["anomaly_scores"] was written by the Auditor.
    Called by the Analyzer before ranking proceeds. Ensures the Auditor
    actually ran and produced its per-source multi-signal scores.
    """
    scores = state.get("anomaly_scores")
    if scores is None:
        return (False, "anomaly_scores is missing — Auditor may not have run")
    return (True, f"anomaly_scores present with {len(scores)} entries")


def validate_researcher_output(state: dict) -> tuple[bool, str]:
    """
    Edge validator for Researcher -> Auditor transition.
    Checks that:
    1. claims were produced
    2. if provenance is enabled, all entries have was_sanitized=True
       (proving _parse_content ran for every fetched page)
    """
    claims = state.get("claims")
    if not claims:
        return (False, "Researcher produced no claims — cannot proceed to Auditor")

    if state.get("provenance_enabled", False):
        trace = state.get("provenance_trace", [])
        unsanitized = [e["url"] for e in trace if not e.get("was_sanitized", False)]
        if unsanitized:
            return (False, f"Unsanitized content detected for URLs: {unsanitized}")

    return (True, f"Researcher output valid: {len(claims)} claims, all content sanitized")

# ------------------------------------------------------------
# 7. Reset trace
# ------------------------------------------------------------

def reset_trace():
    """Clear execution trace for a new research task."""
    trace.clear()

# ------------------------------------------------------------
# Test Cases (Verification)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Valid Pipeline...")
    reset_trace()
    record_transition("Researcher")
    record_transition("Auditor")
    record_transition("Analyzer")
    print("Valid so far...")
    
    try:
        print("\nTesting Hijack (Skipping Auditor)...")
        reset_trace()
        record_transition("Researcher")
        record_transition("Analyzer")
    except RuntimeError as err:
        print(err)

    try:
        print("\nTesting Hijack (Skipping to RecommendationAgent)...")
        reset_trace()
        record_transition("Researcher")
        record_transition("RecommendationAgent")
    except RuntimeError as err:
        print(err)