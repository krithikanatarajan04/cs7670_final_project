"""
CONTROLVALVE-inspired CFG enforcement for deterministic agent execution.

Planning-time commitment:
- Grammar is defined and parser compiled at module import time,
  ensuring guardrails exist before any untrusted content is ingested.
"""

from lark import Lark, UnexpectedInput

# ------------------------------------------------------------
# 1. Grammar string (Researcher -> Analyzer -> Verifier -> RecAgent)
# ------------------------------------------------------------

CFG_GRAMMAR = r"""
?start: pipeline

# This structure enforces the exact order while allowing 
# the sequence to be valid at any point (prefix-valid).
pipeline: researcher
        | researcher analyzer
        | researcher analyzer verifier
        | researcher analyzer verifier rec_agent

researcher: "Researcher"
analyzer: "Analyzer"
verifier: "Verifier"
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
            f"Expected Order: Researcher -> Analyzer -> Verifier -> RecommendationAgent"
        ) from e

# ------------------------------------------------------------
# 5. Reset trace
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
    record_transition("Analyzer")
    print("Valid so far...")
    
    try:
        print("\nTesting Hijack (Skipping Verifier)...")
        record_transition("RecommendationAgent")
    except RuntimeError as err:
        print(err)