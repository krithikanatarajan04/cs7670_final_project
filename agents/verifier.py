from pipeline.cfg import record_transition

def verifier_node(state: dict) -> dict:
    record_transition("Verifier")
    return {
        **state,
        "concentration_flag":    False,
        "provenance_log":        None,
        "concentration_score":   None,
        "concentration_flagged": None,
        "defense_triggered":     None,
    }