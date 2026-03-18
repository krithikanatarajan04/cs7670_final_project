from pipeline.cfg import record_transition

def verifier_node(state: dict) -> dict:
    """
    Stub Verifier: Currently performs no actual validation.
    """
    # Mandatory CFG Guardrail
    record_transition("Verifier")

    # stub — full concentration check implemented in next phase
    state["concentration_flag"] = False

    print("[Verifier] Stub active: Concentration check skipped.")
    return state