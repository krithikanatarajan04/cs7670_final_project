from pipeline.cfg import record_transition

def recommendation_node(state: dict) -> dict:
    """
    Final output agent: Formats the results into a user-facing report.
    """
    # Mandatory CFG Guardrail
    record_transition("RecommendationAgent")

    rankings = state.get("rankings", [])
    reasoning = state.get("reasoning", "No analysis reasoning found.")

    # Construct the readable report string
    report_lines = [
        "==========================================",
        "      BANGKOK HOTEL RECOMMENDATIONS       ",
        "==========================================",
        ""
    ]
    
    if not rankings:
        report_lines.append("No suitable hotels found based on the research criteria.")
    else:
        for i, hotel in enumerate(rankings, 1):
            report_lines.append(f"{i}. {hotel}")
    
    report_lines.append("\n### RESEARCH SUMMARY & REASONING")
    report_lines.append(reasoning)
    report_lines.append("\n==========================================")

    # Store the final product in state
    state["final_report"] = "\n".join(report_lines)

    print("[RecommendationAgent] Final report generated and stored in state.")
    return state