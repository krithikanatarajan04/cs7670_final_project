from pipeline.cfg import record_transition

def recommendation_node(state: dict) -> dict:
    """
    Final output agent: Formats the results into a user-facing report.

    If defense_triggered is True, the top-ranked hotel was flagged by
    the Verifier as having suspicious evidence — either a single source
    dominated its claim set (concentration) or two sources were
    anomalously coordinated (coordination). In that case the top hotel
    is removed from the rankings and the report includes an alert
    explaining why, with the flagged source URLs. The next-best hotel
    becomes the recommendation.

    This is the intervention step that converts detection into defense.
    The Verifier detects and records; the RecommendationAgent acts.
    """
    record_transition("RecommendationAgent")

    rankings = state.get("rankings", [])
    reasoning = state.get("reasoning", "No analysis reasoning found.")
    defense_triggered = state.get("defense_triggered", False)
    flagged_sources = state.get("flagged_sources") or []

    report_lines = [
        "==========================================",
        "      BANGKOK HOTEL RECOMMENDATIONS       ",
        "==========================================",
        ""
    ]

    if defense_triggered and rankings:
        # The top-ranked hotel's evidence was flagged as suspicious.
        # Remove it and surface the next-best option instead.
        flagged_hotel = rankings[0]
        clean_rankings = rankings[1:]

        flagged_urls = [f["url"] for f in flagged_sources]
        reasons = list({f["reason"] for f in flagged_sources})

        report_lines.append(
            f"⚠️  ALERT: The top-ranked recommendation ({flagged_hotel}) "
            f"has been withheld."
        )
        report_lines.append(
            f"   Reason: Evidence supporting this recommendation was flagged "
            f"as potentially manipulated ({', '.join(reasons)})."
        )
        report_lines.append(
            f"   Flagged source(s): {', '.join(flagged_urls)}"
        )
        report_lines.append(
            f"   These source(s) contributed a disproportionate or "
            f"anomalously coordinated share of the supporting claims."
        )
        report_lines.append("")
        report_lines.append("   Recommendations based on remaining evidence:")
        report_lines.append("")

        if not clean_rankings:
            report_lines.append(
                "   No alternative recommendations available after "
                "removing flagged evidence."
            )
        else:
            for i, hotel in enumerate(clean_rankings, 1):
                report_lines.append(f"   {i}. {hotel}")

        # Update state rankings to reflect the defended output
        state["rankings"] = clean_rankings
        state["defended_hotel"] = flagged_hotel

    else:
        # No defense triggered — output rankings as normal
        if not rankings:
            report_lines.append(
                "No suitable hotels found based on the research criteria."
            )
        else:
            for i, hotel in enumerate(rankings, 1):
                report_lines.append(f"{i}. {hotel}")

    report_lines.append("\n### RESEARCH SUMMARY & REASONING")
    report_lines.append(reasoning)
    report_lines.append("\n==========================================")

    state["final_report"] = "\n".join(report_lines)

    defended = " (DEFENSE ACTIVE)" if defense_triggered else ""
    print(f"[RecommendationAgent] Final report generated{defended}.")
    return state