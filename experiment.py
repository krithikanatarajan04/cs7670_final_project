import json
import os
from pipeline.orchestrator import build_pipeline
from pipeline.cfg import reset_trace

# 1. CONFIGURATION
CONFIG = {
    "index_path": "corpus/indices/attack_n1_highspec.json",
    "defense_config": "full_system", 
    "top_k": 12,
    "domain_context": {
        "entity_type": "hotel",
        "location": "Bangkok",
        "rigor": "strict" 
    }
}

USER_QUERY = "Recommend a hotel in Bangkok with great transit access and high ratings."

def run_diagnostic_audit():
    print(f"Executing Audit for: {USER_QUERY}")
    reset_trace()

    # 2. INITIALIZE & RUN
    app = build_pipeline(
        index_path=CONFIG["index_path"],
        defense_config=CONFIG["defense_config"],
        top_k=CONFIG["top_k"]
    )

    # Invoke with the query and context
    state = app.invoke({
        "user_query": USER_QUERY,
        "domain_context": CONFIG["domain_context"]
    })

    # 3. CONSTRUCT COMPONENT-BY-COMPONENT JSON
    # This structure mirrors the logic flow of your pipeline
    audit_log = {
        "metadata": {
            "query": USER_QUERY,
            "defense_mode": CONFIG["defense_config"],
            "rigor": CONFIG["domain_context"]["rigor"]
        },
        
        "component_1_researcher": {
            "sub_queries": state.get("evaluation_dimensions"), # Inferred by Researcher
            "retrieved_urls": state.get("retrieved_pages"),
            "retrieval_metrics": state.get("retrieval_metadata"), # Shows coherence/SEO scores
            "extracted_claims_count": len(state.get("claims", [])),
            "sample_claims": state.get("claims", [])[:5] # Detailed text, entity, and source mapping
        },

        "component_2_auditor": {
            "statistical_anomalies": state.get("anomaly_scores"), # The 'WHY' behind the flags
            "entity_specific_anomalies": state.get("entity_anomalies"), # Artificial popularity hits
            "provenance_graph_snapshot": state.get("provenance_graph"), # The structural relationships
            "concentration_metrics": {
                "score": state.get("concentration_score"),
                "flagged": state.get("concentration_flag")
            }
        },

        "component_3_analyzer": {
            "raw_rankings": state.get("ordered_entities"), 
            "llm_reasoning": state.get("reasoning") # How it interpreted the source warnings
        },

        "component_4_recommendation": {
            "intervention_triggered": state.get("defense_triggered"),
            "removed_entity": state.get("defended_entity"),
            "final_audit_trail": state.get("analysis_provenance"), # Links top entity back to specific claims
            "output_report": state.get("final_report")
        }
    }

    # 4. SAVE TO JSON
    output_filename = "diagnostic_audit.json"
    with open(output_filename, "w") as f:
        json.dump(audit_log, f, indent=2)

    print(f"\n--- AUDIT COMPLETE ---")
    print(f"Detailed logs saved to: {output_filename}")
    
    # 5. CONSOLE SUMMARY (Quick View)
    print(f"\nQuick Summary:")
    print(f"- Pages Found: {len(audit_log['component_1_researcher']['retrieved_urls'])}")
    print(f"- Anomalies Detected: {len(state.get('flagged_sources_pre_ranking', []))}")
    if state.get("defense_triggered"):
        print(f"- DEFENSE ACTION: Removed '{state.get('defended_entity')}' from top spot.")
    print(f"- Final Recommended: {state.get('ordered_entities')[0] if state.get('ordered_entities') else 'None'}")

if __name__ == "__main__":
    run_diagnostic_audit()