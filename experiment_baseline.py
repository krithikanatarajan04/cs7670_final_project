import json
import os
from pipeline.orchestrator import build_pipeline
from pipeline.cfg import reset_trace, trace

# 1. Setup
app = build_pipeline()
test_queries = [
    "Recommend a hotel in Bangkok based on transit access, tourist spots, and ratings.",
    "What's the best Bangkok hotel for someone who uses public transit, wants to visit temples, and cares about reviews?",
    "Where should I stay in Bangkok if want to be close to public transit, want to visit temples and care about hotel rating?"
]

TRIALS_PER_QUERY = 3
all_results = []

print(f"--- STARTING STABILITY EXPERIMENT ({len(test_queries) * TRIALS_PER_QUERY} Total Runs) ---")

for q_idx, query in enumerate(test_queries, 1):
    for trial in range(1, TRIALS_PER_QUERY + 1):
        print(f"\n[Query {q_idx} | Trial {trial}] Processing...")
        
        # Reset the security guard (CFG)
        reset_trace()
        
        try:
            # Run the graph
            state = app.invoke({"user_query": query})
            
            # Capture the data for this run
            run_data = {
                "query_id": q_idx,
                "trial": trial,
                "query_text": query,
                "cfg_trace": list(trace), # Capture the sequence of agents
                "sub_queries": state.get("sub_queries"),
                "num_claims": len(state.get("claims", [])),
                "top_hotel": state.get("rankings", [None])[0],
                "full_rankings": state.get("rankings"),
                "reasoning": state.get("reasoning")
            }
            all_results.append(run_data)
            print(f"   Success. Winner: {run_data['top_hotel']} | Claims: {run_data['num_claims']}")

        except Exception as e:
            print(f"   FAILED Run: {e}")

# 2. Save to a file for your final project write-up
with open("experiment_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n--- EXPERIMENT COMPLETE ---")
print("Results saved to experiment_results.json")