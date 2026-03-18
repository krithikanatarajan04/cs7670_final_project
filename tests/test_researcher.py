import json
from agents.researcher import researcher_node
from pipeline.cfg import reset_trace

# 1. Prepare a mock state
initial_state = {
    "user_query": "I want a hotel near Lumpini Park with a high rating and easy BTS access."
}

# 2. Reset the CFG trace so the guardrail allows the Researcher to run
reset_trace()

print("--- STARTING RESEARCHER TEST ---")

# 3. Call the node directly
final_state = researcher_node(initial_state)

# 4. Inspect the output
print("\n--- TEST RESULTS ---")
print(f"Sub-Queries Generated: {final_state.get('sub_queries')}")
print(f"Pages Retrieved: {len(final_state.get('retrieved_pages', []))}")
print(f"Total Claims Found: {len(final_state.get('claims', []))}")

# Print the first 2 claims to see the structure
print("\nSample Claims:")
for claim in final_state.get('claims', [])[:2]:
    print(json.dumps(claim, indent=2))