from pipeline.orchestrator import build_pipeline
from pipeline.cfg import reset_trace

# 1. Build the graph
app = build_pipeline()

# 2. Define the input
inputs = {
    "user_query": "Recommend a hotel in Bangkok based on transit access, tourist spots, and ratings."
}

# 3. Reset CFG for a fresh run
reset_trace()

print("--- EXECUTING SECURE PIPELINE ---")
# 4. Run the graph
final_state = app.invoke(inputs)

# 5. Show the final result
print(final_state["final_report"])