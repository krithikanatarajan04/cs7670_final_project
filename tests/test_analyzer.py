from agents.analyzer import analyzer_node
from pipeline.cfg import reset_trace, record_transition

# 1. Prepare a mock state with the data you already got from the Researcher
mock_state = {
    "user_query": "I want a hotel near Lumpini Park with a high rating and easy BTS access.",
    "claims": [
        {
            "text": "Hotel Lumpini sits just 180 meters from BTS Sala Daeng on the Silom Line.",
            "hotel": "Hotel Lumpini",
            "criterion": "transit_access"
        },
        {
            "text": "Hotel Lumpini holds a 4.7 out of 5 rating across 2,340 reviews on TripAdvisor.",
            "hotel": "Hotel Lumpini",
            "criterion": "ratings"
        },
        {
            "text": "Hotel Silom is located 220 meters from BTS Chong Nonsi.",
            "hotel": "Hotel Silom",
            "criterion": "transit_access"
        }
        # ... you can add more of your 31 claims here if you want to be thorough
    ]
}

# 2. CFG requires Researcher to have run first
reset_trace()
record_transition("Researcher") 

print("--- STARTING ANALYZER TEST ---")

# 3. Call the Analyzer
final_state = analyzer_node(mock_state)

# 4. Results
print("\n--- ANALYZER RANKINGS ---")
for i, hotel in enumerate(final_state['rankings'], 1):
    print(f"{i}. {hotel}")

print("\n--- REASONING ---")
print(final_state['reasoning'])