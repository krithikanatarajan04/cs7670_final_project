from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# Import your agent nodes
from agents.researcher import researcher_node
from agents.analyzer import analyzer_node
from agents.verifier import verifier_node
from agents.recommendation import recommendation_node

# Define the shared state structure
class PipelineState(TypedDict):
    user_query: str
    sub_queries: List[str]
    retrieved_pages: List[str]
    claims: List[dict]
    rankings: List[str]
    reasoning: str
    concentration_flag: bool
    final_report: str

def build_pipeline():
    """
    Wires the agents together in a sequence:
    Researcher -> Analyzer -> Verifier -> RecommendationAgent
    """
    # 1. Initialize the Graph with our state type
    workflow = StateGraph(PipelineState)

    # 2. Add Nodes 
    # The names here are used for internal routing, but the 'record_transition'
    # calls inside these functions must match the CFG grammar strings.
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Analyzer", analyzer_node)
    workflow.add_node("Verifier", verifier_node)
    workflow.add_node("RecommendationAgent", recommendation_node)

    # 3. Add Edges (The "Intended" Path)
    workflow.add_edge(START, "Researcher")
    workflow.add_edge("Researcher", "Analyzer")
    workflow.add_edge("Analyzer", "Verifier")
    workflow.add_edge("Verifier", "RecommendationAgent")
    workflow.add_edge("RecommendationAgent", END)

    # 4. Compile the graph
    return workflow.compile()