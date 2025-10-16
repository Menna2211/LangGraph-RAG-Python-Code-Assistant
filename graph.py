# graph.py
from langgraph.graph import StateGraph, END
from state import AssistantState
from nodes_langchain import chat_node, router_node, generate_code_node, explain_code_node, route_by_intent

def build_blueprint_graph():
    """Build the exact state machine from blueprint"""
    workflow = StateGraph(AssistantState)
    # Add nodes
    workflow.add_node("chat", chat_node)
    workflow.add_node("router", router_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("explain_code", explain_code_node)
    
    # Set entry point
    workflow.set_entry_point("chat")
    
    # Define edges
    workflow.add_edge("chat", "router")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "generate_code": "generate_code",
            "explain_code": "explain_code",
        }
    )
    
    # After generation/explanation, END the flow 
    workflow.add_edge("generate_code", END)
    workflow.add_edge("explain_code", END)
    
    return workflow.compile()

# Create the graph instance
graph = build_blueprint_graph()
app = graph  # Alias for visualization