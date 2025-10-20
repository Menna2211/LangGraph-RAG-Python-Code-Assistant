from state import AssistantState
from rag_langchain import code_rag_chain, explain_rag_chain, retriever
from langchain_core.messages import HumanMessage, AIMessage

def chat_node(state: AssistantState) -> AssistantState:
    """Process user input"""
    print("🔄 [chat] Processing input...")
    
    if not state["messages"] or not any(isinstance(msg, HumanMessage) and msg.content == state["user_input"] for msg in state["messages"]):
        state["messages"].append(HumanMessage(content=state["user_input"]))
    
    return state

def router_node(state: AssistantState) -> AssistantState:
    """Classify user intent"""
    print("🔄 [router] Classifying intent...")
    
    user_input = state["user_input"].lower()
    
    generate_keywords = {"generate", "create", "write", "make", "build", "code", "function", "implement"}
    explain_keywords = {"explain", "describe", "how", "what", "why", "works", "meaning", "understand"}
    
    generate_matches = len([k for k in generate_keywords if k in user_input])
    explain_matches = len([k for k in explain_keywords if k in user_input])
    
    if generate_matches > explain_matches:
        state["intent"] = "generate_code"
    elif explain_matches > generate_matches:
        state["intent"] = "explain_code"
    else:
        if any(q in user_input for q in ["how", "what", "why", "?"]):
            state["intent"] = "explain_code"
        else:
            state["intent"] = "generate_code"
    
    print(f"✅ [router] Intent: {state['intent']}")
    return state

def generate_code_node(state: AssistantState) -> AssistantState:
    """Generate code with LangChain RAG"""
    print("🔄 [generate_code] Generating code with RAG...")
    
    try:
        # Use code-specific RAG chain
        response = code_rag_chain.invoke(state["user_input"])
        
        # Store retrieved context info
        docs = retriever.invoke(state["user_input"])
        state["retrieved_context"] = [
            {
                "content": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        
        state["llm_response"] = response
        state["messages"].append(AIMessage(content=response))
        print("✅ [generate_code] Code generated with LangChain RAG")
        
    except Exception as e:
        error_msg = f"Error generating code: {str(e)}"
        state["llm_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
        print(f"❌ [generate_code] {error_msg}")
    
    return state

def explain_code_node(state: AssistantState) -> AssistantState:
    """Explain code with LangChain RAG"""
    print("🔄 [explain_code] Generating explanation with RAG...")
    
    try:
        # Use explanation-specific RAG chain
        response = explain_rag_chain.invoke(state["user_input"])
        
        # Store retrieved context info
        docs = retriever.invoke(state["user_input"])
        state["retrieved_context"] = [
            {
                "content": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        
        state["llm_response"] = response
        state["messages"].append(AIMessage(content=response))
        print("✅ [explain_code] Explanation generated with LangChain RAG")
        
    except Exception as e:
        error_msg = f"Error generating explanation: {str(e)}"
        state["llm_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
        print(f"❌ [explain_code] {error_msg}")
    
    return state

def route_by_intent(state: AssistantState) -> str:
    """Route to appropriate node based on intent"""
    return state["intent"]