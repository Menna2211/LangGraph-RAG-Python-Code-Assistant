# main.py
from rag_langchain import setup_rag_pipeline
from graph import graph
from state import AssistantState
from plot import save_langgraph_png


def initialize_system():
    """Initialize the RAG system with LangChain"""
    print("🚀 Initializing RAG LangGraph System with LangChain...")
    
    # Setup LangChain RAG pipeline
    global code_rag_chain, explain_rag_chain, retriever, vectorstore
    code_rag_chain, explain_rag_chain, retriever, vectorstore = setup_rag_pipeline()
    
    print("✅ System ready!")

def process_query(user_input: str) -> str:
    """Process a user query through the state machine"""
    initial_state = {
        "messages": [],
        "user_input": user_input,
        "intent": "",
        "retrieved_context": [],
        "llm_response": ""
    }
    
    try:
        # Execute the graph
        final_state = graph.invoke(initial_state)
        
        # Return the latest AI response
        for message in reversed(final_state["messages"]):
            if hasattr(message, 'content'):
                return message.content
        
        return "No response generated."
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

def chat_loop():
    """Main chat loop"""
    initialize_system()
    
    # Show graph structure
    save_langgraph_png()
    
    print("\n" + "="*50)
    print("🤖 RAG Code Assistant")
    print("="*50)
    print("Ask me to generate or explain Python code!")
    print("Examples:")
    print("  - 'Generate a function to calculate factorial'")
    print("  - 'Explain how binary search works'")
    print("  - 'Write a function to reverse a string'")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process through state machine
            response = process_query(user_input)
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# Global variables for the RAG pipeline
code_rag_chain = None
explain_rag_chain = None
retriever = None
vectorstore = None

if __name__ == "__main__":
    chat_loop()