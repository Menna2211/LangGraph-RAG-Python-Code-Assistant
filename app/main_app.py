from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from app.config import settings
from app.Pydantic_Models import (
    QueryRequest, 
    QueryResponse, 
    ContextItem,
    HealthResponse,
    ExamplesResponse
)

# Import your existing modules
from graph import graph
from rag_langchain import code_rag_chain, explain_rag_chain, retriever
from state import AssistantState

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG system on startup"""
    print("🚀 Initializing RAG LangGraph System...")
    
    # Your RAG system is already initialized in rag_langchain.py
    # Just verify it's ready
    if code_rag_chain is None or explain_rag_chain is None:
        raise RuntimeError("RAG chains not initialized!")
    
    print("✅ System ready!")
    yield
    print("👋 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered code generation and explanation assistant using LangGraph and LangChain",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= ROUTES =============

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="online",
        message=f"{settings.PROJECT_NAME} API is running. Visit /docs for documentation."
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )

@app.get("/examples", response_model=ExamplesResponse)
async def get_examples():
    """Get example queries"""
    return ExamplesResponse(
        examples=[
            "Generate a function to calculate factorial",
            "Explain how binary search works",
            "Write a function to reverse a string",
            "Create a function to check if a number is prime",
            "Explain what recursion is",
            "Generate a function to sort a list",
            "How does dynamic programming work?",
            "Write a function to find the nth Fibonacci number"
        ]
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the RAG LangGraph system
    
    This uses your existing state machine with router for intent classification
    """
    try:
        # Use your existing state structure
        initial_state: AssistantState = {
            "messages": [],
            "user_input": request.query,
            "intent": "",
            "retrieved_context": [],
            "llm_response": ""
        }
        
        # Execute your graph
        final_state = graph.invoke(initial_state)
        
        # Extract response
        response_text = final_state.get("llm_response", "No response generated.")
        
        return QueryResponse(
            success=True,
            query=request.query,
            intent=final_state.get("intent", "unknown"),
            response=response_text,
            retrieved_context=[
                ContextItem(
                    content=ctx.get("content", ""),
                    metadata=ctx.get("metadata", {})
                )
                for ctx in final_state.get("retrieved_context", [])
            ]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/generate", response_model=QueryResponse)
async def generate_code(request: QueryRequest):
    """
    Force code generation (skip router)
    
    Uses your existing generate_code_node directly
    """
    try:
        from nodes_langchain import chat_node, generate_code_node
        
        # Create initial state
        initial_state: AssistantState = {
            "messages": [],
            "user_input": request.query,
            "intent": "generate_code",
            "retrieved_context": [],
            "llm_response": ""
        }
        
        # Process through nodes
        state = chat_node(initial_state)
        state["intent"] = "generate_code"
        final_state = generate_code_node(state)
        
        return QueryResponse(
            success=True,
            query=request.query,
            intent="generate_code",
            response=final_state.get("llm_response", "No response generated."),
            retrieved_context=[
                ContextItem(
                    content=ctx.get("content", ""),
                    metadata=ctx.get("metadata", {})
                )
                for ctx in final_state.get("retrieved_context", [])
            ]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating code: {str(e)}"
        )

@app.post("/explain", response_model=QueryResponse)
async def explain_code(request: QueryRequest):
    """
    Force code explanation (skip router)
    
    Uses your existing explain_code_node directly
    """
    try:
        from nodes_langchain import chat_node, explain_code_node
        
        # Create initial state
        initial_state: AssistantState = {
            "messages": [],
            "user_input": request.query,
            "intent": "explain_code",
            "retrieved_context": [],
            "llm_response": ""
        }
        
        # Process through nodes
        state = chat_node(initial_state)
        state["intent"] = "explain_code"
        final_state = explain_code_node(state)
        
        return QueryResponse(
            success=True,
            query=request.query,
            intent="explain_code",
            response=final_state.get("llm_response", "No response generated."),
            retrieved_context=[
                ContextItem(
                    content=ctx.get("content", ""),
                    metadata=ctx.get("metadata", {})
                )
                for ctx in final_state.get("retrieved_context", [])
            ]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error explaining code: {str(e)}"
        )

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )