from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
from typing import Optional

from rag_langchain import setup_rag_pipeline
from graph import graph
from state import AssistantState

# Global state for sessions
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting RAG Code Assistant API...")
    setup_rag_pipeline()
    print("✅ System ready!")
    yield
    # Shutdown
    print("👋 Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Code Assistant API",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str

class HealthResponse(BaseModel):
    status: str
    version: str

@app.get("/")
async def root():
    """API info"""
    return {
        "message": "RAG Code Assistant API",
        "endpoints": {
            "POST /chat": "Send message to assistant",
            "GET /health": "Check API status",
            "GET /session/{id}": "Get session history"
        }
    }

# In your app.py - updated chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a chat message"""
    try:
        # Create or get session
        session_id = request.session_id or f"session_{len(sessions) + 1}"
        if session_id not in sessions:
            sessions[session_id] = []
        
        # Process through state machine
        initial_state = {
            "messages": [],
            "user_input": request.message,
            "intent": "",
            "retrieved_context": [],
            "llm_response": ""
        }
        
        final_state = graph.invoke(initial_state)
        
        # Extract response
        response_text = ""
        for message in reversed(final_state["messages"]):
            if hasattr(message, 'content'):
                response_text = message.content
                break
        
        # Clean up the response (remove extra quotes if needed)
        if response_text.startswith('"') and response_text.endswith('"'):
            response_text = response_text[1:-1]
        
        # Update session
        sessions[session_id].append({
            "user": request.message,
            "assistant": response_text,
            "intent": final_state.get("intent", "")
        })
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            intent=final_state.get("intent", "")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
        

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session history"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "messages": sessions[session_id],
        "count": len(sessions[session_id])
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)