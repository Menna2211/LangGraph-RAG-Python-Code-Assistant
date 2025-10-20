from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "query": "Generate a function to calculate factorial"
            }]
        }
    }

class ContextItem(BaseModel):
    """Retrieved context item"""
    content: str
    metadata: Dict

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    success: bool
    query: str
    intent: str
    response: str
    retrieved_context: List[ContextItem]
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ExamplesResponse(BaseModel):
    """Examples response"""
    examples: List[str]