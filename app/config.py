from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "RAG Code Assistant"
    VERSION: str = "1.0.0"
    
    # API
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    

    # Optional: Override if needed
    LLM_MODEL: Optional[str] = None
    LLM_TEMPERATURE: Optional[float] = None
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()