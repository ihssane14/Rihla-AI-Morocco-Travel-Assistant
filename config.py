
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    
    # Pinecone Configuration
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rihla-morocco")
    
    # Hugging Face Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # RAG Configuration
    RAG_TOP_K: int = 3
    RAG_TEMPERATURE: float = 0.7
    RAG_MAX_TOKENS: int = 400
    
    # CORS Configuration
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

settings = Settings()