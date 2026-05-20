"""
Configuration settings cho RAG + Tavily Search API
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if not exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============= TAVILY API SETTINGS =============
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
TAVILY_SEARCH_DEPTH = "advanced"  # basic, advanced
USE_MOCK_TAVILY = os.getenv("USE_MOCK_TAVILY", "false").lower() == "true"  # Default to False = use real API
TAVILY_TIMEOUT = int(os.getenv("TAVILY_TIMEOUT", "30"))  # API timeout in seconds
TAVILY_MAX_RETRIES = int(os.getenv("TAVILY_MAX_RETRIES", "3"))  # Retry attempts
TAVILY_BATCH_SIZE = int(os.getenv("TAVILY_BATCH_SIZE", "5"))  # Max concurrent requests

# ============= LLM SETTINGS =============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000
LLM_LANGUAGE = os.getenv("LLM_LANGUAGE", "vietnamese")  # "vietnamese", "english", or other language

# ============= EMBEDDING SETTINGS =============
# Sử dụng mô hình embedding mạnh hơn cho đa ngôn ngữ/tiếng Việt
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
EMBEDDING_DIMENSION = 768  # for paraphrase-multilingual-mpnet-base-v2
VECTOR_DB_PATH = DATA_DIR / "vector_db"

# ============= RAG SETTINGS =============
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# ============= LOGGING SETTINGS =============
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_PATH = LOGS_DIR / "app.log"

# ============= RAG OPTIMIZATION SETTINGS =============
USE_QUERY_EXPANSION = True
USE_RERANKING = True
USE_CACHING = True
CACHE_TTL = 3600  # 1 hour

# ============= RERANKING SETTINGS =============
RERANKING_STRATEGY = os.getenv("RERANKING_STRATEGY", "mock_llm")  # mock_llm (recommended), hybrid, cross_encoder, semantic, llm
RERANKING_TOP_K = int(os.getenv("RERANKING_TOP_K", "5"))  # Documents to rerank

# Reranker specific settings
RERANKER_CROSS_ENCODER_MODEL = os.getenv("RERANKER_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_LLM_MODEL = os.getenv("RERANKER_LLM_MODEL", "gpt-3.5-turbo")

# Hybrid reranker weights (only used if strategy is 'hybrid')
RERANKER_WEIGHTS = {
    "cross_encoder": 0.5,
    "semantic": 0.3,
    "llm": 0.2
}

# Multi-stage retrieval
RETRIEVAL_STAGES = {
    "stage1_tavily": {
        "enabled": True,
        "weight": 0.4,
        "top_k": 5,
    },
    "stage2_vector": {
        "enabled": True,
        "weight": 0.35,
        "top_k": 5,
    },
    "stage3_hybrid": {
        "enabled": True,
        "weight": 0.25,
        "top_k": 5,
    },
}
