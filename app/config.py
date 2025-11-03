import os
from typing import Set

"""
Configuration file for model names and settings
All models are loaded from Ollama container
"""

# --- Model Configuration ---
EMBEDDING_MODEL: str = "nomic-embed-text"
GENERATION_MODEL: str = "qwen3:0.6b"
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_CLIENT_TIMEOUT: float = 30.0

# --- Agent Configuration ---
AGENT_TEMPERATURE: float = 0.7
AGENT_MAX_TOKENS: int = 500

# --- Database Configuration ---
POSTGRES_DB: str = os.getenv("POSTGRES_DB")
POSTGRES_USER: str = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
DEFAULT_POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "postgres")
DEFAULT_POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
DEFAULT_TABLE_NAME: str = os.getenv("POSTGRES_TABLE", "rag_vectors")
EMBEDDING_DIM: int = 768

# --- Vector Store & Retrieval Configuration ---
DEFAULT_SIMILARITY_TOP_K: int = 3
DEFAULT_RETRIEVAL_LIMIT: int = 5
DEFAULT_DOC_LIMIT: int = 100
DEFAULT_DOC_LIMIT_SOURCE: int = 50
DEFAULT_DOC_LIMIT_METADATA: int = 50
DEFAULT_DOC_LIMIT_PATTERN: int = 50

# --- Data Directories Configuration ---
DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")
RAW_DOCS_DIR: str = os.path.join(DATA_DIR, "raw_docs")
CLEAR_DOCS_DIR: str = os.path.join(DATA_DIR, "clear_docs")

# --- Chunker Configuration ---
DEFAULT_CHUNK_SIZE: int = 1000
MIN_CHUNK_SIZE: int = 100
MAX_CHUNK_SIZE: int = 10000
CHUNK_OVERLAP_RATIO: float = 0.1  # 10% overlap

# --- API & Upload Configuration ---
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
ALLOWED_UPLOAD_EXTENSIONS: Set[str] = {'.docx', '.doc', '.md'}

DEFAULT_USER_ID: str = "test_user"
