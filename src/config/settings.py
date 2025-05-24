from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
import logging

# Load .env file from the project root
# project_root = os.path.dirname(os.path.abspath(__file__)) # This is src/config
# load_dotenv(os.path.join(project_root, ".env")) # This would look for .env in src/config

# Correctly determine project root (two levels up from src/config)
project_root_corrected = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up two levels
dotenv_path = os.path.join(project_root_corrected, ".env")
load_dotenv(dotenv_path)


logger = logging.getLogger(__name__)

logger.info(f"Project root for .env: {project_root_corrected}, attempting to load .env from: {dotenv_path}")

class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    PDFS_DIR: str = "pdfs"
    INDICES_DIR: str = "indices"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNKING_MODE: str = "both" # 'tokens', 'paragraphs', 'both'
    
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    OLLAMA_MODEL_NAME: str = "llama3.2"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 120

    RETRIEVAL_K: int = 4 # Default K for simple vector retrieval if used directly
    INITIAL_VECTOR_K: int = 50
    VECTOR_DISTANCE_THRESHOLD: float = 1.0
    FINAL_BM25_K: int = 6
    DEVICE_CONFIGURATION: str = "cpu" # 'cpu' || 'cuda' || 'npu' || 'mps'

    API_PDF_MAX_SIZE_MB: int = 100
    
    UVICORN_HOST: str = "0.0.0.0"
    UVICORN_PORT: int = 8000
    UVICORN_TIMEOUT_KEEP_ALIVE: int = 120

    RESPONSE_CACHE_MAX_SIZE: int = 100
    RESPONSE_CACHE_TTL_SECONDS: int = 3600

    SERVICE_CACHE_MAX_SIZE: int = 10 # Max number of service instances (and their vector stores) to keep in memory

    # Ensure directories exist
    def __init__(self, **values):
        super().__init__(**values)
        os.makedirs(self.PDFS_DIR, exist_ok=True)
        os.makedirs(self.INDICES_DIR, exist_ok=True)

settings = Settings()

# Example usage:
if __name__ == "__main__":
    logger.info(f"PDFs directory: {settings.PDFS_DIR}")
    logger.info(f"Ollama Model: {settings.OLLAMA_MODEL_NAME}")
    logger.info(f"Max PDF Upload Size (MB): {settings.API_PDF_MAX_SIZE_MB}")
