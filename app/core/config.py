from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "MemoryBot"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    IMAGES_DIR: Path = BASE_DIR / "images"
    STATIC_DIR: Path = BASE_DIR / "app" / "static"
    
    # Qdrant settings
    QDRANT_PATH: str = "qdrant_storage"
    QDRANT_COLLECTION_NAME: str = "image_embeddings"
    VECTOR_SIZE: int = 512

    # Gemini settings
    GEMINI_API_KEY: str

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
