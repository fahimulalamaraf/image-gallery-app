from qdrant_client import QdrantClient
from pathlib import Path
from app.core.config import settings
import atexit
import time
import os

# Global client instance
_client = None

def get_qdrant_client() -> QdrantClient:
    """Get or create the Qdrant client instance"""
    global _client
    if _client is None:
        # Use existing database or create new one
        db_path = Path(settings.QDRANT_PATH)
        
        # Clean up any stale lock files
        lock_file = db_path / ".lock"
        if lock_file.exists():
            try:
                os.remove(lock_file)
            except:
                pass
        
        # Create directory if it doesn't exist
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Try to create client with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                _client = QdrantClient(path=str(db_path))
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
        
        # Register cleanup
        atexit.register(cleanup_client)
    
    return _client

def cleanup_client():
    """Cleanup function to close client connection"""
    global _client
    if _client is not None:
        try:
            _client.close()
        except:
            pass
        _client = None 