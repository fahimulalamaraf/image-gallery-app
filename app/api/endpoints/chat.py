from fastapi import APIRouter, Depends, HTTPException
from app.services import QdrantService
from app.core.database import get_qdrant_client
from typing import Dict, Any, List

router = APIRouter()

@router.post("/query", response_model=Dict[str, Any])
async def chat_query(
    query: str,
    qdrant_service: QdrantService = Depends(lambda: QdrantService(get_qdrant_client()))
):
    """Handle chat queries about images"""
    try:
        # For now, just do a text search
        results = await qdrant_service.search_by_text(query)
        
        return {
            "response": f"I found {len(results)} images matching your query.",
            "images": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recent", response_model=List[Dict[str, Any]])
async def get_recent_chats():
    """Get recent chat history"""
    # This will be implemented later when we add chat history
    return []
