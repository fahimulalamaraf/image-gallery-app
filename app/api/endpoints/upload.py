from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional
from app.models.image import ImageMetadata, ImageUploadResponse
from app.services import QdrantService
from app.core.database import get_qdrant_client
import shutil
import os
from app.core.config import settings
from datetime import datetime

router = APIRouter()

@router.post("/", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    who: Optional[str] = Form(None),
    place: Optional[str] = Form(None),
    event: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    qdrant_service: QdrantService = Depends(lambda: QdrantService(get_qdrant_client()))
):
    """Upload image with optional metadata"""
    try:
        # Create images directory if it doesn't exist
        os.makedirs(str(settings.IMAGES_DIR), exist_ok=True)
        
        # Save file
        file_path = settings.IMAGES_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Prepare metadata
        metadata = {
            "who": who,
            "place": place,
            "event": event,
            "year": year,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in Qdrant with embedding
        success = await qdrant_service.store_image_with_metadata(
            str(file_path), 
            metadata
        )
        
        if success:
            return ImageUploadResponse(
                success=True,
                filename=file.filename,
                message="Image uploaded successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to store image")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
