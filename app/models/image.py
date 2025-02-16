from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

class ImageAnalysis(BaseModel):
    description: str
    keywords: List[str]

class ImageMetadata(BaseModel):
    who: Optional[str] = None
    place: Optional[str] = None
    event: Optional[str] = None
    year: Optional[str] = None
    description: Optional[str] = None
    date: Optional[datetime] = None
    keywords: Optional[List[str]] = None
    analysis: Optional[ImageAnalysis] = None

class ImageResponse(BaseModel):
    filename: str
    path: str
    metadata: ImageMetadata

class ImageUploadResponse(BaseModel):
    success: bool
    filename: str
    message: Optional[str] = None
