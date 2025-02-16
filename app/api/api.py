from fastapi import APIRouter
from app.api.endpoints import gallery, upload, chat

api_router = APIRouter()

api_router.include_router(gallery.router, prefix="/gallery", tags=["gallery"])
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
