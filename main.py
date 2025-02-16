from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
from app.core.config import settings
from app.api.api import api_router
from app.services import QdrantService, get_qdrant_client
from contextlib import asynccontextmanager
import logging
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from app.core.logging_config import setup_logging
from app.core.middleware import RateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
rate_limiter = RateLimiter()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        # Initialize Qdrant client and service
        qdrant_client = get_qdrant_client()
        app.state.qdrant_service = QdrantService(qdrant_client)
        
        # Sync images with vector database
        new_images = await app.state.qdrant_service.sync_images_with_db()
        logger.info(f"Added {new_images} new images to vector database")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        
    yield
    
    # Shutdown
    try:
        from app.core.database import cleanup_client
        cleanup_client()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Create FastAPI app instance with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A web application for managing and searching photos",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Mount static files and images
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Initialize templates
templates = Jinja2Templates(directory="app/templates")

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Setup logging
setup_logging()

# Add rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/v1/gallery/search"):
        await rate_limiter.check_rate_limit(request)
    response = await call_next(request)
    return response

# Template routes
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        name="premium-homepage.html",
        context={"request": request}
    )

@app.get("/gallery")
async def gallery_page(request: Request):
    """Render the gallery page with images"""
    try:
        # Get services
        qdrant_client = get_qdrant_client()
        qdrant_service = QdrantService(qdrant_client)
        
        # Get all images
        images = await qdrant_service.get_all_images()
        
        # Format images for template
        formatted_images = []
        for image in images:
            formatted_images.append({
                "filename": image.get("filename", ""),
                "metadata": {
                    "description": image.get("analysis", {}).get("description", ""),
                    "place": image.get("place", "Unknown"),
                    "event": image.get("event", "Unknown"),
                    "who": image.get("who", "Unknown"),
                    "year": image.get("year", "Unknown")
                }
            })

        # Render template with images
        return templates.TemplateResponse(
            "premium-gallery.html",
            {
                "request": request,
                "images": formatted_images
            }
        )
    except Exception as e:
        logger.error(f"Error loading gallery: {e}")
        # Return template with error message
        return templates.TemplateResponse(
            "premium-gallery.html",
            {
                "request": request,
                "images": [],
                "error": str(e)
            }
        )

@app.get("/upload")
async def upload_page(request: Request):
    """Display the upload form"""
    return templates.TemplateResponse(
        name="upload.html",  # Make sure this template exists
        context={"request": request}
    )

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    who: str = Form(None),
    place: str = Form(None),
    event: str = Form(None),
    year: str = Form(None),
    description: str = Form(None)
):
    try:
        # Create images directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        
        # Save file
        file_path = f"images/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Prepare metadata
        metadata = {
            "who": who,
            "place": place,
            "event": event,
            "year": year,
            "description": description
        }
        
        # Store in Qdrant with metadata
        success = await app.state.qdrant_service.store_image_with_metadata(
            file_path, 
            metadata
        )
        
        if success:
            return {"success": True, "filename": file.filename}
        else:
            return {"success": False, "error": "Failed to store image"}
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/viewer")
async def viewer(request: Request):
    return templates.TemplateResponse(
        name="image-viewer.html",
        context={"request": request}
    )

@app.get("/chat")
async def chat(request: Request):
    return templates.TemplateResponse(
        name="chat-interface.html",
        context={"request": request}
    )

@app.get("/api/placeholder/{width}/{height}")
async def get_placeholder_image(width: int, height: int):
    """Generate a placeholder image with given dimensions"""
    try:
        # Create a new image with a dark background
        img = Image.new('RGB', (width, height), color='#1a1b23')
        draw = ImageDraw.Draw(img)
        
        # Draw a subtle border
        border_color = '#2a2b33'
        draw.rectangle([0, 0, width-1, height-1], outline=border_color)
        
        # Draw placeholder text
        text = f'{width} × {height}'
        # For simplicity, we'll use a basic font
        # You can add a custom font file to your project for better styling
        try:
            font_size = min(width, height) // 10
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Fallback to default font if arial is not available
            font = ImageFont.load_default()
            
        # Get text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate center position
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text in center
        draw.text((x, y), text, fill='#4a4b53', font=font)
        
        # Save image to bytes
        img_byte_array = BytesIO()
        img.save(img_byte_array, format='PNG')
        img_byte_array.seek(0)
        
        return StreamingResponse(img_byte_array, media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating placeholder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/{filename}")
async def image_detail(request: Request, filename: str):
    """Show detailed view of an image"""
    try:
        # Get services
        qdrant_client = get_qdrant_client()
        qdrant_service = QdrantService(qdrant_client)
        
        # Get image data
        image_data = await qdrant_service.get_image_by_filename(filename)
        
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Format the image data with proper structure
        formatted_image = {
            "filename": image_data.get("filename", ""),
            "metadata": {
                "description": image_data.get("analysis", {}).get("description", ""),
                "place": image_data.get("place", "Unknown"),
                "event": image_data.get("event", "Unknown"),
                "who": image_data.get("who", "Unknown"),
                "year": image_data.get("year", "Unknown"),
                "keywords": image_data.get("analysis", {}).get("keywords", [])
            }
        }
        
        return templates.TemplateResponse(
            "image-detail.html",
            {
                "request": request,
                "image": formatted_image
            }
        )
    except Exception as e:
        logger.error(f"Error loading image detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
