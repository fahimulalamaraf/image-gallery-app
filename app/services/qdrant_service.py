from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import settings
from app.services.clip_service import ClipService
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid
from pathlib import Path
import os
from app.services.gemini_service import GeminiService
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.clip_service = ClipService()
        self.gemini_service = GeminiService()
        self.collection_name = "image_embeddings"
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with proper schema"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    async def sync_images_with_db(self):
        """Sync images from directory with vector database"""
        try:
            # Get all images from directory
            image_dir = Path("images")
            image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}]
            
            logger.info(f"Found {len(image_files)} images in directory")
            
            # Get existing images from Qdrant
            existing = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000
            )[0]
            existing_paths = {p.payload.get('path') for p in existing if p.payload}
            
            # Find new images
            new_images = [f for f in image_files if str(f.absolute()) not in existing_paths]
            logger.info(f"Found {len(new_images)} new images to process")

            # Store new images
            success_count = 0
            for image_file in new_images:
                try:
                    success = await self.store_image_with_metadata(
                        str(image_file.absolute()),
                        metadata=None
                    )
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {e}")

            return success_count

        except Exception as e:
            logger.error(f"Error syncing images: {e}")
            return 0

    async def store_image_with_metadata(
        self, 
        image_path: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store image with its embedding and metadata"""
        try:
            # Generate embedding
            embedding = self.clip_service.get_image_embedding(image_path)
            
            # Initialize metadata if None
            if metadata is None:
                metadata = {}
            
            # Generate image analysis
            analysis = self.gemini_service.analyze_image(image_path)
            
            # Prepare metadata with defaults and analysis
            image_data = {
                "id": str(uuid.uuid4()),
                "filename": Path(image_path).name,
                "path": str(image_path),
                "who": metadata.get('who', 'Unknown'),
                "place": metadata.get('place', 'Unknown'),
                "year": metadata.get('year', 'Unknown'),
                "event": metadata.get('event', 'Unknown'),
                "timestamp": datetime.now().isoformat(),
                "description": metadata.get('description') or analysis.get('description', ''),
                "keywords": analysis.get('keywords', []),
                "analysis": analysis
            }

            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=image_data["id"],
                    vector=embedding.tolist(),
                    payload=image_data
                )]
            )
            return True

        except Exception as e:
            logger.error(f"Error storing image: {e}")
            return False

    async def get_all_images(self) -> List[Dict[str, Any]]:
        """Get all images from the collection"""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,  # limit for efficiency
                with_payload=True,
                with_vectors=False  # We don't need vectors for display
            )[0]
            
            # Return full payload for each image
            return [result.payload for result in results]
        except Exception as e:
            logger.error(f"Error getting images: {e}")
            raise

    def search(self, query_vector: List[float], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection
        """
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return search_results
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    async def add_image(self, vector: List[float], payload: Dict[str, Any]) -> bool:
        """Add an image vector and metadata to the collection"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=payload.get("id"),
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            return True
        except Exception as e:
            logger.error(f"Error adding image: {e}")
            return False

    def delete_image(self, image_id: str) -> bool:
        """Delete an image from the collection"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[image_id]
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting image: {e}")
            return False

    def _format_result(self, result) -> Dict[str, Any]:
        """Format a single result"""
        try:
            return {
                "filename": result.payload.get("filename", ""),
                "path": f"/images/{Path(result.payload.get('filename', '')).name}",
                "metadata": {
                    "who": result.payload.get("who", "Unknown"),
                    "place": result.payload.get("place", "Unknown"),
                    "year": result.payload.get("year", "Unknown"),
                    "event": result.payload.get("event", "Unknown"),
                    "description": result.payload.get("description", ""),
                    "date": result.payload.get("timestamp", ""),
                    "keywords": result.payload.get("keywords", []),
                    "score": getattr(result, 'score', None)
                }
            }
        except Exception as e:
            logger.error(f"Error formatting result: {e}")
            return {
                "filename": "error.jpg",
                "path": "/images/error.jpg",
                "metadata": {
                    "description": "Error loading image",
                    "date": "",
                    "place": "",
                    "event": "",
                    "who": "",
                    "year": "",
                    "keywords": [],
                    "score": None
                }
            }

    async def search_by_text(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search images using text query"""
        try:
            # Generate text embedding
            text_embedding = self.clip_service.get_text_embedding(query)
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=text_embedding.tolist(),
                limit=limit,
                with_payload=True
            )
            
            return [self._format_result(result) for result in results]
        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return []

    async def search_similar_images(self, image_path: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar images"""
        try:
            # Generate image embedding
            image_embedding = self.clip_service.get_image_embedding(image_path)
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=image_embedding.tolist(),
                limit=limit
            )
            
            return self._format_results(results)
        except Exception as e:
            logger.error(f"Error searching similar images: {e}")
            return []

    def _format_results(self, results: List) -> List[Dict[str, Any]]:
        """Format search results"""
        formatted = []
        for result in results:
            formatted.append({
                "filename": result.payload["filename"],
                "path": f"/images/{result.payload['filename']}",
                "metadata": {
                    "description": result.payload.get("analysis", {}).get("description", ""),
                    "date": result.payload.get("timestamp", ""),
                    "place": result.payload.get("place", "Unknown"),
                    "event": result.payload.get("event", ""),
                    "score": result.score
                }
            })
        return formatted

    async def store_image(self, image_data: Dict[str, Any], vector: List[float]):
        """Store image data and its vector in Qdrant"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=image_data["id"],
                    vector=vector,
                    payload=image_data
                )]
            )
            return True
        except Exception as e:
            logger.error(f"Error storing image: {e}")
            return False

    async def get_image_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get image data by filename"""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )[0]
            
            if not results:
                return None
            
            # Get the raw payload
            payload = results[0].payload
            
            # Return formatted data
            return {
                "filename": payload.get("filename", ""),
                "place": payload.get("place", "Unknown"),
                "event": payload.get("event", "Unknown"),
                "who": payload.get("who", "Unknown"),
                "year": payload.get("year", "Unknown"),
                "analysis": {  # Keep analysis for description and keywords
                    "description": payload.get("analysis", {}).get("description", ""),
                    "keywords": payload.get("analysis", {}).get("keywords", [])
                }
            }
        except Exception as e:
            logger.error(f"Error getting image by filename: {e}")
            raise

    def get_payload_field(self, payload: Dict, field: str, default: str = "Unknown") -> str:
        """Safely get field from payload"""
        try:
            value = payload.get(field, default)
            return value if value and value.lower() != "unknown" else default
        except:
            return default
