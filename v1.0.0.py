import torch
import clip
import google.generativeai as genai
from PIL import Image
import os
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import uuid
import torch.nn.functional as F
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass, asdict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysis:
    """Structure for image analysis results"""
    description: str
    objects: List[str]
    activities: List[str]
    colors: List[str]
    style: str
    tags: List[str]
    emotions: List[str]


@dataclass
class ImageMetadata:
    """Structure for image metadata"""
    filename: str
    path: str
    who: str
    place: str
    year: str
    event: str
    analysis: ImageAnalysis
    timestamp: str


class FunctionRegistry:
    """Registry of available functions for Gemini"""

    @staticmethod
    def search_by_description(query: str, top_k: int = 5) -> Dict:
        """Search images by natural language description"""
        return {
            "name": "search_by_description",
            "description": "Search images using natural language description",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5}
                }
            }
        }

    @staticmethod
    def search_by_metadata(filters: Dict[str, str], top_k: int = 5) -> Dict:
        """Search images by metadata (year, place, event, etc.)"""
        return {
            "name": "search_by_metadata",
            "description": "Search images using metadata filters",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "properties": {
                            "year": {"type": "string"},
                            "place": {"type": "string"},
                            "event": {"type": "string"},
                            "who": {"type": "string"}
                        }
                    },
                    "top_k": {"type": "integer", "default": 5}
                }
            }
        }

    @staticmethod
    def visual_search(image_path: str, top_k: int = 5) -> Dict:
        """Search visually similar images"""
        return {
            "name": "visual_search",
            "description": "Find visually similar images",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5}
                }
            }
        }


class ConversationalMemoryBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Initialize Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
            }
        )

        # Initialize Qdrant
        self.qdrant = QdrantClient(path="qdrant_db")
        self.COLLECTION_NAME = "image_embeddings"
        self.image_dir = "../images"

        # Initialize collection
        self._ensure_collection_exists()

        # Register available functions
        self.functions = {
            "search_by_description": self._search_by_description,
            "search_by_metadata": self._search_by_metadata,
            "visual_search": self._visual_search
        }

        # Initialize conversation context
        self.conversation_context = []

    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with proper schema"""
        try:
            self.qdrant.get_collection(self.COLLECTION_NAME)
        except:
            self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )

    async def handle_query(self, query: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for handling user queries"""
        try:
            # Update conversation context
            self.conversation_context.append({"role": "user", "content": query})

            # Determine query intent using Gemini
            intent = await self._determine_intent(query, image_path)

            # Execute appropriate function based on intent
            if intent["function"] == "search_by_description":
                results = await self._search_by_description(query)
            elif intent["function"] == "search_by_metadata":
                results = await self._search_by_metadata(intent["filters"])
            elif intent["function"] == "visual_search" and image_path:
                results = await self._visual_search(image_path)
            else:
                return {"error": "Could not determine appropriate action"}

            # Generate response
            response = await self._generate_response(query, results)

            return {
                "type": "search_results",
                "results": results,
                "response": response,
                "message": intent.get("message", "")
            }

        except Exception as e:
            logger.error(f"Error handling query: {e}")
            return {"error": str(e)}

    async def _determine_intent(self, query: str, image_path: Optional[str] = None) -> Dict[str, str]:
        """Determine query intent using Gemini"""
        prompt = f"""
        Analyze this query and determine the appropriate function to call:
        Query: {query}
        Image Provided: {"Yes" if image_path else "No"}

        Available functions:
        1. search_by_description: For natural language queries about image content
        2. search_by_metadata: For queries about specific metadata (year, place, event)
        3. visual_search: For finding similar images (requires image input)

        Return a JSON object with:
        1. function: The function name to call
        2. filters: Any metadata filters (for metadata search)
        3. message: A natural language response to the user
        """

        response = await self.gemini.generate_content(prompt)
        return json.loads(response.text)

    async def _search_by_description(self, query: str, top_k: int = 5) -> List[Dict]:
        """Execute natural language search"""
        # Generate text embedding
        text_embedding = self._get_text_embedding(query)

        # Search in Qdrant
        results = self.qdrant.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=text_embedding.tolist(),
            limit=top_k
        )

        return self._format_results(results)

    async def _search_by_metadata(self, filters: Dict[str, str], top_k: int = 5) -> List[Dict]:
        """Execute metadata-based search"""
        filter_conditions = {"should": []}
        for key, value in filters.items():
            if value:
                filter_conditions["should"].append({
                    "key": key,
                    "match": {"value": value.lower()}
                })

        results = self.qdrant.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=models.Filter(**filter_conditions),
            limit=top_k
        )[0]

        return self._format_results(results)

    async def _visual_search(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """Execute visual similarity search"""
        # Generate image embedding
        image_embedding = self._get_image_embedding(image_path)

        # Search in Qdrant
        results = self.qdrant.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=image_embedding.tolist(),
            limit=top_k
        )

        return self._format_results(results)

    async def upload_image(self, image_path: str, metadata: Optional[Dict] = None) -> bool:
        """Upload and index a new image"""
        try:
            # Analyze image
            analysis = await self._analyze_image(image_path)

            # Prepare metadata
            image_metadata = ImageMetadata(
                filename=os.path.basename(image_path),
                path=image_path,
                who=metadata.get('who', 'Unknown'),
                place=metadata.get('place', 'Unknown'),
                year=metadata.get('year', 'Unknown'),
                event=metadata.get('event', 'Unknown'),
                analysis=analysis,
                timestamp=datetime.now().isoformat()
            )

            # Generate embedding
            embedding = self._get_image_embedding(image_path)

            # Store in Qdrant
            self.qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=asdict(image_metadata)
                )]
            )

            return True

        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            return False

    async def _analyze_image(self, image_path: str) -> ImageAnalysis:
        """Analyze image using Gemini"""
        image = Image.open(image_path).convert('RGB')

        prompt = """
        Analyze this image and provide:
        1. Detailed description
        2. List of objects
        3. Activities happening
        4. Dominant colors
        5. Visual style
        6. Relevant tags
        7. Emotions conveyed

        Return as JSON.
        """

        response = await self.gemini.generate_content([image, prompt])
        analysis_data = json.loads(response.text)

        return ImageAnalysis(**analysis_data)

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Generate CLIP text embedding"""
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            embedding = self.clip_model.encode_text(text_tokens)
            return F.normalize(embedding, p=2, dim=-1)[0]

    def _get_image_embedding(self, image_path: str) -> torch.Tensor:
        """Generate CLIP image embedding"""
        with torch.no_grad():
            image = self.preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.device)
            embedding = self.clip_model.encode_image(image)
            return F.normalize(embedding, p=2, dim=-1)[0]

    def _format_results(self, results: List) -> List[Dict]:
        """Format search results"""
        formatted = []
        for result in results:
            formatted.append({
                "filename": result.payload["filename"],
                "metadata": result.payload,
                "score": result.score if hasattr(result, 'score') else None
            })
        return formatted

    async def _generate_response(self, query: str, results: List[Dict]) -> str:
        """Generate natural language response"""
        prompt = f"""
        Generate a natural response to the user's query based on the search results.

        Query: {query}
        Results: {json.dumps(results, indent=2)}

        Respond conversationally, mentioning key details about the found images.
        """

        response = await self.gemini.generate_content(prompt)
        return response.text


def display_results(results: List[Dict]):
    """Display search results"""
    if not results:
        print("No images found.")
        return

    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        image_path = os.path.join("../images", result["filename"])
        if os.path.exists(image_path):
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis("off")

            score = result.get("score")
            if isinstance(score, (float, int)):
                ax.set_title(f"Score: {score:.2f}")
            else:
                ax.set_title(result["filename"])

    plt.show()


async def main():
    """Main conversation loop"""
    bot = ConversationalMemoryBot()
    print("👋 Welcome to your Personal Photo Memory Bot!")
    print("\nYou can:")
    print("1. Search by description (e.g., 'Show me photos of dogs')")
    print("2. Search by metadata (e.g., 'Find pictures from 2023 in Paris')")
    print("3. Search by similarity (e.g., 'Find similar images to this one')")
    print("4. Upload new images")
    print("\nType 'exit' to quit")

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Check if user wants to upload an image
        if user_input.lower().startswith("upload"):
            image_path = input("Enter image path: ").strip()
            metadata = {
                "who": input("Who is in the image? (Press Enter to skip): ").strip() or "Unknown",
                "place": input("Where was this taken? (Press Enter to skip): ").strip() or "Unknown",
                "year": input("What year? (Press Enter to skip): ").strip() or "Unknown",
                "event": input("What event? (Press Enter to skip): ").strip() or "Unknown"
            }

            success = await bot.upload_image(image_path, metadata)
            if success:
                print("✅ Image uploaded successfully!")
            else:
                print("❌ Failed to upload image")
            continue

        # Handle regular queries
        try:
            # Check if query includes an image
            image_path = None
            if "similar to" in user_input.lower():
                image_path = input("Enter path to reference image: ").strip()

            # Process query
            response = await bot.handle_query(user_input, image_path)

            if "error" in response:
                print(f"❌ Error: {response['error']}")
                continue

            # Display natural language response
            print(f"\n🤖 Bot: {response['response']}")

            # Display results
            if response.get("results"):
                print("\nFound these images:")
                display_results(response["results"])
            else:
                print("\nNo matching images found.")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Something went wrong: {str(e)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())