import google.generativeai as genai
from PIL import Image
import os
from typing import Dict, Any, List
from app.core.config import settings
import logging
from fastapi import Request
from app.core.middleware import RateLimiter

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        try:
            # Initialize Gemini with API key from settings
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("Gemini service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {e}")
            raise
        
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Generate description and keywords for an image"""
        try:
            img = Image.open(image_path)
            
            # Generate description
            try:
                description = self._generate_description(img)
            except Exception as e:
                logger.error(f"Error generating description: {e}")
                description = ""
            
            # Generate keywords
            try:
                keywords = self._generate_keywords(img)
            except Exception as e:
                logger.error(f"Error generating keywords: {e}")
                keywords = []
            
            return {
                "description": description,
                "keywords": keywords
            }
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                "description": "",
                "keywords": []
            }

    def _generate_description(self, image: Image) -> str:
        """Generate a natural description of the image"""
        try:
            prompt = """
            Analyze this image and provide a natural, detailed description of what you see.
            Focus on:
            - The main subjects/people
            - The setting/location
            - The activity or event
            - Notable details or emotions
            Keep the description concise but informative (2-3 sentences).
            """
            
            response = self.model.generate_content([prompt, image])
            return response.text.strip() if response.text else ""
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return ""

    def _generate_keywords(self, image: Image) -> List[str]:
        """Generate relevant keywords/tags for the image"""
        try:
            prompt = """
            Analyze this image and generate relevant keywords/tags.
            Include:
            - Objects/subjects
            - Actions/activities
            - Emotions/mood
            - Setting/location
            - Style/composition
            Return only the keywords as a comma-separated list.
            """
            
            response = self.model.generate_content([prompt, image])
            if response.text:
                keywords = [k.strip() for k in response.text.split(',')]
                return keywords
            return []
        except Exception as e:
            logger.error(f"Error generating keywords: {e}")
            return []

    async def generate_content(self, prompt: str, request: Request = None) -> str:
        """Generate content with rate limiting"""
        try:
            if request:
                # Check Gemini rate limit
                is_allowed, wait_time = await RateLimiter.check_rate_limit(request, is_gemini=True)
                if not is_allowed:
                    logger.warning(f"Gemini rate limit exceeded. Wait time: {wait_time} seconds")
                    return {
                        "type": "semantic",  # Fall back to semantic search
                        "parameters": {
                            "who": [],
                            "place": [],
                            "event": [],
                            "year": []
                        },
                        "semantic_description": prompt,
                        "rate_limited": True,
                        "wait_time": wait_time
                    }

            response = self.model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise
