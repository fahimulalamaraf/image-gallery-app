from fastapi import APIRouter, HTTPException, Depends, Query, status, Request
from app.services import QdrantService
from app.core.database import get_qdrant_client
from app.services.clip_service import ClipService
from app.services.gemini_service import GeminiService
from typing import List, Dict, Any
import logging
from qdrant_client.http import models
from app.models.search import SearchQuery
from app.core.cache import SearchCache
from app.core.middleware import RateLimiter
import hashlib
import time
import json

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services with dependency injection
def get_services():
    qdrant_client = get_qdrant_client()
    return {
        "qdrant_service": QdrantService(qdrant_client),
        "clip_service": ClipService(),
        "gemini_service": GeminiService()
    }

# Initialize cache and rate limiter
search_cache = SearchCache()
rate_limiter = RateLimiter()

class SearchError(Exception):
    """Custom exception for search-related errors"""
    def __init__(self, message: str, details: Dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

@router.get("/search")
async def search_images(
    request: Request,
    query: str = Query(..., description="Search query", min_length=2),
    services: Dict = Depends(get_services)
):
    """
    Search images using natural language query
    Handles both metadata and semantic search
    """
    try:
        # Log search request
        logger.info(f"Search request received - Query: {query}")

        # Generate cache key
        cache_key = hashlib.md5(query.encode()).hexdigest()

        # Check cache
        cached_result = search_cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {query}")
            return cached_result

        qdrant_service = services["qdrant_service"]
        clip_service = services["clip_service"]
        gemini_service = services["gemini_service"]

        # Extract search parameters using Gemini
        query_intent = await determine_query_intent(query, gemini_service, request)
        
        # Check if we hit Gemini rate limit
        if query_intent.get("rate_limited"):
            wait_time = query_intent.get("wait_time", 60)
            return {
                "success": True,
                "rate_limited": True,
                "message": f"Please wait {wait_time} seconds before making more complex queries. Using basic search for now.",
                "images": await perform_basic_search(query, services),  # Implement basic search
                "wait_time": wait_time
            }

        logger.info(f"Query intent determined: {query_intent}")

        all_results = []
        
        # Try metadata search first
        if query_intent.get("type") == "metadata":
            metadata_results = await search_by_metadata(query_intent, qdrant_service)
            all_results.extend(metadata_results)
            logger.info(f"Found {len(metadata_results)} metadata matches")

        # If no metadata results or semantic search needed
        if not all_results or query_intent.get("type") == "semantic":
            semantic_query = query_intent.get("semantic_description", query)
            query_embedding = clip_service.get_text_embedding(semantic_query)
            semantic_results = qdrant_service.search(
                query_vector=query_embedding.tolist(),
                limit=20
            )
            all_results.extend(semantic_results)
            logger.info(f"Found {len(semantic_results)} semantic matches")

        # Format and return results
        images = []
        seen_filenames = set()
        
        # Process metadata results first
        for result in all_results:
            filename = result.payload.get("filename")
            if filename and filename not in seen_filenames:
                seen_filenames.add(filename)
                
                # Calculate confidence score
                score = getattr(result, 'score', 1.0)
                if query_intent.get("type") == "metadata":
                    # Boost score for metadata matches
                    who_match = any(
                        person.lower() in result.payload.get("who", "").lower()
                        for person in query_intent["parameters"].get("who", [])
                    )
                    place_match = any(
                        place.lower() in result.payload.get("place", "").lower()
                        for place in query_intent["parameters"].get("place", [])
                    )
                    
                    if who_match:
                        score *= 1.5
                    if place_match:
                        score *= 1.3

                images.append({
                    "filename": filename,
                    "metadata": {
                        "description": result.payload.get("description", ""),
                        "place": result.payload.get("place", "Unknown"),
                        "event": result.payload.get("event", "Unknown"),
                        "who": result.payload.get("who", "Unknown"),
                        "year": result.payload.get("year", "Unknown")
                    },
                    "score": score
                })

        # Sort images by score
        images.sort(key=lambda x: x["score"], reverse=True)

        response_data = {
            "success": True,
            "images": images,
            "total": len(images),
            "query_understanding": query_intent
        }

        # Cache results
        search_cache.set(cache_key, response_data)
        
        return response_data

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return {
            "success": False,
            "message": str(e)
        }

async def determine_query_intent(query: str, gemini_service: GeminiService, request: Request) -> Dict:
    """Use Gemini to extract search parameters from natural language query"""
    prompt = f"""
    Analyze this search query and extract metadata parameters.
    Query: "{query}"

    IMPORTANT RULES:
    1. ONLY consider as metadata when:
       - Specific named people (e.g., "Ratul", "John") are mentioned
       - Specific locations (e.g., "Dhaka", "Times Square") are mentioned
       - Specific events (e.g., "birthday party", "graduation") are mentioned
       - Specific dates/years (e.g., "2023", "last summer") are mentioned
    
    2. Do NOT consider as metadata:
       - Generic terms like "people", "person", "someone"
       - Generic places like "outdoors", "inside"
       - Generic activities like "smiling", "walking"
       - Unspecified time references

    Return ONLY a JSON object in this EXACT format:
    {{
        "type": "metadata" if specific names/places/events/dates found, else "semantic",
        "parameters": {{
            "who": [],  # Only specific named people
            "place": [],  # Only specific locations
            "event": [],  # Only specific events
            "year": []  # Only specific dates/years
        }},
        "semantic_description": "brief description"
    }}

    Examples:
    1. "Show me photos of Ratul in Dhaka"
    {{
        "type": "metadata",
        "parameters": {{
            "who": ["Ratul"],
            "place": ["Dhaka"],
            "event": [],
            "year": []
        }},
        "semantic_description": "person in location"
    }}

    2. "People smiling outdoors"
    {{
        "type": "semantic",
        "parameters": {{
            "who": [],
            "place": [],
            "event": [],
            "year": []
        }},
        "semantic_description": "people smiling outdoors"
    }}

    Analyze this query: "{query}"
    Return ONLY the JSON object, no explanation.
    """
    
    try:
        # Get Gemini response
        response = await gemini_service.generate_content(prompt)
        logger.info(f"Raw Gemini response: {response}")
        
        try:
            # Parse JSON response
            result = json.loads(response)
            logger.info(f"Parsed result: {result}")
            
            # Ensure parameters exist and are lists
            parameters = result.get("parameters", {})
            cleaned_parameters = {
                "who": [],
                "place": [],
                "event": [],
                "year": []
            }
            
            # Clean and validate parameters
            for key in cleaned_parameters:
                value = parameters.get(key, [])
                if isinstance(value, str):
                    cleaned_parameters[key] = [value] if value else []
                elif isinstance(value, list):
                    cleaned_parameters[key] = [str(v) for v in value if v]
            
            # Check if any parameters have values
            has_metadata = any(
                len(params) > 0 
                for params in cleaned_parameters.values()
            )
            
            final_result = {
                "type": "metadata" if has_metadata else "semantic",
                "parameters": cleaned_parameters,
                "semantic_description": result.get("semantic_description", query)
            }
            
            logger.info(f"Final result: {final_result}")
            return final_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {response}")
            logger.error(f"JSON parse error: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in query intent detection: {str(e)}")
        return {
            "type": "semantic",
            "parameters": {
                "who": [],
                "place": [],
                "event": [],
                "year": []
            },
            "semantic_description": query
        }

async def search_by_metadata(query_intent: Dict, qdrant_service: QdrantService) -> List:
    """Search images by metadata with extracted parameters"""
    try:
        should = []
        parameters = query_intent.get("parameters", {})
        
        # Handle who parameter (people)
        if parameters.get("who"):
            # Use text match for exact and partial matches
            should.append(
                models.FieldCondition(
                    key="who",
                    match=models.MatchText(
                        text=" ".join(parameters["who"]).lower()
                    )
                )
            )
            # Add matches for individual names
            for person in parameters["who"]:
                should.append(
                    models.FieldCondition(
                        key="who",
                        match=models.MatchText(
                            text=person.lower()
                        )
                    )
                )

        # Handle place parameter
        if parameters.get("place"):
            should.append(
                models.FieldCondition(
                    key="place",
                    match=models.MatchText(
                        text=" ".join(parameters["place"]).lower()
                    )
                )
            )
            for place in parameters["place"]:
                should.append(
                    models.FieldCondition(
                        key="place",
                        match=models.MatchText(
                            text=place.lower()
                        )
                    )
                )

        # Execute search with combined conditions
        if should:  # Only search if we have conditions
            results = qdrant_service.client.scroll(
                collection_name=qdrant_service.collection_name,
                scroll_filter=models.Filter(
                    should=should,
                    must=[
                        models.FieldCondition(
                            key="who",
                            match=models.MatchText(
                                text="",
                                should_match_at_least=1
                            )
                        )
                    ]
                ),
                limit=20,
                with_payload=True
            )[0]

            # Enhanced scoring function
            def calculate_score(result, parameters):
                score = 0
                payload = result.payload
                
                # Score for who matches
                if parameters.get("who"):
                    who_text = payload.get("who", "").lower()
                    for person in parameters["who"]:
                        person = person.lower()
                        if person in who_text:
                            score += 10  # Higher weight for person matches
                            if person == who_text:
                                score += 5  # Bonus for exact match
                            if "me" in person and "me" in who_text:
                                score += 15  # Extra weight for "me" matches

                # Score for place matches
                if parameters.get("place"):
                    place_text = payload.get("place", "").lower()
                    for place in parameters["place"]:
                        place = place.lower()
                        if place in place_text:
                            score += 8  # Weight for place matches
                            if place == place_text:
                                score += 4  # Bonus for exact match

                return score

            # Sort results by calculated score
            sorted_results = sorted(
                results,
                key=lambda x: calculate_score(x, parameters),
                reverse=True
            )

            logger.info(f"Found {len(sorted_results)} metadata matches with scores")
            return sorted_results

        return []  # Return empty list if no search conditions

    except Exception as e:
        logger.error(f"Metadata search error: {str(e)}")
        return []

@router.get("/", response_model=List[Dict[str, Any]])
async def get_gallery_images(
    services: Dict = Depends(get_services)
):
    """Get all images from the vector database"""
    try:
        qdrant_service = services["qdrant_service"]
        results = await qdrant_service.get_all_images()
        
        # Format the results to match the expected structure
        formatted_results = []
        for result in results:
            formatted_results.append({
                "filename": result.get("filename", ""),
                "metadata": {
                    "description": result.get("analysis", {}).get("description", ""),
                    "place": result.get("place", "Unknown"),
                    "event": result.get("event", "Unknown"),
                    "who": result.get("who", "Unknown"),
                    "year": result.get("year", "Unknown")
                }
            })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error getting images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Update the test endpoint
@router.get("/test")
async def test_search_endpoints(
    request: Request,
    services: Dict = Depends(get_services)
):
    """Test various search scenarios"""
    test_results = []
    
    # Test cases
    test_cases = [
        {
            "name": "Metadata Search - Person",
            "query": "Show me photos of Ratul in Dhaka",
            "expected_type": "metadata"
        },
        {
            "name": "Metadata Search - Location",
            "query": "Photos from Dhaka",
            "expected_type": "metadata"
        },
        {
            "name": "Semantic Search",
            "query": "People smiling outdoors",
            "expected_type": "semantic"
        },
        {
            "name": "Empty Query",
            "query": "",
            "should_fail": True
        },
        {
            "name": "Short Query",
            "query": "a",
            "should_fail": True
        }
    ]
    
    for test in test_cases:
        try:
            if test.get("should_fail"):
                try:
                    await search_images(
                        request=request,
                        query=test["query"],
                        services=services
                    )
                    result = "Failed: Should have raised validation error"
                except Exception as e:
                    result = f"Passed: Caught expected error - {str(e)}"
            else:
                # Get query intent directly first
                query_intent = await determine_query_intent(test["query"], services["gemini_service"], request)
                logger.info(f"Test query: {test['query']}")
                logger.info(f"Query intent: {query_intent}")
                
                response = await search_images(
                    request=request,
                    query=test["query"],
                    services=services
                )
                
                if response["success"]:
                    query_type = response["query_understanding"]["type"]
                    result = {
                        "status": "Passed" if query_type == test["expected_type"] else "Failed",
                        "expected": test["expected_type"],
                        "got": query_type,
                        "parameters": response["query_understanding"]["parameters"]
                    }
                else:
                    result = f"Failed: {response['message']}"
                    
            test_results.append({
                "test": test["name"],
                "query": test["query"],
                "result": result
            })
            
        except Exception as e:
            test_results.append({
                "test": test["name"],
                "query": test["query"],
                "result": f"Error: {str(e)}"
            })
    
    return {
        "success": True,
        "results": test_results
    }

# Add cache test endpoint
@router.get("/test/cache")
async def test_cache(
    services: Dict = Depends(get_services)
):
    """Test cache functionality"""
    test_query = "Test cache query"
    
    # First request - should miss cache
    start_time = time.time()
    response1 = await search_images(
        request=Request,
        query=test_query,
        services=services
    )
    time1 = time.time() - start_time
    
    # Second request - should hit cache
    start_time = time.time()
    response2 = await search_images(
        request=Request,
        query=test_query,
        services=services
    )
    time2 = time.time() - start_time
    
    return {
        "success": True,
        "cache_test": {
            "first_request_time": time1,
            "second_request_time": time2,
            "cache_improvement": f"{(time1-time2)/time1*100:.2f}%",
            "cache_hit": time2 < time1
        }
    }

async def perform_basic_search(query: str, services: Dict) -> List[Dict]:
    """
    Perform basic vector similarity search when Gemini is rate limited
    Just uses CLIP embeddings to find similar images
    """
    try:
        clip_service = services["clip_service"]
        qdrant_service = services["qdrant_service"]

        # Get query embedding using CLIP
        query_embedding = clip_service.get_text_embedding(query)
        
        # Perform vector similarity search
        results = qdrant_service.search(
            query_vector=query_embedding.tolist(),
            limit=20
        )

        # Format results
        images = []
        seen_filenames = set()
        
        for result in results:
            filename = result.payload.get("filename")
            if filename and filename not in seen_filenames:
                seen_filenames.add(filename)
                images.append({
                    "filename": filename,
                    "metadata": {
                        "description": result.payload.get("description", ""),
                        "place": result.payload.get("place", "Unknown"),
                        "event": result.payload.get("event", "Unknown"),
                        "who": result.payload.get("who", "Unknown"),
                        "year": result.payload.get("year", "Unknown")
                    },
                    "score": getattr(result, 'score', 0.5)  # Default score of 0.5 for basic search
                })

        return images

    except Exception as e:
        logger.error(f"Error in basic search: {str(e)}")
        return []  # Return empty list on error
