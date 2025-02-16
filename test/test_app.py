import asyncio
import aiohttp
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    async def test_endpoints(self):
        async with aiohttp.ClientSession() as session:
            await self.test_home(session)
            await self.test_gallery(session)
            await self.test_upload(session)
            await self.test_search(session)
            
    async def test_home(self, session):
        logger.info("Testing home endpoint...")
        async with session.get(f"{self.base_url}/") as response:
            assert response.status == 200
            logger.info("Home endpoint OK")
            
    async def test_gallery(self, session):
        logger.info("Testing gallery endpoint...")
        async with session.get(f"{self.base_url}/gallery") as response:
            assert response.status == 200
            logger.info("Gallery endpoint OK")
            
    async def test_upload(self, session):
        logger.info("Testing image upload...")
        test_image_path = Path("test_image.jpg")
        if not test_image_path.exists():
            logger.error("test_image.jpg not found!")
            return

        try:
            # Create form data
            data = aiohttp.FormData()
            
            # Read file content first
            file_content = open(test_image_path, 'rb').read()
            
            # Add file using content
            data.add_field('file',
                           file_content,
                           filename='test_image.jpg',
                           content_type='image/jpeg')
            
            # Add metadata fields
            data.add_field('who', 'Test Person')
            data.add_field('place', 'Test Location')
            data.add_field('event', 'Test Event')
            data.add_field('year', '2024')
            data.add_field('description', 'Test image for upload')
            
            # Send request
            async with session.post(
                f"{self.base_url}/upload",
                data=data
            ) as response:
                assert response.status == 200
                result = await response.json()
                assert result['success']
                logger.info("Upload endpoint OK")
                
        except Exception as e:
            logger.error(f"Upload test failed: {str(e)}")
            raise
            
    async def test_search(self, session):
        logger.info("Testing search functionality...")
        queries = ["test", "person", "location"]
        
        for query in queries:
            async with session.get(f"{self.base_url}/api/v1/gallery?query={query}") as response:
                assert response.status == 200
                logger.info(f"Search with query '{query}' OK")

async def main():
    try:
        tester = AppTester()
        await tester.test_endpoints()
        logger.info("✅ All tests passed!")
    except AssertionError as e:
        logger.error(f"❌ Test failed: Assertion Error")
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 