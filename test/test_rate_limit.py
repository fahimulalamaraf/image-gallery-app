import asyncio
import aiohttp
import time

async def test_rate_limit():
    print("Testing rate limiting...")
    
    # Track successful and rate-limited requests
    success_count = 0
    rate_limited_count = 0
    
    async with aiohttp.ClientSession() as session:
        # Make 70 requests (our limit is 60 per minute)
        tasks = []
        for i in range(70):
            tasks.append(
                session.get(
                    'http://localhost:8000/api/v1/gallery/search',
                    params={'query': f'test query {i}'}
                )
            )
        
        # Execute all requests concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze responses
        for resp in responses:
            if isinstance(resp, aiohttp.ClientResponse):
                if resp.status == 200:
                    success_count += 1
                elif resp.status == 429:  # Too Many Requests
                    rate_limited_count += 1
                await resp.close()
        
        print("\nRate Limit Test Results:")
        print(f"Total requests: 70")
        print(f"Successful requests: {success_count}")
        print(f"Rate limited requests: {rate_limited_count}")
        print(f"Total time: {total_time:.2f} seconds")
        
        if rate_limited_count > 0:
            print("\n✅ Rate limiting is working!")
        else:
            print("\n❌ Rate limiting might not be working properly")

if __name__ == "__main__":
    asyncio.run(test_rate_limit()) 