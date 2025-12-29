"""
Mock External API - Simulates external service behavior

Used to demonstrate:
- Normal operation
- Slow responses (to trigger timeouts)
- Failures (to trigger circuit breaker)
"""

import asyncio
import logging
import os
import random
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuration
RESPONSE_DELAY_SECONDS = float(os.getenv("RESPONSE_DELAY_SECONDS", "0"))
RANDOM_DELAY_MAX = float(os.getenv("RANDOM_DELAY_MAX", "0"))
FAILURE_RATE = float(os.getenv("FAILURE_RATE", "0"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock External API")


class WeatherResponse(BaseModel):
    location: str
    temperature: float
    conditions: str
    humidity: int
    timestamp: str
    source: str = "mock_api"


class TimeResponse(BaseModel):
    timezone: str
    time: str
    timestamp: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mock_external_api"}


@app.get("/weather", response_model=WeatherResponse)
async def get_weather(location: str, units: str = "metric"):
    """
    Mock weather endpoint.
    
    Simulates:
    - Configurable delay
    - Random failures
    """
    # Simulate delay (for testing timeouts)
    total_delay = RESPONSE_DELAY_SECONDS
    if RANDOM_DELAY_MAX > 0:
        total_delay += random.uniform(0, RANDOM_DELAY_MAX)
    
    if total_delay > 0:
        logger.info(f"Weather request: delaying {total_delay:.2f}s")
        await asyncio.sleep(total_delay)
    
    # Simulate random failures
    if FAILURE_RATE > 0 and random.random() < FAILURE_RATE:
        logger.warning("Weather request: simulating failure")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    # Generate mock weather data
    temp = random.uniform(-10, 35) if units == "metric" else random.uniform(14, 95)
    conditions = random.choice(["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Overcast"])
    
    return WeatherResponse(
        location=location,
        temperature=round(temp, 1),
        conditions=conditions,
        humidity=random.randint(30, 90),
        timestamp=datetime.now().isoformat()
    )


@app.get("/time", response_model=TimeResponse)
async def get_time(timezone: str = "UTC"):
    """Mock time endpoint."""
    # Simulate delay
    total_delay = RESPONSE_DELAY_SECONDS
    if RANDOM_DELAY_MAX > 0:
        total_delay += random.uniform(0, RANDOM_DELAY_MAX)
    
    if total_delay > 0:
        await asyncio.sleep(total_delay)
    
    # Simulate random failures
    if FAILURE_RATE > 0 and random.random() < FAILURE_RATE:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    return TimeResponse(
        timezone=timezone,
        time=datetime.now().strftime("%H:%M:%S"),
        timestamp=datetime.now().isoformat()
    )


@app.get("/search")
async def search(query: str, limit: int = 10):
    """Mock search endpoint."""
    # Simulate delay
    total_delay = RESPONSE_DELAY_SECONDS
    if RANDOM_DELAY_MAX > 0:
        total_delay += random.uniform(0, RANDOM_DELAY_MAX)
    
    if total_delay > 0:
        await asyncio.sleep(total_delay)
    
    # Simulate random failures
    if FAILURE_RATE > 0 and random.random() < FAILURE_RATE:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
    # Generate mock search results
    results = [
        {
            "title": f"Result {i+1} for '{query}'",
            "url": f"https://example.com/result/{i+1}",
            "snippet": f"This is a mock search result for your query about {query}..."
        }
        for i in range(min(limit, 10))
    ]
    
    return {
        "query": query,
        "total_results": random.randint(100, 10000),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
