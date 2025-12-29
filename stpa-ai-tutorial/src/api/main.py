"""
API Gateway Service - STPA Resilient Implementation

This service implements safety controls derived from STPA analysis:
- SC4: Request timeouts
- SC6: Rate limiting

Mitigates:
- UCA3.1: No rate limiting allows DoS
- UCA1.3: RAG routing delayed causes timeout
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
import redis.asyncio as redis

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    rag_service_url: str = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
    mcp_service_url: str = os.getenv("MCP_SERVICE_URL", "http://localhost:8002")
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "10"))
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    rate_limit_per_user: bool = os.getenv("RATE_LIMIT_PER_USER", "true").lower() == "true"
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

# =============================================================================
# Metrics (for observability - supports STPA feedback loops)
# =============================================================================

REQUEST_COUNT = Counter(
    'api_gateway_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_gateway_request_latency_seconds',
    'Request latency',
    ['method', 'endpoint']
)

RATE_LIMIT_HITS = Counter(
    'api_gateway_rate_limit_hits_total',
    'Rate limit hits',
    ['user_id']
)

TIMEOUT_COUNT = Counter(
    'api_gateway_timeouts_total',
    'Request timeouts',
    ['service']
)

# =============================================================================
# Models
# =============================================================================

class QueryRequest(BaseModel):
    """Input model with validation (SC2 - parameter sanitization)"""
    query: str = Field(..., min_length=1, max_length=10000)
    user_id: Optional[str] = Field(default="anonymous", max_length=100)
    include_weather: bool = Field(default=False)
    include_documents: bool = Field(default=True)


class QueryResponse(BaseModel):
    """Response model"""
    query: str
    rag_response: Optional[dict] = None
    mcp_response: Optional[dict] = None
    processing_time_ms: float
    degraded: bool = False
    warnings: list[str] = []


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: dict[str, str]
    rate_limiter: str


# =============================================================================
# Rate Limiter (SC6 Implementation)
# =============================================================================

class RateLimiter:
    """
    Sliding window rate limiter using Redis.
    
    STPA Context:
    - Enforces SC6: Request rate must be limited per user and globally
    - Prevents UCA3.1: No rate limiting allows DoS
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.enabled = config.rate_limit_enabled
        self.max_requests = config.rate_limit_requests
        self.window_seconds = config.rate_limit_window
    
    async def check_rate_limit(self, key: str) -> tuple[bool, int]:
        """
        Check if request is within rate limit.
        
        Returns:
            (allowed: bool, remaining: int)
        """
        if not self.enabled:
            return True, self.max_requests
        
        now = time.time()
        window_start = now - self.window_seconds
        
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Count requests in window
        pipe.zcard(key)
        # Set expiry
        pipe.expire(key, self.window_seconds)
        
        results = await pipe.execute()
        request_count = results[2]
        
        allowed = request_count <= self.max_requests
        remaining = max(0, self.max_requests - request_count)
        
        if not allowed:
            RATE_LIMIT_HITS.labels(user_id=key).inc()
            logger.warning(f"Rate limit exceeded for {key}")
        
        return allowed, remaining


# =============================================================================
# Service Clients with Timeout (SC4 Implementation)
# =============================================================================

class ServiceClient:
    """
    HTTP client with timeout and error handling.
    
    STPA Context:
    - Enforces SC4: External API calls must have timeouts
    - Prevents UCA6.4/UCA9.4: Operations not terminated on timeout
    """
    
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
    
    async def call(self, endpoint: str, data: dict) -> dict:
        """Make a request with timeout handling."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await self.client.post(url, json=data)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        
        except httpx.TimeoutException:
            TIMEOUT_COUNT.labels(service=self.base_url).inc()
            logger.error(f"Timeout calling {url}")
            return {"success": False, "error": "timeout", "message": "Service timeout"}
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from {url}: {e.response.status_code}")
            return {"success": False, "error": "http_error", "status": e.response.status_code}
        
        except Exception as e:
            logger.error(f"Error calling {url}: {e}")
            return {"success": False, "error": "unknown", "message": str(e)}
    
    async def health_check(self) -> bool:
        """Check service health."""
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=httpx.Timeout(2.0)
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        await self.client.aclose()


# =============================================================================
# Application Setup
# =============================================================================

redis_client: Optional[redis.Redis] = None
rate_limiter: Optional[RateLimiter] = None
rag_client: Optional[ServiceClient] = None
mcp_client: Optional[ServiceClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global redis_client, rate_limiter, rag_client, mcp_client
    
    # Initialize Redis
    redis_client = redis.from_url(config.redis_url, decode_responses=True)
    rate_limiter = RateLimiter(redis_client)
    
    # Initialize service clients with timeout (SC4)
    rag_client = ServiceClient(config.rag_service_url, config.request_timeout)
    mcp_client = ServiceClient(config.mcp_service_url, config.request_timeout)
    
    logger.info("API Gateway started with STPA safety controls")
    logger.info(f"  - Request timeout: {config.request_timeout}s")
    logger.info(f"  - Rate limiting: {config.rate_limit_enabled}")
    
    yield
    
    # Cleanup
    await rag_client.close()
    await mcp_client.close()
    await redis_client.close()


app = FastAPI(
    title="AI System API Gateway",
    description="STPA-compliant API Gateway with safety controls",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Middleware for Rate Limiting
# =============================================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Apply rate limiting before processing request.
    
    STPA: Enforces SC6, prevents UCA3.1
    """
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Determine rate limit key
    if config.rate_limit_per_user:
        # Try to get user ID from header or use IP
        user_id = request.headers.get("X-User-ID", request.client.host)
        key = f"rate_limit:user:{user_id}"
    else:
        key = "rate_limit:global"
    
    # Check rate limit
    allowed, remaining = await rate_limiter.check_rate_limit(key)
    
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after_seconds": config.rate_limit_window
            },
            headers={"Retry-After": str(config.rate_limit_window)}
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = str(config.rate_limit_requests)
    
    return response


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    STPA: Provides feedback for control structure monitoring
    """
    rag_healthy = await rag_client.health_check()
    mcp_healthy = await mcp_client.health_check()
    
    try:
        await redis_client.ping()
        redis_healthy = True
    except Exception:
        redis_healthy = False
    
    all_healthy = rag_healthy and mcp_healthy and redis_healthy
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services={
            "rag": "healthy" if rag_healthy else "unhealthy",
            "mcp": "healthy" if mcp_healthy else "unhealthy",
            "redis": "healthy" if redis_healthy else "unhealthy"
        },
        rate_limiter="enabled" if config.rate_limit_enabled else "disabled"
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint for observability."""
    return JSONResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain"
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main query endpoint with parallel service calls and timeout handling.
    
    STPA Safety Controls:
    - SC4: All service calls have timeouts
    - Graceful degradation if services fail
    """
    start_time = time.time()
    warnings = []
    degraded = False
    
    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/query",
        status="processing"
    ).inc()
    
    # Build tasks for parallel execution
    tasks = {}
    
    if request.include_documents:
        tasks["rag"] = rag_client.call("/retrieve", {
            "query": request.query,
            "user_id": request.user_id
        })
    
    if request.include_weather:
        tasks["mcp"] = mcp_client.call("/execute", {
            "tool": "weather",
            "params": {"query": request.query},
            "user_id": request.user_id
        })
    
    # Execute with global timeout (SC4)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks.values(), return_exceptions=True),
            timeout=config.request_timeout
        )
        results_dict = dict(zip(tasks.keys(), results))
    except asyncio.TimeoutError:
        TIMEOUT_COUNT.labels(service="global").inc()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout exceeded"
        )
    
    # Process results with graceful degradation
    rag_response = None
    mcp_response = None
    
    if "rag" in results_dict:
        result = results_dict["rag"]
        if isinstance(result, Exception):
            warnings.append(f"RAG service error: {str(result)}")
            degraded = True
        elif result.get("success"):
            rag_response = result.get("data")
        else:
            warnings.append(f"RAG service: {result.get('message', 'Unknown error')}")
            degraded = True
    
    if "mcp" in results_dict:
        result = results_dict["mcp"]
        if isinstance(result, Exception):
            warnings.append(f"MCP service error: {str(result)}")
            degraded = True
        elif result.get("success"):
            mcp_response = result.get("data")
        else:
            warnings.append(f"MCP service: {result.get('message', 'Unknown error')}")
            degraded = True
    
    processing_time = (time.time() - start_time) * 1000
    
    REQUEST_LATENCY.labels(method="POST", endpoint="/query").observe(processing_time / 1000)
    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/query",
        status="degraded" if degraded else "success"
    ).inc()
    
    return QueryResponse(
        query=request.query,
        rag_response=rag_response,
        mcp_response=mcp_response,
        processing_time_ms=processing_time,
        degraded=degraded,
        warnings=warnings
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
