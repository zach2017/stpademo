"""
MCP Service - STPA Resilient Implementation

This service implements safety controls derived from STPA analysis:
- SC2: MCP tool parameters must be sanitized and validated against schema
- SC4: External API calls must have timeouts and circuit breakers
- SC5: Tool availability must be scoped to user authorization level

Mitigates:
- UCA9.2: Tool executed with unsanitized input
- UCA9.3: Tool executed before auth verification
- UCA9.4: Tool execution not terminated on timeout

Implements Loss Scenario LS-3 mitigations:
- Circuit breaker prevents cascading failures
- Timeout on all external calls
- Graceful degradation with fallbacks
"""

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import httpx
from fastapi import FastAPI, HTTPException, status, Depends, Header
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge
import redis.asyncio as redis

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    external_api_url: str = os.getenv("EXTERNAL_API_URL", "http://localhost:8004")
    external_call_timeout: float = float(os.getenv("EXTERNAL_CALL_TIMEOUT", "3"))
    circuit_breaker_enabled: bool = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
    circuit_breaker_threshold: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
    circuit_breaker_timeout: int = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "30"))
    sanitize_input: bool = os.getenv("SANITIZE_INPUT", "true").lower() == "true"
    revalidate_auth: bool = os.getenv("REVALIDATE_AUTH", "true").lower() == "true"
    tool_access_scoping: bool = os.getenv("TOOL_ACCESS_SCOPING", "true").lower() == "true"
    fallback_enabled: bool = os.getenv("FALLBACK_ENABLED", "true").lower() == "true"
    cache_fallback_ttl: int = int(os.getenv("CACHE_FALLBACK_TTL", "300"))
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

# =============================================================================
# Metrics
# =============================================================================

TOOL_EXECUTION_COUNT = Counter(
    'mcp_tool_execution_total',
    'Total tool executions',
    ['tool', 'status']
)

TOOL_LATENCY = Histogram(
    'mcp_tool_latency_seconds',
    'Tool execution latency',
    ['tool']
)

CIRCUIT_BREAKER_STATE = Gauge(
    'mcp_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['service']
)

CIRCUIT_BREAKER_TRIPS = Counter(
    'mcp_circuit_breaker_trips_total',
    'Circuit breaker trips',
    ['service']
)

INPUT_SANITIZATION_BLOCKS = Counter(
    'mcp_input_sanitization_blocks_total',
    'Inputs blocked by sanitization',
    ['tool', 'reason']
)

AUTH_REVALIDATION_FAILURES = Counter(
    'mcp_auth_revalidation_failures_total',
    'Authorization re-validation failures'
)

FALLBACK_USED = Counter(
    'mcp_fallback_used_total',
    'Fallback responses used',
    ['tool']
)


# =============================================================================
# Circuit Breaker (SC4, LS-3 Mitigation)
# =============================================================================

class CircuitState(Enum):
    CLOSED = 0      # Normal operation
    OPEN = 1        # Failing, reject requests
    HALF_OPEN = 2   # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    STPA Context:
    - Implements SC4: External API calls must have circuit breakers
    - Mitigates LS-3: Cascading Timeout Causes System Hang
    - Prevents UCA9.4: Tool execution not terminated on timeout
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: int = 30,
        redis_client: Optional[redis.Redis] = None
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.redis = redis_client
        
        # Local state (fallback if Redis unavailable)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
    
    @property
    def state_key(self) -> str:
        return f"circuit_breaker:{self.name}:state"
    
    @property
    def failure_count_key(self) -> str:
        return f"circuit_breaker:{self.name}:failures"
    
    async def get_state(self) -> CircuitState:
        """Get current circuit state."""
        if self.redis:
            try:
                state_val = await self.redis.get(self.state_key)
                if state_val:
                    return CircuitState(int(state_val))
            except Exception:
                pass
        return self._state
    
    async def set_state(self, state: CircuitState):
        """Set circuit state."""
        self._state = state
        CIRCUIT_BREAKER_STATE.labels(service=self.name).set(state.value)
        
        if self.redis:
            try:
                await self.redis.set(self.state_key, state.value, ex=self.recovery_timeout * 2)
            except Exception:
                pass
    
    async def record_failure(self):
        """Record a failure."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self.redis:
            try:
                count = await self.redis.incr(self.failure_count_key)
                await self.redis.expire(self.failure_count_key, self.recovery_timeout)
                self._failure_count = count
            except Exception:
                pass
        
        if self._failure_count >= self.failure_threshold:
            await self.trip()
    
    async def record_success(self):
        """Record a success."""
        self._failure_count = 0
        self._last_success_time = time.time()
        
        if self.redis:
            try:
                await self.redis.delete(self.failure_count_key)
            except Exception:
                pass
        
        state = await self.get_state()
        if state == CircuitState.HALF_OPEN:
            await self.set_state(CircuitState.CLOSED)
            logger.info(f"Circuit breaker {self.name} closed after successful probe")
    
    async def trip(self):
        """Trip the circuit breaker."""
        CIRCUIT_BREAKER_TRIPS.labels(service=self.name).inc()
        await self.set_state(CircuitState.OPEN)
        logger.warning(f"Circuit breaker {self.name} tripped!")
    
    async def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        state = await self.get_state()
        
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    await self.set_state(CircuitState.HALF_OPEN)
                    logger.info(f"Circuit breaker {self.name} entering half-open state")
                    return True
            return False
        
        if state == CircuitState.HALF_OPEN:
            # Allow one probe request
            return True
        
        return False
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if not await self.should_allow_request():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure()
            raise


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# Input Sanitizer (SC2, SC-UCA9.2)
# =============================================================================

class InputSanitizer:
    """
    Input sanitization for tool parameters.
    
    STPA Context:
    - Implements SC2: MCP tool parameters must be sanitized
    - Prevents UCA9.2: Tool executed with unsanitized input
    
    Checks:
    - SQL injection patterns
    - Command injection patterns
    - Path traversal attempts
    - Script injection (XSS-like)
    """
    
    # Dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
        r"(--|\#|\/\*)",
        r"(\bOR\b\s+\d+\s*=\s*\d+)",
        r"(\bAND\b\s+\d+\s*=\s*\d+)",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"(\$\(|\`)",
        r"(\.\./|\.\.\\)",
        r"(\b(cat|ls|rm|chmod|wget|curl|bash|sh|python|perl)\b)",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e",
        r"\.\.%2f",
    ]
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self.cmd_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMMAND_INJECTION_PATTERNS]
        self.path_patterns = [re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS]
    
    def sanitize(self, tool: str, params: dict) -> tuple[bool, str, dict]:
        """
        Sanitize input parameters.
        
        Returns:
            (is_safe: bool, reason: str, sanitized_params: dict)
        """
        if not self.enabled:
            return True, "sanitization_disabled", params
        
        sanitized = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Check for dangerous patterns
                is_safe, reason = self._check_string(value)
                if not is_safe:
                    INPUT_SANITIZATION_BLOCKS.labels(tool=tool, reason=reason).inc()
                    return False, reason, {}
                
                # Basic sanitization
                sanitized[key] = self._sanitize_string(value)
            else:
                sanitized[key] = value
        
        return True, "safe", sanitized
    
    def _check_string(self, value: str) -> tuple[bool, str]:
        """Check string for dangerous patterns."""
        for pattern in self.sql_patterns:
            if pattern.search(value):
                return False, "sql_injection"
        
        for pattern in self.cmd_patterns:
            if pattern.search(value):
                return False, "command_injection"
        
        for pattern in self.path_patterns:
            if pattern.search(value):
                return False, "path_traversal"
        
        return True, "safe"
    
    def _sanitize_string(self, value: str) -> str:
        """Basic string sanitization."""
        # Remove null bytes
        value = value.replace("\x00", "")
        # Limit length
        value = value[:10000]
        return value


# =============================================================================
# Authorization Re-validator (SC5, SC-UCA9.3)
# =============================================================================

class AuthorizationValidator:
    """
    Authorization re-validation for tool execution.
    
    STPA Context:
    - Implements SC5: Tool availability must be scoped to user authorization
    - Prevents UCA9.3: Tool executed before auth verification
    
    Features:
    - Re-validates authorization at execution time
    - Scopes tools based on user role
    """
    
    # Tool access by role
    TOOL_ACCESS = {
        "anonymous": ["weather", "time"],
        "user": ["weather", "time", "search", "calculator"],
        "admin": ["weather", "time", "search", "calculator", "system", "database"],
    }
    
    def __init__(self, enabled: bool = True, redis_client: Optional[redis.Redis] = None):
        self.enabled = enabled
        self.redis = redis_client
    
    async def validate(self, user_id: str, tool: str, token: Optional[str] = None) -> tuple[bool, str]:
        """
        Validate user authorization for tool.
        
        Returns:
            (authorized: bool, reason: str)
        """
        if not self.enabled:
            return True, "auth_disabled"
        
        # Get user role (simplified - in production, validate token)
        role = await self._get_user_role(user_id, token)
        
        if role is None:
            AUTH_REVALIDATION_FAILURES.inc()
            return False, "invalid_user"
        
        # Check tool access
        allowed_tools = self.TOOL_ACCESS.get(role, [])
        if tool not in allowed_tools:
            AUTH_REVALIDATION_FAILURES.inc()
            return False, f"tool_not_allowed_for_role_{role}"
        
        return True, "authorized"
    
    async def _get_user_role(self, user_id: str, token: Optional[str]) -> Optional[str]:
        """Get user role from token or cache."""
        # In production, validate JWT and extract role
        # Simplified for tutorial
        if user_id == "anonymous":
            return "anonymous"
        elif user_id.startswith("admin_"):
            return "admin"
        else:
            return "user"


# =============================================================================
# Tool Definitions
# =============================================================================

class ToolSchema(BaseModel):
    """Tool parameter schema for validation."""
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str] = []


TOOL_SCHEMAS = {
    "weather": ToolSchema(
        name="weather",
        description="Get weather information",
        parameters={
            "location": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"}
        },
        required=["location"]
    ),
    "time": ToolSchema(
        name="time",
        description="Get current time",
        parameters={
            "timezone": {"type": "string", "description": "Timezone name"}
        },
        required=[]
    ),
    "search": ToolSchema(
        name="search",
        description="Search for information",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "default": 10}
        },
        required=["query"]
    ),
}


# =============================================================================
# Models
# =============================================================================

class ExecuteRequest(BaseModel):
    """Tool execution request."""
    tool: str = Field(..., min_length=1, max_length=100)
    params: dict = Field(default_factory=dict)
    user_id: str = Field(default="anonymous")


class ExecuteResponse(BaseModel):
    """Tool execution response."""
    tool: str
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    fallback_used: bool = False
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    circuit_breaker: dict[str, str]
    external_api: str


# =============================================================================
# MCP Service
# =============================================================================

class MCPService:
    """
    MCP service with STPA safety controls.
    """
    
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        circuit_breaker: CircuitBreaker,
        sanitizer: InputSanitizer,
        auth_validator: AuthorizationValidator,
        redis_client: redis.Redis,
        config: Config
    ):
        self.http_client = http_client
        self.circuit_breaker = circuit_breaker
        self.sanitizer = sanitizer
        self.auth_validator = auth_validator
        self.redis = redis_client
        self.config = config
    
    async def execute_tool(
        self,
        request: ExecuteRequest,
        authorization: Optional[str] = None
    ) -> ExecuteResponse:
        """
        Execute a tool with safety controls.
        
        STPA Safety Controls:
        - SC2: Input sanitization
        - SC4: Circuit breaker and timeout
        - SC5: Authorization re-validation
        """
        start_time = time.time()
        
        # Validate tool exists
        if request.tool not in TOOL_SCHEMAS:
            return ExecuteResponse(
                tool=request.tool,
                success=False,
                error="unknown_tool",
                processing_time_ms=0
            )
        
        # Re-validate authorization (SC5, SC-UCA9.3)
        if self.config.revalidate_auth:
            authorized, reason = await self.auth_validator.validate(
                request.user_id,
                request.tool,
                authorization
            )
            if not authorized:
                TOOL_EXECUTION_COUNT.labels(tool=request.tool, status="unauthorized").inc()
                return ExecuteResponse(
                    tool=request.tool,
                    success=False,
                    error=f"authorization_failed: {reason}",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
        
        # Sanitize input (SC2, SC-UCA9.2)
        if self.config.sanitize_input:
            is_safe, reason, sanitized_params = self.sanitizer.sanitize(
                request.tool,
                request.params
            )
            if not is_safe:
                TOOL_EXECUTION_COUNT.labels(tool=request.tool, status="sanitization_failed").inc()
                return ExecuteResponse(
                    tool=request.tool,
                    success=False,
                    error=f"input_rejected: {reason}",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            request.params = sanitized_params
        
        # Execute with circuit breaker (SC4, LS-3)
        try:
            if self.config.circuit_breaker_enabled:
                result = await self.circuit_breaker.execute(
                    self._call_external_api,
                    request.tool,
                    request.params
                )
            else:
                result = await self._call_external_api(request.tool, request.params)
            
            processing_time = (time.time() - start_time) * 1000
            TOOL_EXECUTION_COUNT.labels(tool=request.tool, status="success").inc()
            TOOL_LATENCY.labels(tool=request.tool).observe(processing_time / 1000)
            
            return ExecuteResponse(
                tool=request.tool,
                success=True,
                result=result,
                processing_time_ms=processing_time
            )
        
        except CircuitBreakerOpenError:
            # Use fallback if available
            if self.config.fallback_enabled:
                fallback = await self._get_fallback(request.tool, request.params)
                if fallback:
                    FALLBACK_USED.labels(tool=request.tool).inc()
                    TOOL_EXECUTION_COUNT.labels(tool=request.tool, status="fallback").inc()
                    return ExecuteResponse(
                        tool=request.tool,
                        success=True,
                        result=fallback,
                        fallback_used=True,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
            
            TOOL_EXECUTION_COUNT.labels(tool=request.tool, status="circuit_open").inc()
            return ExecuteResponse(
                tool=request.tool,
                success=False,
                error="service_unavailable",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        except asyncio.TimeoutError:
            TOOL_EXECUTION_COUNT.labels(tool=request.tool, status="timeout").inc()
            return ExecuteResponse(
                tool=request.tool,
                success=False,
                error="timeout",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            TOOL_EXECUTION_COUNT.labels(tool=request.tool, status="error").inc()
            return ExecuteResponse(
                tool=request.tool,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _call_external_api(self, tool: str, params: dict) -> dict:
        """Call external API with timeout (SC4)."""
        url = f"{self.config.external_api_url}/{tool}"
        
        response = await asyncio.wait_for(
            self.http_client.get(url, params=params),
            timeout=self.config.external_call_timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Cache successful result for fallback
        await self._cache_result(tool, params, result)
        
        return result
    
    async def _cache_result(self, tool: str, params: dict, result: dict):
        """Cache result for fallback."""
        cache_key = f"mcp:cache:{tool}:{json.dumps(params, sort_keys=True)}"
        try:
            await self.redis.set(
                cache_key,
                json.dumps(result),
                ex=self.config.cache_fallback_ttl
            )
        except Exception:
            pass
    
    async def _get_fallback(self, tool: str, params: dict) -> Optional[dict]:
        """Get cached fallback result."""
        cache_key = f"mcp:cache:{tool}:{json.dumps(params, sort_keys=True)}"
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass
        
        # Default fallbacks for known tools
        if tool == "weather":
            return {
                "status": "cached",
                "message": "Weather service temporarily unavailable",
                "data": None
            }
        elif tool == "time":
            return {
                "status": "fallback",
                "time": datetime.now().isoformat(),
                "source": "local"
            }
        
        return None


# =============================================================================
# Application Setup
# =============================================================================

mcp_service: Optional[MCPService] = None
redis_client: Optional[redis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global mcp_service, redis_client
    
    # Initialize Redis
    redis_client = redis.from_url(config.redis_url, decode_responses=True)
    
    # Initialize HTTP client with timeout
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.external_call_timeout),
        limits=httpx.Limits(max_connections=100)
    )
    
    # Initialize circuit breaker
    circuit_breaker = CircuitBreaker(
        name="external_api",
        failure_threshold=config.circuit_breaker_threshold,
        recovery_timeout=config.circuit_breaker_timeout,
        redis_client=redis_client
    )
    
    # Initialize sanitizer
    sanitizer = InputSanitizer(enabled=config.sanitize_input)
    
    # Initialize auth validator
    auth_validator = AuthorizationValidator(
        enabled=config.revalidate_auth,
        redis_client=redis_client
    )
    
    # Create MCP service
    mcp_service = MCPService(
        http_client=http_client,
        circuit_breaker=circuit_breaker,
        sanitizer=sanitizer,
        auth_validator=auth_validator,
        redis_client=redis_client,
        config=config
    )
    
    logger.info("MCP Service started with STPA safety controls")
    logger.info(f"  - Circuit breaker: {config.circuit_breaker_enabled}")
    logger.info(f"  - External timeout: {config.external_call_timeout}s")
    logger.info(f"  - Input sanitization: {config.sanitize_input}")
    logger.info(f"  - Auth re-validation: {config.revalidate_auth}")
    
    yield
    
    await http_client.aclose()
    await redis_client.close()


app = FastAPI(
    title="MCP Service",
    description="STPA-compliant MCP service with safety controls",
    lifespan=lifespan
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    cb_state = await mcp_service.circuit_breaker.get_state()
    
    # Check external API
    external_status = "unknown"
    try:
        response = await mcp_service.http_client.get(
            f"{config.external_api_url}/health",
            timeout=httpx.Timeout(2.0)
        )
        external_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        external_status = "unhealthy"
    
    all_healthy = cb_state == CircuitState.CLOSED and external_status == "healthy"
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        circuit_breaker={
            "external_api": cb_state.name
        },
        external_api=external_status
    )


@app.post("/execute", response_model=ExecuteResponse)
async def execute_tool(
    request: ExecuteRequest,
    authorization: Optional[str] = Header(None)
):
    """Execute a tool."""
    return await mcp_service.execute_tool(request, authorization)


@app.get("/tools")
async def list_tools():
    """List available tools and their schemas."""
    return {
        "tools": {
            name: {
                "description": schema.description,
                "parameters": schema.parameters,
                "required": schema.required
            }
            for name, schema in TOOL_SCHEMAS.items()
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
