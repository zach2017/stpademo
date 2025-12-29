"""
RAG Service - STPA Resilient Implementation

This service implements safety controls derived from STPA analysis:
- SC1: All retrieved context must be validated for relevance
- SC3: Vector store health must be verified before retrieval

Mitigates:
- UCA6.2: Query with malformed embedding returns wrong results
- UCA6.3: Query during index rebuild returns stale data
- UCA6.4: Query timeout too long blocks service
- UCA7.2: Response generated with low-relevance context
"""

import asyncio
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    chromadb_host: str = os.getenv("CHROMADB_HOST", "localhost")
    chromadb_port: int = int(os.getenv("CHROMADB_PORT", "8000"))
    validate_embeddings: bool = os.getenv("VALIDATE_EMBEDDINGS", "true").lower() == "true"
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    min_embedding_norm: float = float(os.getenv("MIN_EMBEDDING_NORM", "0.001"))
    query_timeout: float = float(os.getenv("QUERY_TIMEOUT", "3"))
    max_connections: int = int(os.getenv("MAX_CONNECTIONS", "20"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.5"))
    health_check_chromadb: bool = os.getenv("HEALTH_CHECK_CHROMADB", "true").lower() == "true"
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    model_name: str = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

# =============================================================================
# Metrics
# =============================================================================

RETRIEVAL_COUNT = Counter(
    'rag_retrieval_total',
    'Total retrieval operations',
    ['status']
)

RETRIEVAL_LATENCY = Histogram(
    'rag_retrieval_latency_seconds',
    'Retrieval latency'
)

EMBEDDING_VALIDATION_FAILURES = Counter(
    'rag_embedding_validation_failures_total',
    'Embedding validation failures',
    ['reason']
)

RELEVANCE_SCORE = Histogram(
    'rag_relevance_score',
    'Relevance scores of retrieved documents',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

LOW_RELEVANCE_FILTERED = Counter(
    'rag_low_relevance_filtered_total',
    'Documents filtered due to low relevance'
)

INDEX_REBUILD_LOCK_WAITS = Counter(
    'rag_index_rebuild_lock_waits_total',
    'Times query waited for index rebuild'
)

CHROMADB_HEALTH = Gauge(
    'rag_chromadb_healthy',
    'ChromaDB health status (1=healthy, 0=unhealthy)'
)


# =============================================================================
# Models
# =============================================================================

class RetrieveRequest(BaseModel):
    """Retrieval request model"""
    query: str = Field(..., min_length=1, max_length=10000)
    user_id: str = Field(default="anonymous")
    top_k: int = Field(default=5, ge=1, le=20)
    min_relevance: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Document(BaseModel):
    """Retrieved document"""
    id: str
    content: str
    metadata: dict
    relevance_score: float


class RetrieveResponse(BaseModel):
    """Retrieval response"""
    query: str
    documents: list[Document]
    total_retrieved: int
    filtered_count: int
    embedding_valid: bool
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    chromadb: str
    redis: str
    model_loaded: bool
    collection_count: Optional[int] = None


# =============================================================================
# Embedding Validator (SC-UCA6.2 Implementation)
# =============================================================================

class EmbeddingValidator:
    """
    Validates embedding vectors before use.
    
    STPA Context:
    - Implements SC-UCA6.2: Embedding vectors must be validated
    - Prevents UCA6.2: Query with malformed embedding returns wrong results
    
    Validation checks:
    1. Correct dimensionality
    2. All values finite (no NaN, no Inf)
    3. Non-zero norm (not a zero vector)
    4. Reasonable magnitude range
    """
    
    def __init__(self, expected_dim: int, min_norm: float):
        self.expected_dim = expected_dim
        self.min_norm = min_norm
    
    def validate(self, embedding: list[float]) -> tuple[bool, str]:
        """
        Validate an embedding vector.
        
        Returns:
            (is_valid: bool, reason: str)
        """
        # Check dimensionality
        if len(embedding) != self.expected_dim:
            EMBEDDING_VALIDATION_FAILURES.labels(reason="wrong_dimension").inc()
            return False, f"Expected {self.expected_dim} dimensions, got {len(embedding)}"
        
        # Convert to numpy for efficient computation
        arr = np.array(embedding)
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(arr)):
            EMBEDDING_VALIDATION_FAILURES.labels(reason="non_finite").inc()
            return False, "Embedding contains NaN or Inf values"
        
        # Check norm
        norm = np.linalg.norm(arr)
        if norm < self.min_norm:
            EMBEDDING_VALIDATION_FAILURES.labels(reason="zero_norm").inc()
            return False, f"Embedding norm {norm} below minimum {self.min_norm}"
        
        # Check for extreme values (potential corruption)
        if np.max(np.abs(arr)) > 100:
            EMBEDDING_VALIDATION_FAILURES.labels(reason="extreme_values").inc()
            return False, "Embedding contains extreme values"
        
        return True, "valid"
    
    def normalize(self, embedding: list[float]) -> list[float]:
        """Normalize embedding to unit length."""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()


# =============================================================================
# Index Rebuild Lock (LS-1 Mitigation)
# =============================================================================

class IndexRebuildLock:
    """
    Distributed lock for coordinating index rebuilds.
    
    STPA Context:
    - Mitigates LS-1: Stale Context During Index Rebuild
    - Prevents UCA6.3: Query during index rebuild returns stale data
    
    Uses Redis to coordinate read/write access across instances.
    """
    
    LOCK_KEY = "chromadb:index_rebuild_lock"
    LOCK_TTL = 300  # 5 minutes max rebuild time
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_rebuilding(self) -> bool:
        """Check if index rebuild is in progress."""
        return await self.redis.exists(self.LOCK_KEY) > 0
    
    async def acquire_rebuild_lock(self) -> bool:
        """Acquire lock for index rebuild."""
        return await self.redis.set(
            self.LOCK_KEY, 
            "rebuilding",
            nx=True,
            ex=self.LOCK_TTL
        )
    
    async def release_rebuild_lock(self):
        """Release rebuild lock."""
        await self.redis.delete(self.LOCK_KEY)
    
    async def wait_for_rebuild(self, timeout: float = 30) -> bool:
        """Wait for rebuild to complete."""
        start = time.time()
        while await self.is_rebuilding():
            INDEX_REBUILD_LOCK_WAITS.inc()
            if time.time() - start > timeout:
                return False
            await asyncio.sleep(0.5)
        return True


# =============================================================================
# RAG Service Core
# =============================================================================

class RAGService:
    """
    RAG service with STPA safety controls.
    """
    
    def __init__(
        self,
        chroma_client: chromadb.HttpClient,
        embedding_model: SentenceTransformer,
        validator: EmbeddingValidator,
        rebuild_lock: IndexRebuildLock,
        config: Config
    ):
        self.chroma = chroma_client
        self.model = embedding_model
        self.validator = validator
        self.rebuild_lock = rebuild_lock
        self.config = config
        self.collection = None
    
    async def initialize(self):
        """Initialize collection."""
        try:
            self.collection = self.chroma.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.config.collection_name}' initialized")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        embedding = self.model.encode(text).tolist()
        return embedding
    
    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """
        Retrieve relevant documents with safety controls.
        
        STPA Safety Controls:
        - SC-UCA6.2: Embedding validation
        - SC-UCA6.3: Rebuild lock check
        - SC-UCA6.4: Query timeout
        - SC1: Relevance filtering
        """
        start_time = time.time()
        
        # Check for index rebuild (LS-1 mitigation)
        if await self.rebuild_lock.is_rebuilding():
            logger.warning("Index rebuild in progress, waiting...")
            if not await self.rebuild_lock.wait_for_rebuild(timeout=10):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Index rebuild in progress, please retry"
                )
        
        # Generate and validate embedding (SC-UCA6.2)
        try:
            embedding = self.generate_embedding(request.query)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate embedding"
            )
        
        embedding_valid = True
        if self.config.validate_embeddings:
            is_valid, reason = self.validator.validate(embedding)
            if not is_valid:
                logger.error(f"Embedding validation failed: {reason}")
                embedding_valid = False
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid embedding: {reason}"
                )
            # Normalize for consistent similarity computation
            embedding = self.validator.normalize(embedding)
        
        # Query with timeout (SC-UCA6.4)
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self.collection.query,
                    query_embeddings=[embedding],
                    n_results=request.top_k * 2,  # Over-fetch for filtering
                    include=["documents", "metadatas", "distances"]
                ),
                timeout=self.config.query_timeout
            )
        except asyncio.TimeoutError:
            RETRIEVAL_COUNT.labels(status="timeout").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Vector store query timeout"
            )
        except Exception as e:
            RETRIEVAL_COUNT.labels(status="error").inc()
            logger.error(f"ChromaDB query failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Vector store query failed"
            )
        
        # Process results with relevance filtering (SC1)
        min_relevance = request.min_relevance or self.config.min_relevance_score
        documents = []
        filtered_count = 0
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score
                # ChromaDB returns L2 distance for cosine space after normalization
                distance = results["distances"][0][i] if results["distances"] else 0
                relevance = 1 - (distance / 2)  # Convert to 0-1 range
                
                RELEVANCE_SCORE.observe(relevance)
                
                # Filter low relevance (SC1: context validation)
                if relevance < min_relevance:
                    filtered_count += 1
                    LOW_RELEVANCE_FILTERED.inc()
                    continue
                
                documents.append(Document(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    relevance_score=relevance
                ))
                
                # Stop at requested top_k
                if len(documents) >= request.top_k:
                    break
        
        processing_time = (time.time() - start_time) * 1000
        RETRIEVAL_LATENCY.observe(processing_time / 1000)
        RETRIEVAL_COUNT.labels(status="success").inc()
        
        return RetrieveResponse(
            query=request.query,
            documents=documents,
            total_retrieved=len(results["ids"][0]) if results["ids"] else 0,
            filtered_count=filtered_count,
            embedding_valid=embedding_valid,
            processing_time_ms=processing_time
        )
    
    async def health_check(self) -> dict:
        """Check service health including ChromaDB."""
        health = {
            "chromadb": "unknown",
            "collection_count": None
        }
        
        if self.config.health_check_chromadb:
            try:
                # Verify ChromaDB connectivity
                heartbeat = self.chroma.heartbeat()
                health["chromadb"] = "healthy"
                
                # Check collection
                if self.collection:
                    count = self.collection.count()
                    health["collection_count"] = count
                
                CHROMADB_HEALTH.set(1)
            except Exception as e:
                logger.error(f"ChromaDB health check failed: {e}")
                health["chromadb"] = "unhealthy"
                CHROMADB_HEALTH.set(0)
        
        return health


# =============================================================================
# Application Setup
# =============================================================================

rag_service: Optional[RAGService] = None
redis_client: Optional[redis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global rag_service, redis_client
    
    # Initialize Redis
    redis_client = redis.from_url(config.redis_url, decode_responses=True)
    
    # Initialize ChromaDB client
    chroma_client = chromadb.HttpClient(
        host=config.chromadb_host,
        port=config.chromadb_port,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Load embedding model
    logger.info(f"Loading embedding model: {config.model_name}")
    embedding_model = SentenceTransformer(config.model_name)
    
    # Initialize components
    validator = EmbeddingValidator(config.embedding_dim, config.min_embedding_norm)
    rebuild_lock = IndexRebuildLock(redis_client)
    
    # Create RAG service
    rag_service = RAGService(
        chroma_client=chroma_client,
        embedding_model=embedding_model,
        validator=validator,
        rebuild_lock=rebuild_lock,
        config=config
    )
    
    await rag_service.initialize()
    
    logger.info("RAG Service started with STPA safety controls")
    logger.info(f"  - Embedding validation: {config.validate_embeddings}")
    logger.info(f"  - Query timeout: {config.query_timeout}s")
    logger.info(f"  - Min relevance: {config.min_relevance_score}")
    
    yield
    
    await redis_client.close()


app = FastAPI(
    title="RAG Service",
    description="STPA-compliant RAG service with safety controls",
    lifespan=lifespan
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    chroma_health = await rag_service.health_check()
    
    try:
        await redis_client.ping()
        redis_status = "healthy"
    except Exception:
        redis_status = "unhealthy"
    
    all_healthy = (
        chroma_health["chromadb"] == "healthy" and 
        redis_status == "healthy"
    )
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        chromadb=chroma_health["chromadb"],
        redis=redis_status,
        model_loaded=rag_service.model is not None,
        collection_count=chroma_health.get("collection_count")
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """Retrieve relevant documents for a query."""
    return await rag_service.retrieve(request)


@app.post("/index")
async def add_documents(documents: list[dict]):
    """Add documents to the index."""
    # This would trigger rebuild lock in production
    # Simplified for tutorial
    collection = rag_service.collection
    
    ids = [doc["id"] for doc in documents]
    contents = [doc["content"] for doc in documents]
    metadatas = [doc.get("metadata", {}) for doc in documents]
    
    # Generate embeddings
    embeddings = [rag_service.generate_embedding(c) for c in contents]
    
    collection.add(
        ids=ids,
        documents=contents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    return {"status": "success", "count": len(documents)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
