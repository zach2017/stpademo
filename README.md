

# System Theoretic Process Analysis (STPA) for AI Systems


STPA is based on the System-Theoretic Accident Model and Process (STAMP), which sees accidents as emergent properties of flawed control.

## A Comprehensive Tutorial for RAG + MCP Architecture with ChromaDB



## Table of Contents

1. [Introduction to STPA](#1-introduction-to-stpa)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [STPA Step 1: Define Purpose of Analysis](#3-stpa-step-1-define-purpose-of-analysis)
4. [STPA Step 2: Model the Control Structure](#4-stpa-step-2-model-the-control-structure)
5. [STPA Step 3: Identify Unsafe Control Actions](#5-stpa-step-3-identify-unsafe-control-actions)
6. [STPA Step 4: Identify Loss Scenarios](#6-stpa-step-4-identify-loss-scenarios)
7. [Failure Example and Analysis](#7-failure-example-and-analysis)
8. [Solutions and Mitigations](#8-solutions-and-mitigations)
9. [Implementation Guide](#9-implementation-guide)

---

## 1. Introduction to STPA

System Theoretic Process Analysis (STPA) is a hazard analysis technique based on systems thinking and control theory, developed by Nancy Leveson at MIT. Unlike traditional safety analysis methods (FMEA, FTA) that focus on component failures, STPA treats safety as a control problem.

### Why STPA for AI Systems?

Traditional safety analysis assumes failures cause accidents. However, in complex AI systems:

- Components may work exactly as designed but still cause harm
- Emergent behaviors arise from component interactions
- Software doesn't "fail" in the traditional sense—it does exactly what it's programmed to do
- AI systems have non-deterministic behaviors that traditional methods can't capture

### Core STPA Concepts

**Losses (L)**: Something of value to stakeholders that we want to prevent (e.g., incorrect information, data breach, system unavailability)

**Hazards (H)**: System states or conditions that, combined with environmental conditions, lead to losses

**Constraints (SC)**: Behaviors the system must exhibit to prevent hazards

**Control Actions (CA)**: Actions taken by controllers to affect the controlled process

**Unsafe Control Actions (UCA)**: Control actions that lead to hazards in specific contexts

---

## 2. System Architecture Overview

Our target system is an AI assistant with:

- **RAG (Retrieval Augmented Generation)**: Enhances LLM responses with retrieved context
- **MCP (Model Context Protocol)**: Standardized protocol for tool integration
- **ChromaDB**: Vector database for semantic search
- **External API**: Third-party service integration

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY (FastAPI)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │Rate Limiter │  │ Auth Guard  │  │ Request Validator       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌───────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│   RAG SERVICE     │ │   MCP SERVICE   │ │  EXTERNAL API       │
│  ┌─────────────┐  │ │ ┌─────────────┐ │ │  (Weather, etc.)    │
│  │  Embedder   │  │ │ │Tool Router  │ │ └─────────────────────┘
│  └─────────────┘  │ │ └─────────────┘ │
│  ┌─────────────┐  │ │ ┌─────────────┐ │
│  │  Retriever  │  │ │ │Tool Executor│ │
│  └─────────────┘  │ │ └─────────────┘ │
│  ┌─────────────┐  │ └─────────────────┘
│  │  Generator  │  │
│  └─────────────┘  │
└───────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CHROMADB                                │
│              (Vector Database for Embeddings)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. STPA Step 1: Define Purpose of Analysis

### 3.1 Identify Losses

| ID | Loss Description | Impact |
|----|------------------|--------|
| L1 | Loss of response accuracy | User receives incorrect/hallucinated information |
| L2 | Loss of data confidentiality | Sensitive data exposed to unauthorized parties |
| L3 | Loss of system availability | Users cannot access the AI assistant |
| L4 | Loss of response relevance | Retrieved context doesn't match user intent |
| L5 | Loss of data integrity | Stored embeddings become corrupted or stale |

### 3.2 Identify System-Level Hazards

| ID | Hazard | Related Losses |
|----|--------|----------------|
| H1 | System returns information without proper context validation | L1, L4 |
| H2 | System executes MCP tools with incorrect parameters | L1, L2 |
| H3 | System retrieves from stale or corrupted vector store | L1, L4, L5 |
| H4 | System fails to handle external API timeouts gracefully | L3 |
| H5 | System exposes internal tool capabilities inappropriately | L2 |
| H6 | System processes queries without rate limiting | L3 |

### 3.3 Define System-Level Constraints

| ID | Safety Constraint | Enforces |
|----|-------------------|----------|
| SC1 | All retrieved context must be validated for relevance before generation | H1 |
| SC2 | MCP tool parameters must be sanitized and validated against schema | H2 |
| SC3 | Vector store health must be verified before retrieval operations | H3 |
| SC4 | External API calls must have timeouts and circuit breakers | H4 |
| SC5 | Tool availability must be scoped to user authorization level | H5 |
| SC6 | Request rate must be limited per user and globally | H6 |

---

## 4. STPA Step 2: Model the Control Structure

### 4.1 Hierarchical Control Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LEVEL 0: HUMAN OPERATORS                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      System Administrator                        │    │
│  │  Control Actions: Configure, Deploy, Monitor, Update             │    │
│  │  Feedback: System metrics, logs, alerts                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    Control Actions │ │ Feedback
                                    ▼ │
┌─────────────────────────────────────────────────────────────────────────┐
│                      LEVEL 1: ORCHESTRATION LAYER                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      API Gateway Controller                      │    │
│  │  Control Actions:                                                │    │
│  │    CA1: Route request to RAG service                            │    │
│  │    CA2: Route request to MCP service                            │    │
│  │    CA3: Apply rate limiting                                     │    │
│  │    CA4: Validate authentication                                 │    │
│  │  Process Model:                                                 │    │
│  │    - Current request count per user                             │    │
│  │    - Authentication state                                       │    │
│  │    - Service health status                                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    Control Actions │ │ Feedback
                                    ▼ │
┌─────────────────────────────────────────────────────────────────────────┐
│                       LEVEL 2: SERVICE LAYER                            │
│                                                                         │
│  ┌──────────────────────────┐    ┌──────────────────────────┐          │
│  │    RAG Controller        │    │    MCP Controller        │          │
│  │  Control Actions:        │    │  Control Actions:        │          │
│  │    CA5: Generate embed   │    │    CA8: Select tool      │          │
│  │    CA6: Query ChromaDB   │    │    CA9: Execute tool     │          │
│  │    CA7: Generate response│    │    CA10: Return result   │          │
│  │  Process Model:          │    │  Process Model:          │          │
│  │    - Query embedding     │    │    - Available tools     │          │
│  │    - Retrieved contexts  │    │    - Tool schemas        │          │
│  │    - Relevance scores    │    │    - Execution state     │          │
│  └──────────────────────────┘    └──────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    Control Actions │ │ Feedback
                                    ▼ │
┌─────────────────────────────────────────────────────────────────────────┐
│                    LEVEL 3: DATA/EXTERNAL LAYER                         │
│                                                                         │
│  ┌──────────────────────────┐    ┌──────────────────────────┐          │
│  │       ChromaDB           │    │     External APIs        │          │
│  │  (Controlled Process)    │    │  (Controlled Process)    │          │
│  │                          │    │                          │          │
│  │  State Variables:        │    │  State Variables:        │          │
│  │    - Collection health   │    │    - API availability    │          │
│  │    - Index status        │    │    - Rate limit status   │          │
│  │    - Document count      │    │    - Response latency    │          │
│  └──────────────────────────┘    └──────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Control Actions Inventory

| Controller | CA ID | Control Action | Target |
|------------|-------|----------------|--------|
| API Gateway | CA1 | Route to RAG | RAG Service |
| API Gateway | CA2 | Route to MCP | MCP Service |
| API Gateway | CA3 | Apply rate limit | Request flow |
| API Gateway | CA4 | Validate auth | User session |
| RAG Service | CA5 | Generate embedding | Embedding model |
| RAG Service | CA6 | Query vector store | ChromaDB |
| RAG Service | CA7 | Generate response | LLM |
| MCP Service | CA8 | Select tool | Tool router |
| MCP Service | CA9 | Execute tool | External API |
| MCP Service | CA10 | Return result | API Gateway |

---

## 5. STPA Step 3: Identify Unsafe Control Actions

For each control action, we analyze four types of unsafe behavior:

1. **Not providing** the control action leads to hazard
2. **Providing** the control action leads to hazard  
3. **Providing too early, too late, or out of sequence**
4. **Stopped too soon or applied too long**

### 5.1 UCA Analysis Table

| CA | Not Providing | Providing Causes Hazard | Wrong Timing | Duration |
|----|---------------|------------------------|--------------|----------|
| **CA1: Route to RAG** | UCA1.1: Query requiring context goes to direct LLM, causing hallucination [H1] | UCA1.2: Non-contextual query routed to RAG wastes resources [H6] | UCA1.3: RAG routing delayed causes timeout [H4] | N/A |
| **CA3: Apply rate limit** | UCA3.1: No rate limiting allows DoS [H6] | UCA3.2: Rate limit on critical health checks blocks monitoring [H4] | UCA3.3: Rate limit applied after request processed [H6] | UCA3.4: Rate limit window too short causes false positives [H3] |
| **CA6: Query vector store** | UCA6.1: Not querying returns no context [H1] | UCA6.2: Query with malformed embedding returns wrong results [H1,H4] | UCA6.3: Query during index rebuild returns stale data [H3] | UCA6.4: Query timeout too long blocks service [H4] |
| **CA7: Generate response** | UCA7.1: No response generated for valid query [H4] | UCA7.2: Response generated with low-relevance context [H1] | UCA7.3: Response generated before retrieval completes [H1] | UCA7.4: Generation continues despite context invalidation [H1] |
| **CA9: Execute tool** | UCA9.1: Tool not executed when required [H4] | UCA9.2: Tool executed with unsanitized input [H2,H5] | UCA9.3: Tool executed before auth verification [H5] | UCA9.4: Tool execution not terminated on timeout [H4] |

### 5.2 Critical UCA Deep Dive

#### UCA6.2: Query ChromaDB with Malformed Embedding

**Context**: RAG service sends embedding query to ChromaDB

**Hazardous Scenario**: 
- Embedding model returns zero vector or NaN values
- ChromaDB query proceeds without validation
- Similarity search returns random documents
- Generator produces response with irrelevant context

**Safety Constraint**: SC-UCA6.2: Embedding vectors must be validated for dimensionality, non-zero norm, and finite values before querying

#### UCA9.2: Execute MCP Tool with Unsanitized Input

**Context**: MCP service executes external tool based on user request

**Hazardous Scenario**:
- User request contains injection payload
- Tool parameters passed without sanitization
- External API executes unintended operations
- Data exfiltration or unauthorized actions occur

**Safety Constraint**: SC-UCA9.2: All tool parameters must be validated against JSON schema and sanitized for injection patterns

---

## 6. STPA Step 4: Identify Loss Scenarios

For each UCA, we identify why it might occur by examining:

1. **Controller failures**: Why the controller might give unsafe commands
2. **Feedback failures**: Why the controller might have wrong information
3. **Control path failures**: Why commands might not execute correctly

### 6.1 Loss Scenario Analysis

#### Scenario LS-1: Stale Context Retrieval (UCA6.3)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOSS SCENARIO LS-1                           │
│              Stale Context During Index Rebuild                 │
└─────────────────────────────────────────────────────────────────┘

SEQUENCE OF EVENTS:
1. Background job triggers ChromaDB index rebuild
2. User submits query during rebuild window
3. RAG service queries ChromaDB (unaware of rebuild)
4. ChromaDB returns results from partially updated index
5. Some documents missing, relevance scores incorrect
6. Generator produces response with incomplete context
7. User receives inaccurate information

CAUSAL FACTORS:
┌─────────────────────────────────────────────────────────────────┐
│ Controller (RAG Service)                                        │
│   └─► Process model doesn't include ChromaDB rebuild state      │
│   └─► No mechanism to detect partial index state                │
├─────────────────────────────────────────────────────────────────┤
│ Feedback Path                                                   │
│   └─► ChromaDB doesn't signal rebuild-in-progress               │
│   └─► Health check only verifies connectivity, not consistency  │
├─────────────────────────────────────────────────────────────────┤
│ Controlled Process (ChromaDB)                                   │
│   └─► Index rebuild is non-atomic                               │
│   └─► Queries allowed during rebuild by default                 │
└─────────────────────────────────────────────────────────────────┘

MITIGATIONS:
M1: Implement read/write locking during rebuild
M2: Add rebuild state to health check response
M3: Queue queries during rebuild window
M4: Use blue-green deployment for index updates
```

#### Scenario LS-2: Tool Execution Without Authorization Check (UCA9.3)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOSS SCENARIO LS-2                           │
│            Tool Execution Bypasses Authorization                │
└─────────────────────────────────────────────────────────────────┘

SEQUENCE OF EVENTS:
1. User sends request with valid session token
2. API Gateway validates token and forwards request
3. Token expires during processing
4. MCP service receives tool execution request
5. MCP service doesn't re-validate authorization
6. Tool executes with elevated privileges from cached state
7. Unauthorized data access occurs

CAUSAL FACTORS:
┌─────────────────────────────────────────────────────────────────┐
│ Controller (MCP Service)                                        │
│   └─► Assumes upstream auth is sufficient                       │
│   └─► No local authorization check before execution             │
│   └─► Cached user permissions not refreshed                     │
├─────────────────────────────────────────────────────────────────┤
│ Control Path                                                    │
│   └─► Auth state not propagated with request                    │
│   └─► No token refresh mechanism in service chain               │
├─────────────────────────────────────────────────────────────────┤
│ Controlled Process (External API)                               │
│   └─► API trusts service-level authentication                   │
│   └─► No user-level authorization at API                        │
└─────────────────────────────────────────────────────────────────┘

MITIGATIONS:
M1: Pass token with each inter-service request
M2: Implement token validation at MCP service
M3: Use short-lived, request-scoped tokens
M4: Add user context to external API calls
```

#### Scenario LS-3: Cascading Timeout Failure (UCA6.4 + UCA9.4)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOSS SCENARIO LS-3                           │
│              Cascading Timeout Causes System Hang               │
└─────────────────────────────────────────────────────────────────┘

SEQUENCE OF EVENTS:
1. External weather API experiences latency spike
2. MCP tool call waits indefinitely (no timeout)
3. Request thread blocked, connection held
4. Subsequent requests queue up
5. ChromaDB connection pool exhausted
6. RAG service cannot query vector store
7. All services appear healthy but unresponsive
8. System effectively unavailable

CAUSAL FACTORS:
┌─────────────────────────────────────────────────────────────────┐
│ Design Flaw                                                     │
│   └─► No timeout configuration on external calls                │
│   └─► Shared connection pool across services                    │
│   └─► No circuit breaker pattern implemented                    │
├─────────────────────────────────────────────────────────────────┤
│ Feedback Failure                                                │
│   └─► Health checks pass (services running)                     │
│   └─► Latency metrics not monitored                             │
│   └─► Connection pool exhaustion not detected                   │
├─────────────────────────────────────────────────────────────────┤
│ Inadequate Control                                              │
│   └─► No backpressure mechanism                                 │
│   └─► No request prioritization                                 │
│   └─► No graceful degradation path                              │
└─────────────────────────────────────────────────────────────────┘

MITIGATIONS:
M1: Configure timeouts at every boundary
M2: Implement circuit breaker with fallback
M3: Isolate connection pools per service
M4: Add latency-based health checks
M5: Implement request shedding under load
```

---

## 7. Failure Example and Analysis

We'll now implement a concrete failure scenario demonstrating LS-3 (Cascading Timeout) and analyze it using STPA principles.

### 7.1 The Failure Scenario

**Setup**: User asks "What's the weather in New York and find related documents"

**Expected Behavior**:
1. API Gateway routes to both MCP (weather) and RAG (documents)
2. Both services respond within 5 seconds
3. Results aggregated and returned

**Actual Behavior** (with bug):
1. External weather API is slow (simulated)
2. MCP service waits indefinitely
3. ChromaDB connections exhausted
4. RAG service times out
5. User receives error or no response

### 7.2 STPA Analysis of Failure

**Root Cause Identification using STPA**:

| STPA Element | Finding |
|--------------|---------|
| Control Action | CA9 (Execute tool) provided without timeout |
| UCA | UCA9.4: Tool execution not terminated on timeout |
| Loss Scenario | LS-3: Cascading timeout |
| Violated Constraint | SC4: External API calls must have timeouts |
| Hazard | H4: System fails to handle external API timeouts |
| Loss | L3: Loss of system availability |

### 7.3 Control Structure Gap Analysis

```
BEFORE (Vulnerable):
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   MCP Service  │────►│ External API   │     │   No timeout   │
│                │     │ (Weather)      │     │   No fallback  │
│  Waits forever │◄────│ Slow response  │     │   No circuit   │
└────────────────┘     └────────────────┘     │   breaker      │
        │                                      └────────────────┘
        ▼
┌────────────────┐
│ Blocks thread  │
│ Exhausts pool  │
│ Cascades fail  │
└────────────────┘

AFTER (Resilient):
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   MCP Service  │────►│ Circuit Breaker│────►│ External API   │
│                │     │                │     │ (Weather)      │
│  3s timeout    │◄────│ Fallback path  │◄────│                │
└────────────────┘     └────────────────┘     └────────────────┘
        │                     │
        ▼                     ▼
┌────────────────┐     ┌────────────────┐
│ Graceful       │     │ Returns cached │
│ degradation    │     │ or default     │
└────────────────┘     └────────────────┘
```

---

## 8. Solutions and Mitigations

Based on STPA analysis, here are the implemented solutions:

### 8.1 Solution Matrix

| UCA | Mitigation | Implementation |
|-----|------------|----------------|
| UCA6.2 | Embedding validation | `validate_embedding()` function |
| UCA6.3 | Index rebuild lock | Redis-based read/write lock |
| UCA6.4 | Query timeout | 3-second timeout with fallback |
| UCA9.2 | Input sanitization | JSON schema validation |
| UCA9.3 | Re-authorization | Token validation at MCP |
| UCA9.4 | Circuit breaker | `circuitbreaker` library |

### 8.2 Defense in Depth Layers

```
Layer 1: Input Validation
├── Request schema validation
├── Parameter sanitization  
└── Rate limiting

Layer 2: Processing Safety
├── Timeout on all operations
├── Circuit breakers on external calls
├── Fallback responses
└── Embedding validation

Layer 3: Data Integrity
├── Vector store health checks
├── Index rebuild coordination
└── Stale data detection

Layer 4: Output Safety
├── Response relevance scoring
├── Confidence thresholds
└── Source attribution

Layer 5: Monitoring
├── Latency percentiles
├── Error rate tracking
├── Circuit breaker state
└── Connection pool metrics
```

### 8.3 Key Code Patterns

**Pattern 1: Circuit Breaker**
```python
from circuitbreaker import circuit

@circuit(failure_threshold=3, recovery_timeout=30)
async def call_external_api(url: str, params: dict):
    async with httpx.AsyncClient(timeout=3.0) as client:
        return await client.get(url, params=params)
```

**Pattern 2: Embedding Validation**
```python
def validate_embedding(embedding: list[float]) -> bool:
    if len(embedding) != EXPECTED_DIM:
        return False
    if not all(math.isfinite(x) for x in embedding):
        return False
    norm = math.sqrt(sum(x*x for x in embedding))
    if norm < 1e-10:  # Near-zero vector
        return False
    return True
```

**Pattern 3: Graceful Degradation**
```python
async def get_weather_with_fallback(location: str):
    try:
        return await call_external_api(WEATHER_URL, {"q": location})
    except CircuitBreakerError:
        return {"status": "degraded", "message": "Weather unavailable"}
    except TimeoutError:
        return {"status": "timeout", "cached": get_cached_weather(location)}
```

---

## 9. Implementation Guide

See the accompanying Docker Compose setup and source code for a complete working example demonstrating:

1. **Vulnerable configuration** (`docker-compose.vulnerable.yml`)
   - No timeouts
   - No circuit breakers
   - Shared connection pools
   
2. **Resilient configuration** (`docker-compose.yml`)
   - Proper timeouts at all boundaries
   - Circuit breakers on external calls
   - Isolated resources
   - Health checks with latency monitoring

### Running the Demo

```bash
# Start vulnerable version (demonstrates failure)
docker-compose -f docker-compose.vulnerable.yml up

# Trigger failure
curl http://localhost:8000/query -d '{"query": "weather in NYC and related docs"}'

# Start resilient version (demonstrates recovery)
docker-compose -f docker-compose.yml up

# Same query now handles gracefully
curl http://localhost:8000/query -d '{"query": "weather in NYC and related docs"}'
```

---

## Appendix A: STPA Checklist for AI Systems

- [ ] Identify all losses relevant to stakeholders
- [ ] Map losses to system-level hazards
- [ ] Define safety constraints for each hazard
- [ ] Model hierarchical control structure
- [ ] Enumerate all control actions
- [ ] Analyze UCAs using 4-column method
- [ ] Identify loss scenarios for critical UCAs
- [ ] Design mitigations for each scenario
- [ ] Implement defense in depth
- [ ] Validate with failure injection testing

## Appendix B: References

1. Leveson, N. (2011). Engineering a Safer World. MIT Press.
2. STPA Handbook. MIT Partnership for Systems Approaches to Safety and Security.
3. Anthropic MCP Specification. https://modelcontextprotocol.io/
4. ChromaDB Documentation. https://docs.trychroma.com/

---
