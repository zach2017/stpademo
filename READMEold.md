# STPA Tutorial: AI System Safety Analysis

## System Theoretic Process Analysis for RAG + MCP Architecture

This tutorial demonstrates how to apply **System Theoretic Process Analysis (STPA)** to an AI system with:

- **RAG** (Retrieval Augmented Generation)
- **MCP** (Model Context Protocol)  
- **ChromaDB** (Vector Database)
- **External API Integration**

## 📚 Contents

```
stpa-ai-tutorial/
├── docs/
│   └── STPA_Tutorial.md       # Complete STPA tutorial document
├── src/
│   ├── api/                   # API Gateway service
│   ├── rag_service/           # RAG service with ChromaDB
│   └── mcp_service/           # MCP service with circuit breaker
├── tests/
│   └── test_failure_scenarios.py  # Failure demonstration script
├── config/
│   └── prometheus.yml         # Monitoring configuration
├── docker-compose.yml         # Resilient configuration
└── docker-compose.vulnerable.yml  # Vulnerable configuration
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for running tests)

### 1. Start the Resilient Configuration

```bash
# Build and start all services
docker-compose up --build

# Wait for all services to be healthy
docker-compose ps
```

### 2. Run the Failure Demonstration

```bash
# Install test dependencies
pip install httpx

# Run all test scenarios
python tests/test_failure_scenarios.py resilient --scenario all

# Or test specific scenarios
python tests/test_failure_scenarios.py resilient --scenario timeout
python tests/test_failure_scenarios.py resilient --scenario rate
python tests/test_failure_scenarios.py resilient --scenario circuit
```

### 3. Compare with Vulnerable Configuration

```bash
# Stop resilient services
docker-compose down

# Start vulnerable configuration
docker-compose -f docker-compose.vulnerable.yml up --build

# Run the same tests
python tests/test_failure_scenarios.py vulnerable --scenario all
```

## 🔬 STPA Analysis Summary

### Losses Identified

| ID | Loss | Impact |
|----|------|--------|
| L1 | Loss of response accuracy | Incorrect/hallucinated information |
| L2 | Loss of data confidentiality | Data breach |
| L3 | Loss of system availability | Service outage |
| L4 | Loss of response relevance | Poor user experience |
| L5 | Loss of data integrity | Corrupted embeddings |

### Hazards Identified

| ID | Hazard | Related Losses |
|----|--------|----------------|
| H1 | Information returned without context validation | L1, L4 |
| H2 | MCP tools executed with incorrect parameters | L1, L2 |
| H3 | Retrieval from stale/corrupted vector store | L1, L4, L5 |
| H4 | External API timeouts not handled | L3 |
| H5 | Internal tools exposed inappropriately | L2 |
| H6 | Queries processed without rate limiting | L3 |

### Safety Constraints Implemented

| ID | Constraint | Implementation |
|----|------------|----------------|
| SC1 | Context must be validated for relevance | MIN_RELEVANCE_SCORE filter |
| SC2 | Tool parameters must be sanitized | InputSanitizer class |
| SC3 | Vector store health must be verified | Health checks + rebuild lock |
| SC4 | External calls must have timeouts | Circuit breaker pattern |
| SC5 | Tool access must be scoped | AuthorizationValidator |
| SC6 | Requests must be rate limited | RateLimiter class |

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Rate Limiter │  │ Auth Guard  │  │ Request Timeout     │  │
│  │    (SC6)    │  │    (SC5)    │  │       (SC4)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────┐     ┌───────────────────────────────┐
│     RAG SERVICE       │     │        MCP SERVICE            │
│  ┌─────────────────┐  │     │  ┌─────────────────────────┐  │
│  │Embedding Valid. │  │     │  │  Circuit Breaker (SC4)  │  │
│  │   (SC-UCA6.2)   │  │     │  └─────────────────────────┘  │
│  └─────────────────┘  │     │  ┌─────────────────────────┐  │
│  ┌─────────────────┐  │     │  │ Input Sanitizer (SC2)   │  │
│  │ Rebuild Lock    │  │     │  └─────────────────────────┘  │
│  │   (LS-1)        │  │     │  ┌─────────────────────────┐  │
│  └─────────────────┘  │     │  │ Auth Validator (SC5)    │  │
│  ┌─────────────────┐  │     │  └─────────────────────────┘  │
│  │Relevance Filter │  │     └───────────────────────────────┘
│  │     (SC1)       │  │                   │
│  └─────────────────┘  │                   ▼
└───────────────────────┘     ┌───────────────────────────────┐
            │                 │      External API             │
            ▼                 │  (with fallback response)     │
┌───────────────────────┐     └───────────────────────────────┘
│      ChromaDB         │
│   (Vector Store)      │
└───────────────────────┘
```

## 🧪 Failure Scenarios

### LS-3: Cascading Timeout

**Scenario**: External weather API becomes slow.

**Vulnerable Behavior**:
1. MCP service waits indefinitely for response
2. Thread pool exhausted
3. Connection pool depleted
4. All services become unresponsive

**Resilient Behavior**:
1. 3-second timeout triggers
2. Circuit breaker records failure
3. After 3 failures, circuit opens
4. Subsequent requests get cached/fallback responses
5. System remains responsive

### Test Results Comparison

| Metric | Vulnerable | Resilient |
|--------|------------|-----------|
| Success Rate | ~10% | >90% |
| Avg Latency | >30s | <500ms |
| Timeouts | Many | None |
| Graceful Degradation | No | Yes |

## 🔧 Configuration Options

### API Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| REQUEST_TIMEOUT | 10 | Global request timeout (seconds) |
| RATE_LIMIT_ENABLED | true | Enable rate limiting |
| RATE_LIMIT_REQUESTS | 100 | Requests per window |
| RATE_LIMIT_WINDOW | 60 | Window size (seconds) |

### RAG Service

| Variable | Default | Description |
|----------|---------|-------------|
| VALIDATE_EMBEDDINGS | true | Validate embedding vectors |
| QUERY_TIMEOUT | 3 | ChromaDB query timeout |
| MIN_RELEVANCE_SCORE | 0.5 | Minimum context relevance |

### MCP Service

| Variable | Default | Description |
|----------|---------|-------------|
| EXTERNAL_CALL_TIMEOUT | 3 | External API timeout |
| CIRCUIT_BREAKER_ENABLED | true | Enable circuit breaker |
| CIRCUIT_BREAKER_THRESHOLD | 3 | Failures before trip |
| SANITIZE_INPUT | true | Enable input sanitization |

## 📈 Monitoring

Access monitoring dashboards:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

Key metrics to watch:

- `mcp_circuit_breaker_state` - Circuit breaker status
- `rag_embedding_validation_failures_total` - Embedding issues
- `api_gateway_rate_limit_hits_total` - Rate limit triggers
- `*_latency_seconds` - Service latencies

## 📖 Full Tutorial

See [docs/STPA_Tutorial.md](docs/STPA_Tutorial.md) for the complete STPA analysis including:

- Detailed control structure modeling
- Unsafe Control Action (UCA) analysis
- Loss scenario identification
- Mitigation strategies

## 🎓 Learning Resources

- [STPA Handbook](http://psas.scripts.mit.edu/home/get_file.php?name=STPA_handbook.pdf)
- [Engineering a Safer World](https://mitpress.mit.edu/books/engineering-safer-world) by Nancy Leveson
- [MCP Specification](https://modelcontextprotocol.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## 📝 License

This tutorial is provided for educational purposes. See individual component licenses for their respective terms.
