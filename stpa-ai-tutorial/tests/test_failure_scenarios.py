#!/usr/bin/env python3
"""
STPA Failure Demonstration Script

This script demonstrates:
1. How the vulnerable configuration fails under load
2. How the resilient configuration handles the same conditions

Run with:
    python tests/test_failure_scenarios.py [vulnerable|resilient]
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TestConfig:
    api_url: str = "http://localhost:8000"
    concurrent_requests: int = 10
    total_requests: int = 50
    timeout: float = 30.0


# =============================================================================
# Test Cases
# =============================================================================

class TestResult:
    def __init__(self):
        self.successes = 0
        self.failures = 0
        self.timeouts = 0
        self.degraded = 0
        self.latencies = []
        self.errors = []
    
    def record_success(self, latency: float, degraded: bool = False):
        self.successes += 1
        self.latencies.append(latency)
        if degraded:
            self.degraded += 1
    
    def record_failure(self, error: str, latency: float):
        self.failures += 1
        self.latencies.append(latency)
        self.errors.append(error)
    
    def record_timeout(self):
        self.timeouts += 1
    
    def summary(self) -> dict:
        total = self.successes + self.failures + self.timeouts
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        p95_latency = sorted(self.latencies)[int(len(self.latencies) * 0.95)] if self.latencies else 0
        
        return {
            "total_requests": total,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "degraded_responses": self.degraded,
            "success_rate": f"{(self.successes / total * 100):.1f}%" if total > 0 else "N/A",
            "avg_latency_ms": f"{avg_latency:.1f}",
            "p95_latency_ms": f"{p95_latency:.1f}",
            "unique_errors": list(set(self.errors))
        }


async def make_request(
    client: httpx.AsyncClient,
    url: str,
    data: dict,
    result: TestResult,
    request_id: int
):
    """Make a single request and record the result."""
    start = time.time()
    
    try:
        response = await client.post(url, json=data, timeout=30.0)
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            body = response.json()
            degraded = body.get("degraded", False)
            result.record_success(latency, degraded)
            status = "✓ SUCCESS" if not degraded else "⚠ DEGRADED"
            print(f"  Request {request_id:3d}: {status} ({latency:.0f}ms)")
        elif response.status_code == 429:
            result.record_failure("rate_limited", latency)
            print(f"  Request {request_id:3d}: ⚡ RATE LIMITED ({latency:.0f}ms)")
        elif response.status_code == 504:
            result.record_failure("gateway_timeout", latency)
            print(f"  Request {request_id:3d}: ⏱ GATEWAY TIMEOUT ({latency:.0f}ms)")
        else:
            result.record_failure(f"http_{response.status_code}", latency)
            print(f"  Request {request_id:3d}: ✗ HTTP {response.status_code} ({latency:.0f}ms)")
    
    except httpx.TimeoutException:
        result.record_timeout()
        print(f"  Request {request_id:3d}: ⏱ CLIENT TIMEOUT")
    
    except Exception as e:
        latency = (time.time() - start) * 1000
        result.record_failure(str(type(e).__name__), latency)
        print(f"  Request {request_id:3d}: ✗ ERROR: {e}")


async def run_load_test(config: TestConfig, scenario: str) -> TestResult:
    """Run load test with concurrent requests."""
    result = TestResult()
    
    # Request data based on scenario
    if scenario == "weather_and_docs":
        data = {
            "query": "What's the weather in New York?",
            "include_weather": True,
            "include_documents": True
        }
    elif scenario == "docs_only":
        data = {
            "query": "Find documents about machine learning",
            "include_weather": False,
            "include_documents": True
        }
    else:
        data = {
            "query": "What's the weather?",
            "include_weather": True,
            "include_documents": False
        }
    
    async with httpx.AsyncClient() as client:
        # Send requests in batches
        for batch_start in range(0, config.total_requests, config.concurrent_requests):
            batch_end = min(batch_start + config.concurrent_requests, config.total_requests)
            batch_size = batch_end - batch_start
            
            print(f"\n  Batch {batch_start // config.concurrent_requests + 1}: "
                  f"Sending {batch_size} concurrent requests...")
            
            tasks = [
                make_request(
                    client,
                    f"{config.api_url}/query",
                    data,
                    result,
                    batch_start + i
                )
                for i in range(batch_size)
            ]
            
            await asyncio.gather(*tasks)
            
            # Small delay between batches
            await asyncio.sleep(0.5)
    
    return result


async def check_health(config: TestConfig) -> bool:
    """Check if the API is healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.api_url}/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


# =============================================================================
# Test Scenarios
# =============================================================================

async def test_scenario_ls3_cascading_timeout(config: TestConfig):
    """
    Test Scenario: LS-3 Cascading Timeout
    
    STPA Context:
    - Tests UCA9.4: Tool execution not terminated on timeout
    - Tests cascading failure when external API is slow
    
    Expected behavior:
    - Vulnerable: System hangs, all requests timeout
    - Resilient: Circuit breaker trips, fallback responses provided
    """
    print("\n" + "=" * 70)
    print("TEST SCENARIO: LS-3 Cascading Timeout")
    print("=" * 70)
    print("""
    This test simulates the external weather API being slow.
    
    In the VULNERABLE configuration:
    - No timeout on external calls
    - No circuit breaker
    - Requests will hang indefinitely
    - Connection pool will exhaust
    - System becomes unresponsive
    
    In the RESILIENT configuration:
    - 3-second timeout on external calls
    - Circuit breaker trips after 3 failures
    - Fallback responses provided
    - System remains responsive
    """)
    
    # Check health first
    if not await check_health(config):
        print("❌ API is not healthy. Please start the services first.")
        return None
    
    print("Running load test with weather requests...")
    result = await run_load_test(config, "weather_and_docs")
    
    return result


async def test_scenario_rate_limiting(config: TestConfig):
    """
    Test Scenario: Rate Limiting
    
    STPA Context:
    - Tests SC6: Request rate must be limited
    - Tests UCA3.1: No rate limiting allows DoS
    """
    print("\n" + "=" * 70)
    print("TEST SCENARIO: Rate Limiting (SC6)")
    print("=" * 70)
    print("""
    This test sends many requests quickly to test rate limiting.
    
    In the VULNERABLE configuration:
    - No rate limiting
    - All requests processed (until system overloads)
    
    In the RESILIENT configuration:
    - Rate limiting enabled (100 requests/minute)
    - Excess requests return 429 Too Many Requests
    """)
    
    if not await check_health(config):
        print("❌ API is not healthy. Please start the services first.")
        return None
    
    # Use more aggressive settings for rate limit test
    test_config = TestConfig(
        api_url=config.api_url,
        concurrent_requests=20,
        total_requests=150,
        timeout=10.0
    )
    
    print("Running load test (documents only to avoid external API)...")
    result = await run_load_test(test_config, "docs_only")
    
    return result


async def test_scenario_circuit_breaker(config: TestConfig):
    """
    Test Scenario: Circuit Breaker
    
    STPA Context:
    - Tests SC4: External API calls must have circuit breakers
    - Tests graceful degradation
    """
    print("\n" + "=" * 70)
    print("TEST SCENARIO: Circuit Breaker (SC4)")
    print("=" * 70)
    print("""
    This test checks circuit breaker behavior.
    
    The mock external API has a 10% failure rate configured.
    After 3 failures, the circuit breaker should open.
    Subsequent requests should get fallback responses.
    """)
    
    if not await check_health(config):
        print("❌ API is not healthy. Please start the services first.")
        return None
    
    print("Running weather requests to trigger circuit breaker...")
    result = await run_load_test(config, "weather_only")
    
    return result


# =============================================================================
# Main
# =============================================================================

def print_summary(result: TestResult, scenario_name: str):
    """Print test summary."""
    print("\n" + "-" * 70)
    print(f"RESULTS: {scenario_name}")
    print("-" * 70)
    
    summary = result.summary()
    for key, value in summary.items():
        if key == "unique_errors" and value:
            print(f"  {key}:")
            for error in value:
                print(f"    - {error}")
        else:
            print(f"  {key}: {value}")
    
    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print("-" * 70)
    
    success_rate = result.successes / (result.successes + result.failures + result.timeouts) * 100 if (result.successes + result.failures + result.timeouts) > 0 else 0
    
    if result.timeouts > 0:
        print("  ⚠ TIMEOUTS DETECTED: This indicates missing timeout configuration")
        print("    STPA Violation: UCA9.4 - Tool execution not terminated on timeout")
    
    if result.degraded > 0:
        print(f"  ✓ GRACEFUL DEGRADATION: {result.degraded} requests returned fallback responses")
        print("    STPA Control Working: SC4 - Circuit breaker and fallback enabled")
    
    if success_rate >= 90:
        print("  ✓ HIGH AVAILABILITY: System maintained >90% success rate")
    elif success_rate >= 50:
        print("  ⚠ DEGRADED: System maintained 50-90% success rate")
    else:
        print("  ✗ FAILURE: System had <50% success rate")
        print("    This indicates missing STPA safety controls")


async def main():
    parser = argparse.ArgumentParser(description="STPA Failure Demonstration")
    parser.add_argument(
        "mode",
        choices=["vulnerable", "resilient", "all"],
        help="Which configuration to test"
    )
    parser.add_argument(
        "--scenario",
        choices=["timeout", "rate", "circuit", "all"],
        default="all",
        help="Which scenario to test"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL"
    )
    
    args = parser.parse_args()
    
    config = TestConfig(api_url=args.url)
    
    print("\n" + "=" * 70)
    print("STPA FAILURE DEMONSTRATION")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"API URL: {config.api_url}")
    print(f"Scenario: {args.scenario}")
    
    scenarios = {
        "timeout": ("LS-3 Cascading Timeout", test_scenario_ls3_cascading_timeout),
        "rate": ("Rate Limiting", test_scenario_rate_limiting),
        "circuit": ("Circuit Breaker", test_scenario_circuit_breaker),
    }
    
    if args.scenario == "all":
        scenarios_to_run = scenarios.items()
    else:
        scenarios_to_run = [(args.scenario, scenarios[args.scenario])]
    
    for name, (description, test_func) in scenarios_to_run:
        result = await test_func(config)
        if result:
            print_summary(result, description)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
