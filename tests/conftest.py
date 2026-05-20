"""Shared test fixtures for HaliosAI SDK tests."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from haliosai._transport import HaliosTransport
from haliosai.client import HaliosClient

# ---------------------------------------------------------------------------
# Mock transport
# ---------------------------------------------------------------------------


class MockTransport(HaliosTransport):
    """Transport that returns pre-configured responses without HTTP calls."""

    def __init__(self, responses: dict[str, Any] | None = None):
        super().__init__(
            base_url="https://api.test.halios.ai",
            api_key="test-key-123",
        )
        self._responses: dict[str, Any] = responses or {}
        self._calls: list[dict[str, Any]] = []

    def set_response(self, method_path: str, body: dict, status: int = 200):
        """Register a response for a method+path key like ``POST /api/v1/evaluate``."""
        self._responses[method_path] = body if callable(body) else (status, body)

    def _make_response(
        self,
        method: str,
        path: str,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> httpx.Response:
        key = f"{method} {path}"
        # 1) Exact match first
        if key in self._responses:
            response = self._responses[key]
            status, body = (
                response(method, path, json_body, params) if callable(response) else response
            )
            return httpx.Response(
                status_code=status,
                json=body,
                request=httpx.Request(method, f"https://api.test.halios.ai{path}"),
            )
        # 2) Prefix match restricted to same method
        best_match = None
        best_len = 0
        for k, v in self._responses.items():
            parts = k.split(" ", 1)
            if len(parts) == 2:
                k_method, k_path = parts
            else:
                k_method, k_path = "", parts[0]
            if k_method == method and path.startswith(k_path) and len(k_path) > best_len:
                best_match = v
                best_len = len(k_path)
        if best_match is not None:
            status, body = (
                best_match(method, path, json_body, params) if callable(best_match) else best_match
            )
            return httpx.Response(
                status_code=status,
                json=body,
                request=httpx.Request(method, f"https://api.test.halios.ai{path}"),
            )
        # Default: return empty 200
        return httpx.Response(
            status_code=200,
            json={},
            request=httpx.Request(method, f"https://api.test.halios.ai{path}"),
        )

    async def request(self, method: str, path: str, *, json: dict | None = None, params: dict | None = None) -> httpx.Response:
        self._calls.append({"method": method, "path": path, "json": json, "params": params})
        return self._make_response(method, path, json, params)

    def request_sync(self, method: str, path: str, *, json: dict | None = None, params: dict | None = None) -> httpx.Response:
        self._calls.append({"method": method, "path": path, "json": json, "params": params})
        return self._make_response(method, path, json, params)

    async def aclose(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Canned response data
# ---------------------------------------------------------------------------

EVALUATE_PASS = {
    "triggered": False,
    "action": "allow",
    "violations": [],
    "check_results": [
        {
            "check_id": "chk-1",
            "check_name": "toxicity",
            "task_name": "Safety",
            "task_slug": "safety",
            "validation_rule_id": "vr-1",
            "validation_rule_name": "toxicity_check",
            "triggered": False,
            "score": 0.05,
            "passed": True,
            "result": {},
            "reasoning": "Content is clean",
            "latency_ms": 120,
        }
    ],
    "trace_id": "abc123",
    "span_id": "span-001",
    "latency_ms": 150,
}

EVALUATE_BLOCK = {
    "triggered": True,
    "action": "block",
    "violations": [
        {
            "check_id": "chk-2",
            "check_name": "pii_detection",
            "validation_rule_id": "vr-2",
            "validation_rule_name": "pii_check",
            "message": "PII detected in content",
            "severity": "high",
        }
    ],
    "check_results": [
        {
            "check_id": "chk-2",
            "check_name": "pii_detection",
            "validation_rule_id": "vr-2",
            "validation_rule_name": "pii_check",
            "triggered": True,
            "score": 0.95,
            "passed": False,
            "result": {},
            "reasoning": "PII detected",
            "latency_ms": 80,
        }
    ],
    "trace_id": "abc124",
    "span_id": "span-002",
    "latency_ms": 100,
}

TRACE_RESPONSE = {
    "trace_id": "trace-001",
    "agent_id": "agent-uuid-1",
    "agent_name": "Test Agent",
    "status": "active",
    "tags": ["test"],
    "metrics": {},
    "trace_stats": {},
    "started_at": "2024-01-01T00:00:00",
    "finalized_at": None,
    "created_at": "2024-01-01T00:00:00",
}

TRACE_DETAIL = {
    **TRACE_RESPONSE,
    "spans": [
        {
            "span_id": "span-001",
            "trace_id": "trace-001",
            "parent_span_id": None,
            "name": "llm_call",
            "kind": "client",
            "status": "ok",
            "input": {"role": "user", "content": "hello"},
            "output": {"role": "assistant", "content": "hi"},
            "attributes": {},
            "started_at": "2024-01-01T00:00:00",
            "ended_at": "2024-01-01T00:00:01",
        }
    ],
}

TRIGGER_EVAL_RESPONSE = {
    "task_id": "task-abc",
    "run_tag": "run:test-abc",
    "status": "completed",
    "trace_count": 5,
    "check_count": 10,
}

CHECK_EXECUTIONS_RESPONSE = {
    "data": [
        {
            "id": "1",
            "trace_id": "trace-001",
            "span_id": "span-001",
            "check_id": "chk-1",
            "check_name": "toxicity",
            "task_name": "Safety",
            "task_slug": "safety",
            "validation_rule_id": "vr-1",
            "validation_rule_name": "toxicity_check",
            "mode": "evaluator",
            "scope": "single",
            "tags": ["nightly"],
            "triggered": False,
            "score": 0.05,
            "passed": True,
            "result": {},
            "reasoning": "Clean",
            "latency_ms": 100,
            "tokens_used": 50,
            "error": None,
            "created_at": "2024-01-01T00:00:00",
        }
    ],
    "next_cursor": None,
    "has_more": False,
    "progress": {
        "total": 1,
        "completed": 1,
        "pending": 0,
    },
}



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transport():
    """Create a MockTransport with common responses pre-loaded."""
    transport = MockTransport()
    transport.set_response("POST /api/v1/evaluate", EVALUATE_PASS)
    transport.set_response("POST /api/v1/traces", TRACE_RESPONSE)
    transport.set_response("GET /api/v1/traces/", TRACE_DETAIL)
    transport.set_response("POST /api/v1/traces/", TRACE_RESPONSE)
    transport.set_response("POST /api/v1/trigger-eval", TRIGGER_EVAL_RESPONSE)
    # Agent-scoped check-executions endpoint
    transport.set_response("GET /api/v1/agents/", CHECK_EXECUTIONS_RESPONSE)
    return transport


@pytest.fixture
def mock_client(mock_transport):
    """Create a HaliosClient with mock transport."""
    client = HaliosClient.__new__(HaliosClient)
    client.agent_id = "test-agent"
    client.api_key = "test-key-123"
    client.base_url = "https://api.test.halios.ai"
    client.timeout = 30.0
    client.max_retries = 3
    client._transport = mock_transport
    return client
