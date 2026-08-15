from __future__ import annotations

import pytest

from haliosai import Client, ConfigError
from haliosai.client import HaliosClient

EVALUATE_RESULT = {
    "triggered": False,
    "action": "allow",
    "violations": [],
    "check_results": [],
    "trace_id": "a" * 32,
    "span_id": "b" * 16,
    "latency_ms": 4,
}


@pytest.mark.asyncio
async def test_evaluate_request_adds_direction_and_w3c_context(client: HaliosClient) -> None:
    client._transport.responses["POST /api/v1/evaluate"] = EVALUATE_RESULT
    result = await client.evaluate_request(
        [{"role": "user", "content": "Can I return this value?"}],
        trace_id="1" * 32,
        parent_span_id="2" * 16,
        labels=["environment:production"],
    )

    payload = client._transport.calls[-1]["json"]
    assert result.action == "allow"
    assert payload["trace_id"] == "1" * 32
    assert len(payload["span_id"]) == 16
    assert payload["parent_span_id"] == "2" * 16
    assert payload["tags"] == ["environment:production", "direction:input"]


@pytest.mark.asyncio
async def test_evaluate_response_requires_assistant_message(client: HaliosClient) -> None:
    with pytest.raises(ConfigError, match="assistant"):
        await client.evaluate_response([{"role": "user", "content": "not a response"}])


@pytest.mark.asyncio
async def test_evaluate_rejects_invalid_w3c_ids(client: HaliosClient) -> None:
    with pytest.raises(ConfigError, match="all-zero"):
        await client.evaluate(
            [{"role": "user", "content": "hello"}],
            trace_id="0" * 32,
        )


def test_default_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HALIOS_BASE_URL", raising=False)
    client = Client(agent_id="agent", api_key="key")
    assert client.base_url == "https://app.halios.ai"
