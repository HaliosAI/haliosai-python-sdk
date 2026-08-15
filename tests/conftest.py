from __future__ import annotations

from typing import Any

import httpx
import pytest

from haliosai._transport import HaliosTransport
from haliosai.client import HaliosClient


class MockTransport(HaliosTransport):
    def __init__(self) -> None:
        super().__init__(base_url="https://app.halios.ai", api_key="test-key")
        self.calls: list[dict[str, Any]] = []
        self.responses: dict[str, dict[str, Any]] = {}

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        self.calls.append({"method": method, "path": path, "json": json, "params": params})
        body = self.responses[f"{method} {path}"]
        return httpx.Response(
            200,
            json=body,
            request=httpx.Request(method, f"https://app.halios.ai{path}"),
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def client() -> HaliosClient:
    instance = HaliosClient(agent_id="support-agent", api_key="test-key")
    instance._transport = MockTransport()
    return instance
