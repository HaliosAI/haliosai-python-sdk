"""HaliosOptimizable protocol — types and HTTP client.

Agents that wish to participate in the Halios optimization loop must expose:

  GET  /halios/config
       → HaliosOptimizableConfig (JSON)

  POST /halios/chat
       Body: {"conversation_id": str, "messages": [{role, content}], ...}
       Optional header: X-Halios-Prompt: <prompt override>
       → {"response": str, "conversation_id": str, ...}

  DELETE /halios/conversation/{conversation_id}
       → 204 No Content

Use ``mount_halios()`` from ``haliosai.optimize.server`` to mount these
routes onto an existing FastAPI app without writing them by hand.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel


class HaliosOptimizableConfig(BaseModel):
    """Shape returned by GET /halios/config."""

    current_prompt: str
    tool_schemas: list[dict[str, Any]] = []
    supported_scenarios: list[str] = []
    metadata: dict[str, Any] = {}


class HaliosOptimizableClient:
    """Async HTTP client for the HaliosOptimizable agent protocol.

    Usage::

        client = HaliosOptimizableClient("http://localhost:8100")
        config = await client.get_config()

        response = await client.chat(
            conversation_id="conv-1",
            messages=[{"role": "user", "content": "Hello"}],
            prompt_override="You are a helpful assistant.",
        )

        await client.delete_conversation("conv-1")
    """

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    async def get_config(self) -> HaliosOptimizableConfig:
        """Fetch the agent's current prompt and capabilities."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base}/halios/config")
            resp.raise_for_status()
            return HaliosOptimizableConfig(**resp.json())

    async def chat(
        self,
        conversation_id: str,
        messages: list[dict[str, str]],
        prompt_override: str | None = None,
    ) -> dict[str, Any]:
        """Send messages to the agent, optionally overriding the system prompt.

        Returns the raw JSON response dict from the agent. At minimum the
        response should contain ``{"response": str}``.

        Agents may also return ``trace_messages`` for the current turn. When
        present, this should be the backend-style interleaved message sequence
        for the turn, for example assistant tool calls, tool outputs, and the
        final assistant reply. The optimizer uses these messages to build
        trace payloads that preserve tool-usage evidence for evaluator checks.
        """
        headers: dict[str, str] = {}
        if prompt_override is not None:
            import base64
            headers["X-Halios-Prompt-B64"] = base64.b64encode(prompt_override.encode()).decode()
        body: dict[str, Any] = {
            "conversation_id": conversation_id,
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base}/halios/chat",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def delete_conversation(self, conversation_id: str) -> None:
        """Tear down a conversation, clearing any session state on the agent."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.delete(
                f"{self._base}/halios/conversation/{conversation_id}"
            )
            if resp.status_code not in (200, 204, 404):
                resp.raise_for_status()
