"""Explicit authenticated access to the small public Halios API surface."""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from typing import Any

from ._transport import HaliosTransport
from ._version import __version__
from .exceptions import ConfigError, HaliosTimeoutError
from .types import EvaluateResult, EvaluationRun, TraceDetail


class HaliosClient:
    """Async client for inline checks and programmatic evaluation of existing traces.

    The client neither depends on nor configures OpenTelemetry. Applications using OpenTelemetry
    can pass the active W3C ``trace_id`` and ``parent_span_id`` so the server-created guardrail span
    joins the application trace.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.agent_id = agent_id or os.getenv("HALIOS_AGENT_ID")
        resolved_key = api_key or os.getenv("HALIOS_API_KEY")
        if not resolved_key:
            raise ConfigError("api_key is required; pass it directly or set HALIOS_API_KEY")
        self._transport = HaliosTransport(
            base_url=(base_url or os.getenv("HALIOS_BASE_URL") or "https://app.halios.ai"),
            api_key=resolved_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.base_url = self._transport.base_url

    def _resolve_agent(self, agent_id: str | None) -> str:
        resolved = agent_id or self.agent_id
        if not resolved:
            raise ConfigError("agent_id is required; pass it directly or set HALIOS_AGENT_ID")
        return resolved

    @staticmethod
    def _validate_trace_id(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) != 32 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ConfigError("trace_id must be 32 lowercase or uppercase hexadecimal characters")
        if normalized == "0" * 32:
            raise ConfigError("trace_id must not be the invalid all-zero W3C trace id")
        return normalized

    @staticmethod
    def _validate_span_id(value: str | None, *, field: str) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) != 16 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ConfigError(f"{field} must be 16 hexadecimal characters")
        if normalized == "0" * 16:
            raise ConfigError(f"{field} must not be the invalid all-zero W3C span id")
        return normalized

    async def evaluate(
        self,
        messages: list[dict[str, Any]],
        *,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        labels: list[str] | None = None,
        agent_id: str | None = None,
    ) -> EvaluateResult:
        """Evaluate conversation evidence synchronously against inline Halios checks."""
        if not messages:
            raise ConfigError("messages must contain at least one message")
        normalized_trace_id = self._validate_trace_id(trace_id)
        normalized_span_id = self._validate_span_id(span_id, field="span_id")
        normalized_parent_id = self._validate_span_id(parent_span_id, field="parent_span_id")
        if normalized_trace_id and not normalized_span_id:
            normalized_span_id = secrets.token_hex(8)
        payload: dict[str, Any] = {
            "agent_id": self._resolve_agent(agent_id),
            "messages": messages,
            "mode": "guardrail",
        }
        if normalized_trace_id:
            payload["trace_id"] = normalized_trace_id
        if normalized_span_id:
            payload["span_id"] = normalized_span_id
        if normalized_parent_id:
            payload["parent_span_id"] = normalized_parent_id
        if labels:
            payload["tags"] = labels
        response = await self._transport.request(
            "POST", f"{self._transport.api_prefix}/evaluate", json=payload
        )
        return EvaluateResult.model_validate(response.json())

    async def evaluate_request(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> EvaluateResult:
        """Evaluate input before executing the protected agent operation."""
        if not messages or messages[-1].get("role") != "user":
            raise ConfigError("evaluate_request messages must end with a user message")
        kwargs["labels"] = [*kwargs.get("labels", []), "direction:input"]
        return await self.evaluate(messages, **kwargs)

    async def evaluate_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> EvaluateResult:
        """Evaluate output after generation and before releasing it to the caller."""
        if not messages or messages[-1].get("role") != "assistant":
            raise ConfigError("evaluate_response messages must end with an assistant message")
        kwargs["labels"] = [*kwargs.get("labels", []), "direction:output"]
        return await self.evaluate(messages, **kwargs)

    async def get_trace(self, trace_id: str) -> TraceDetail:
        normalized = self._validate_trace_id(trace_id)
        response = await self._transport.request(
            "GET", f"{self._transport.api_prefix}/traces/{normalized}"
        )
        return TraceDetail.model_validate(response.json())

    async def evaluate_traces(
        self,
        trace_ids: list[str],
        *,
        run_name: str = "sdk-eval",
        labels: list[str] | None = None,
        fail_below: float | None = None,
        suite_revision: int | None = None,
        agent_id: str | None = None,
    ) -> EvaluationRun:
        """Create an immutable evaluation run over explicit existing W3C trace IDs."""
        if not trace_ids:
            raise ConfigError("trace_ids must contain at least one trace id")
        normalized = [self._validate_trace_id(trace_id) for trace_id in trace_ids]
        if len(set(normalized)) != len(normalized):
            raise ConfigError("trace_ids must be unique")
        payload: dict[str, Any] = {
            "agent_id": self._resolve_agent(agent_id),
            "run_name": run_name,
            "source": "sdk",
            "scenario_ids": [],
            "repetitions": 1,
            "trace_ids": normalized,
            "labels": labels or [],
            "gate": {"fail_below": fail_below} if fail_below is not None else {},
            "provenance": {"client_name": "haliosai-python", "client_version": __version__},
        }
        if suite_revision is not None:
            payload["suite_revision"] = suite_revision
        created = await self._transport.request(
            "POST", f"{self._transport.api_prefix}/runs/evaluations", json=payload
        )
        return await self.get_evaluation_run(str(created.json()["run_id"]))

    async def get_evaluation_run(self, run_id: str) -> EvaluationRun:
        response = await self._transport.request(
            "GET", f"{self._transport.api_prefix}/runs/evaluations/{run_id}"
        )
        return EvaluationRun.model_validate(response.json())

    async def wait_for_evaluation_run(
        self,
        run_id: str,
        *,
        timeout: float = 300,
        poll_interval: float = 2,
    ) -> EvaluationRun:
        """Wait for one immutable evaluation run to complete or fail."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            report = await self.get_evaluation_run(run_id)
            if report.status in {"completed", "failed"}:
                return report
            await asyncio.sleep(poll_interval)
        raise HaliosTimeoutError(f"evaluation run {run_id} did not finish within {timeout:g}s")

    async def close(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> "HaliosClient":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.close()
