"""Tracing module — create, finalize, and manage traces and spans."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ._transport import HaliosTransport
from .types import TraceDetail, TraceResponse

logger = logging.getLogger("haliosai")


# ---------------------------------------------------------------------------
# Low-level trace API wrappers
# ---------------------------------------------------------------------------


async def _create_trace(
    transport: HaliosTransport,
    *,
    agent_id: str,
    tags: list[str] | None = None,
    trace_id: str | None = None,
) -> TraceResponse:
    """Create a new trace via ``POST /api/<version>/traces``.

    The effective API version (``/api/v1`` or ``/api/v2``) is determined at
    runtime from the client's configured ``base_url``. See ``transport.api_prefix``.

    Args:
        transport: HTTP transport instance.
        agent_id: Agent UUID or slug.
        tags: Optional tags to attach to the trace.
        trace_id: Optional explicit trace ID (hex32). Auto-generated if omitted.

    Returns:
        :class:`~haliosai.types.TraceResponse`
    """
    tid = trace_id or uuid.uuid4().hex
    payload: dict[str, Any] = {
        "trace_id": tid,
        "agent_id": agent_id,
    }
    if tags:
        payload["tags"] = tags

    resp = await transport.request("POST", f"{transport.api_prefix}/traces", json=payload)
    return TraceResponse.model_validate(resp.json())


async def _finalize_trace(
    transport: HaliosTransport,
    *,
    trace_id: str,
    trigger_evaluators: bool = False,
    tags: list[str] | None = None,
) -> TraceResponse:
    """Finalize a trace via ``POST /api/<version>/traces/{trace_id}/finalize``.

    The SDK will substitute the runtime API prefix detected from ``base_url``.

    Args:
        transport: HTTP transport instance.
        trace_id: The trace ID to finalize.
        trigger_evaluators: Whether to trigger evaluator checks after finalization.
        tags: Tags to apply to resulting evaluation executions.

    Returns:
        :class:`~haliosai.types.TraceResponse`
    """
    payload: dict[str, Any] = {
        "trigger_evaluators": trigger_evaluators,
    }
    if tags:
        payload["tags"] = tags

    resp = await transport.request(
        "POST", f"{transport.api_prefix}/traces/{trace_id}/finalize", json=payload
    )
    return TraceResponse.model_validate(resp.json())


async def _get_trace(
    transport: HaliosTransport,
    *,
    trace_id: str,
) -> TraceDetail:
    """Retrieve a trace with its spans via ``GET /api/<version>/traces/{trace_id}``.

    The API version used is auto-detected from the client's configured base URL.

    Returns:
        :class:`~haliosai.types.TraceDetail`
    """
    resp = await transport.request("GET", f"{transport.api_prefix}/traces/{trace_id}")
    return TraceDetail.model_validate(resp.json())


# ---------------------------------------------------------------------------
# TracedConversation context manager
# ---------------------------------------------------------------------------


class TracedConversation:
    """Async context manager that wraps a conversation in a trace.

    Opens a trace on entry, auto-finalizes on exit with optional eval
    trigger. Each ``evaluate()`` call within the context automatically
    gets a unique span_id.

    Usage::

        async with client.traced_conversation(tags=["prod"]) as tc:
            # Each evaluate() automatically gets its own span
            result1 = await client.evaluate(
                messages=[{"role": "user", "content": "..."}],
                trace_id=tc.trace_id,  # span_id auto-generated
            )
            
            result2 = await client.evaluate(
                messages=[...],
                trace_id=tc.trace_id,  # different span_id auto-generated
            )
        # trace auto-finalized on exit

    Parameters:
        client: :class:`~haliosai.client.HaliosClient` instance.
        agent_id: Override the client's default agent_id.
        tags: Tags to attach to the trace.
        trigger_evaluators: Whether to trigger evaluators on finalize.
        finalize_tags: Tags to attach to the finalization (eval tags).
    """

    def __init__(
        self,
        client: Any,
        *,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        trigger_evaluators: bool = False,
        finalize_tags: list[str] | None = None,
    ):
        self._client = client
        self._agent_id = agent_id
        self._tags = tags
        self._trigger_evaluators = trigger_evaluators
        self._finalize_tags = finalize_tags
        self._trace: TraceResponse | None = None

    @property
    def trace_id(self) -> str:
        """The active trace ID. Available after entering the context."""
        if self._trace is None:
            raise RuntimeError("TracedConversation has not been entered yet.")
        return self._trace.trace_id

    @property
    def trace(self) -> TraceResponse:
        """The full trace response. Available after entering the context."""
        if self._trace is None:
            raise RuntimeError("TracedConversation has not been entered yet.")
        return self._trace

    async def __aenter__(self) -> TracedConversation:
        self._trace = await self._client.create_trace(
            agent_id=self._agent_id,
            tags=self._tags,
        )
        logger.debug("Trace opened: %s", self._trace.trace_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._trace is None:
            return
        try:
            await self._client.finalize_trace(
                self._trace.trace_id,
                trigger_evaluators=self._trigger_evaluators,
                tags=self._finalize_tags,
            )
            logger.debug("Trace finalized: %s", self._trace.trace_id)
        except Exception:
            logger.warning(
                "Failed to finalize trace %s", self._trace.trace_id, exc_info=True
            )
