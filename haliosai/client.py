"""HaliosClient — top-level SDK entry point.

All domain methods (evaluate, traces, evals, analytics) are delegated to
dedicated modules but exposed as methods on this single client for ergonomics.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ._transport import HaliosTransport
from .exceptions import ConfigError

logger = logging.getLogger("haliosai")

_API_PREFIX = "/api/v1"


class HaliosClient:
    """Top-level HaliosAI client.

    Usage::

        client = HaliosClient(agent_id="my-agent")
        result = await client.evaluate(messages=[...])

    Parameters:
        agent_id: Default agent slug or UUID. Can be overridden per-call.
        api_key: API key (falls back to ``HALIOS_API_KEY`` env var).
        base_url: API base URL (falls back to ``HALIOS_BASE_URL`` env var).
        timeout: HTTP request timeout in seconds.
        max_retries: Number of retry attempts on transient errors.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.agent_id = agent_id
        self.api_key = api_key or os.getenv("HALIOS_API_KEY", "")
        # Accept base_url with or without an API-version suffix; transport
        # will normalize and detect the API prefix (e.g. /api/v1 or /api/v2).
        self.base_url = (base_url or os.getenv("HALIOS_BASE_URL", "https://api.halios.ai")).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ConfigError(
                "api_key is required. Pass it directly or set HALIOS_API_KEY env var."
            )

        self._transport = HaliosTransport(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        # Keep client.base_url in sync with transport-normalized base_url
        self.base_url = self._transport.base_url

    # -- helpers ----------------------------------------------------------

    def _resolve_agent(self, agent_id: str | None = None) -> str:
        aid = agent_id or self.agent_id
        if not aid:
            raise ConfigError(
                "agent_id is required. Pass it to the method or set it on HaliosClient."
            )
        return aid

    # =====================================================================
    # Guardrails
    # =====================================================================

    async def evaluate(
        self,
        messages: list[dict[str, Any]],
        *,
        mode: str = "guardrail",
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
    ):
        """Evaluate messages against configured guardrails.

        Maps to ``POST /api/v1/evaluate``.

        Args:
            messages: Conversation messages to evaluate.
            mode: Evaluation mode ("guardrail" or "evaluator").
            trace_id: Optional trace ID to group spans.
            span_id: Optional span ID (auto-generated if trace_id provided).
            parent_span_id: Optional parent span ID for creating hierarchy.
            tags: Optional tags for filtering/grouping.
            agent_id: Optional agent UUID or slug (defaults to client's agent).

        Returns:
            :class:`~haliosai.types.EvaluateResult`
        """
        from .guardrails import _evaluate

        return await _evaluate(
            self._transport,
            agent_id=self._resolve_agent(agent_id),
            messages=messages,
            mode=mode,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            tags=tags,
        )

    async def validate_request(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ):
        """Convenience: evaluate as ``mode='guardrail'``, raise on trigger."""
        from .guardrails import _validate

        return await _validate(
            self._transport,
            agent_id=self._resolve_agent(kwargs.pop("agent_id", None)),
            messages=messages,
            invocation_type="request",
            **kwargs,
        )

    async def validate_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ):
        """Convenience: evaluate response messages, raise on trigger."""
        from .guardrails import _validate

        return await _validate(
            self._transport,
            agent_id=self._resolve_agent(kwargs.pop("agent_id", None)),
            messages=messages,
            invocation_type="response",
            **kwargs,
        )

    def evaluate_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        mode: str = "guardrail",
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
    ):
        """Synchronous evaluate for notebooks / PySpark."""
        from .guardrails import _evaluate_sync

        return _evaluate_sync(
            self._transport,
            agent_id=self._resolve_agent(agent_id),
            messages=messages,
            mode=mode,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            tags=tags,
        )

    # =====================================================================
    # Tracing
    # =====================================================================

    async def create_trace(
        self,
        *,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        trace_id: str | None = None,
    ):
        """Create a new trace. Returns :class:`~haliosai.types.TraceResponse`."""
        from .tracing import _create_trace

        return await _create_trace(
            self._transport,
            agent_id=self._resolve_agent(agent_id),
            tags=tags,
            trace_id=trace_id,
        )

    async def finalize_trace(
        self,
        trace_id: str,
        *,
        trigger_evaluators: bool = False,
        tags: list[str] | None = None,
    ):
        """Finalize a trace. Returns :class:`~haliosai.types.TraceResponse`."""
        from .tracing import _finalize_trace

        return await _finalize_trace(
            self._transport,
            trace_id=trace_id,
            trigger_evaluators=trigger_evaluators,
            tags=tags,
        )

    async def get_trace(self, trace_id: str):
        """Get trace detail. Returns :class:`~haliosai.types.TraceDetail`."""
        from .tracing import _get_trace

        return await _get_trace(self._transport, trace_id=trace_id)

    def traced_conversation(self, **kwargs: Any):
        """Return a :class:`~haliosai.tracing.TracedConversation` context manager."""
        from .tracing import TracedConversation

        return TracedConversation(client=self, **kwargs)

    # =====================================================================
    # Evaluations
    # =====================================================================

    async def trigger_eval(
        self,
        *,
        agent_id: str | None = None,
        trace_ids: list[str] | None = None,
        trace_tags: list[str] | None = None,
        start_date: str | None = None,
        dataset_id: str | None = None,
        dataset_version: int | None = None,
        check_ids: list[str] | None = None,
        tags: list[str] | None = None,
        run_name: str | None = None,
        run_comment: str | None = None,
        mode: str = "evaluator",
    ):
        """Trigger an evaluation run. Returns :class:`~haliosai.evaluations.EvalRun`."""
        from .evaluations import _trigger_eval

        return await _trigger_eval(
            self._transport,
            agent_id=self._resolve_agent(agent_id),
            trace_ids=trace_ids,
            trace_tags=trace_tags,
            start_date=start_date,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            check_ids=check_ids,
            tags=tags,
            run_name=run_name,
            run_comment=run_comment,
            mode=mode,
        )

    async def trigger_dataset_eval(
        self,
        *,
        dataset_id: str,
        agent_id: str | None = None,
        dataset_version: int | None = None,
        check_ids: list[str] | None = None,
        tags: list[str] | None = None,
        run_name: str | None = None,
        run_comment: str | None = None,
    ):
        """Trigger evaluator checks on all traces in a dataset snapshot."""
        return await self.trigger_eval(
            agent_id=agent_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            check_ids=check_ids,
            tags=tags,
            run_name=run_name,
            run_comment=run_comment,
            mode="evaluator",
        )

    async def list_check_executions(
        self,
        *,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        triggered: bool | None = None,
        mode: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ):
        """List check executions. Returns :class:`~haliosai.types.PaginatedResult`."""
        from .evaluations import _list_check_executions

        return await _list_check_executions(
            self._transport,
            agent_id=self._resolve_agent(agent_id),
            tags=tags,
            triggered=triggered,
            mode=mode,
            limit=limit,
            cursor=cursor,
        )

    # =====================================================================
    # Cohorts / Run metadata / Bulk ingest
    # =====================================================================

    async def get_cohorts(self, *, agent_id: str | None = None):
        """Fetch cohort definitions available to this org/agent."""
        from .cohorts import _get_cohorts

        return await _get_cohorts(
            self._transport,
            agent_id=self._resolve_agent(agent_id) if agent_id else self.agent_id,
        )

    async def validate_cohort_tags(
        self,
        *,
        tags: list[str],
        cohort_slug: str | None = None,
        mode: str | None = None,
        agent_id: str | None = None,
    ):
        """Validate a tag set against the cohort registry."""
        from .cohorts import _validate_cohort_tags

        return await _validate_cohort_tags(
            self._transport,
            tags=tags,
            cohort_slug=cohort_slug,
            mode=mode,
            agent_id=self._resolve_agent(agent_id),
        )

    async def create_run(self, *, run_tag: str, source: str = "dashboard", **kwargs: Any):
        """Create run metadata entry for CI/data-pipeline linking."""
        from .ingest import _create_run

        return await _create_run(
            self._transport,
            agent_id=self._resolve_agent(kwargs.pop("agent_id", None)),
            run_tag=run_tag,
            source=source,
            **kwargs,
        )

    async def ingest_traces_bulk(
        self,
        *,
        traces: list[dict[str, Any]],
        run_tag: str | None = None,
        tags: list[str] | None = None,
        cohort_slug: str | None = None,
        idempotency_key: str | None = None,
        source: str = "runtime",
        agent_id: str | None = None,
    ):
        """Submit bulk trace ingest payload."""
        from .ingest import _bulk_ingest_traces

        return await _bulk_ingest_traces(
            self._transport,
            agent_id=self._resolve_agent(agent_id),
            traces=traces,
            run_tag=run_tag,
            tags=tags,
            cohort_slug=cohort_slug,
            idempotency_key=idempotency_key,
            source=source,
        )

    async def ingest_check_executions_bulk(
        self,
        *,
        executions: list[dict[str, Any]],
        run_tag: str | None = None,
        tags: list[str] | None = None,
        cohort_slug: str | None = None,
        source: str = "harness",
        agent_id: str | None = None,
    ):
        """Submit bulk external check-execution payload."""
        from .ingest import _bulk_ingest_check_executions

        return await _bulk_ingest_check_executions(
            self._transport,
            agent_id=self._resolve_agent(agent_id),
            executions=executions,
            run_tag=run_tag,
            tags=tags,
            cohort_slug=cohort_slug,
            source=source,
        )

    async def get_ingest_task(self, task_id: str):
        """Get bulk ingest task status."""
        from .ingest import _get_ingest_task_status

        return await _get_ingest_task_status(self._transport, task_id=task_id)

    # =====================================================================
    # Resource management
    # =====================================================================

    async def close(self) -> None:
        """Close the underlying HTTP connections."""
        await self._transport.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
