"""Evaluations — trigger eval runs and query check executions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ._transport import HaliosTransport
from .exceptions import EvaluationError, TimeoutError
from .types import (
    CheckExecutionListResult,
    CheckExecutionProgress,
    CheckExecutionResponse,
    TriggerEvalResult,
)

logger = logging.getLogger("haliosai")




# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------


async def _trigger_eval(
    transport: HaliosTransport,
    *,
    agent_id: str,
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
) -> "EvalRun":
    """Trigger an evaluation run via ``POST /api/<version>/trigger-eval``.

    The SDK chooses the API version at runtime based on the configured
    ``base_url`` (``transport.api_prefix``).

    Returns an :class:`EvalRun` helper that can poll for results.
    """
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "mode": mode,
    }
    if trace_ids:
        payload["trace_ids"] = trace_ids
    if trace_tags:
        payload["trace_tags"] = trace_tags
    if start_date:
        payload["start_date"] = start_date
    if dataset_id:
        payload["dataset_id"] = dataset_id
    if dataset_version is not None:
        payload["dataset_version"] = dataset_version
    if check_ids:
        payload["check_ids"] = check_ids
    if tags:
        payload["tags"] = tags
    if run_name:
        payload["run_name"] = run_name
    if run_comment:
        payload["run_comment"] = run_comment

    resp = await transport.request("POST", f"{transport.api_prefix}/trigger-eval", json=payload)
    result = TriggerEvalResult.model_validate(resp.json())
    result_tags = [result.run_tag, *(tags or [])] if result.run_tag else tags

    return EvalRun(
        transport=transport,
        agent_id=agent_id,
        task_id=result.task_id,
        run_tag=result.run_tag,
        status=result.status,
        trace_count=result.trace_count,
        check_count=result.check_count,
        tags=result_tags,
    )


# ---------------------------------------------------------------------------
# List check executions
# ---------------------------------------------------------------------------


async def _list_check_executions(
    transport: HaliosTransport,
    *,
    agent_id: str,
    tags: list[str] | None = None,
    triggered: bool | None = None,
    mode: str | None = None,
    limit: int = 20,
    cursor: str | None = None,
    include_progress: bool = False,
) -> CheckExecutionListResult:
    """Query check executions via ``GET /api/<version>/agents/{agent_id}/check-executions``.

    Uses the new agent-scoped endpoint that supports API-key authentication.
    The runtime API prefix is selected from the client's base_url.

    Returns:
        :class:`~haliosai.types.CheckExecutionListResult` with progress info.
    """
    params: dict[str, Any] = {
        "limit": limit,
        "include_progress": str(include_progress).lower(),
    }
    if tags:
        params["tags"] = tags
    if triggered is not None:
        params["triggered"] = str(triggered).lower()
    if mode:
        params["mode"] = mode
    if cursor:
        params["cursor"] = cursor

    resp = await transport.request(
        "GET",
        f"{transport.api_prefix}/agents/{agent_id}/check-executions",
        params=params,
    )
    body = resp.json()

    data = [CheckExecutionResponse.model_validate(item) for item in body.get("data", [])]
    progress_data = body.get("progress")
    progress = CheckExecutionProgress.model_validate(progress_data) if progress_data else None

    return CheckExecutionListResult(
        data=data,
        next_cursor=body.get("next_cursor"),
        has_more=body.get("has_more", False),
        progress=progress,
    )


# ---------------------------------------------------------------------------
# EvalRun — high-level eval run handle
# ---------------------------------------------------------------------------


class EvalRun:
    """Handle for a triggered evaluation run.

    Provides :meth:`wait` to poll until completion, :meth:`results` to
    fetch all check executions, and :attr:`succeeded` for quick status check.

    Usage::

        run = await client.trigger_eval(tags=["nightly"])
        await run.wait(timeout=120)
        for ex in await run.results():
            print(ex.check_name, ex.triggered, ex.score)

    Attributes:
        task_id: Backend task ID for the evaluation run.
        status: Last known status (``pending``, ``running``, ``completed``, ``failed``).
        trace_count: Number of traces being evaluated.
        check_count: Number of checks being run.
    """

    def __init__(
        self,
        transport: HaliosTransport,
        agent_id: str,
        task_id: str,
        status: str,
        trace_count: int,
        check_count: int,
        run_tag: str = "",
        tags: list[str] | None = None,
    ):
        self.transport = transport
        self.agent_id = agent_id
        self.task_id = task_id
        self.run_tag = run_tag
        self.status = status
        self.trace_count = trace_count
        self.check_count = check_count
        self.tags = tags

    @property
    def succeeded(self) -> bool:
        """Return True if the evaluation completed successfully."""
        return self.status == "completed"

    @property
    def is_done(self) -> bool:
        """Return True if the evaluation is no longer running."""
        return self.status in ("completed", "failed")

    async def wait(
        self,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> "EvalRun":
        """Poll until the evaluation run completes or times out.

        .. deprecated::
            Use :meth:`results` generator instead for streaming results with progress.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between poll attempts.

        Returns:
            Self, with updated status.

        Raises:
            :class:`~haliosai.exceptions.TimeoutError`: If timeout exceeded.
            :class:`~haliosai.exceptions.EvaluationError`: If the run fails.
        """
        # If already completed (sync backend), return immediately
        if self.is_done:
            if self.status == "failed":
                raise EvaluationError(f"Evaluation run {self.task_id} failed")
            return self

        elapsed = 0.0
        expect_results = self.trace_count > 0 and self.check_count > 0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            try:
                task_resp = await self.transport.request(
                    "GET",
                    f"{self.transport.api_prefix}/tasks/{self.task_id}",
                )
                task_status = str(task_resp.json().get("status") or "").upper()
                if task_status in {"SUCCESS", "COMPLETED"}:
                    self.status = "completed"
                    return self
                if task_status in {"FAILURE", "FAILED"}:
                    self.status = "failed"
                    raise EvaluationError(f"Evaluation run {self.task_id} failed")
            except EvaluationError:
                raise
            except Exception as exc:
                logger.debug("Task poll error: %s", exc)

            # Poll by checking progress
            try:
                page = await _list_check_executions(
                    self.transport,
                    agent_id=self.agent_id,
                    tags=self.tags,
                    mode="evaluator",
                    limit=1,
                    include_progress=True,
                )
                # Check if all executions are complete. Use backend progress
                # totals, not trigger-time check_count, because dataset runs
                # can fan out across traces and validation rules.
                if (
                    page.progress
                    and page.progress.pending == 0
                    and page.progress.completed >= page.progress.total
                    and (page.progress.total > 0 or not expect_results)
                ):
                    self.status = "completed"
                    return self
            except Exception as exc:
                logger.debug("Poll error: %s", exc)

        raise TimeoutError(
            f"Evaluation run {self.task_id} did not complete within {timeout}s"
        )

    async def results(
        self,
        *,
        tag: str | None = None,
        triggered: bool | None = None,
        poll_interval: float = 2.0,
        timeout: float = 300,
    ):
        """Stream check execution results as they complete (async generator).

        Polls the backend and yields results incrementally with progress updates.
        Completes when all expected check executions have been received or timeout.

        Args:
            tag: Filter to a specific tag (defaults to eval run tags).
            triggered: Filter to only triggered (or non-triggered) results.
            poll_interval: Seconds between poll attempts.
            timeout: Maximum seconds to wait for all results.

        Yields:
            :class:`~haliosai.types.CheckExecutionResponse` with ``._progress``
            attribute containing :class:`~haliosai.types.CheckExecutionProgress`.

        Example::

            run = await client.trigger_eval(agent_id="agent-123", tags=["golden"])
            async for result in run.results():
                print(f"{result.check_name}: {result.triggered}")
                print(f"Progress: {result._progress.completed}/{result._progress.total}")
        """
        seen_ids = set()
        filter_tags = [tag] if tag else self.tags
        elapsed = 0.0
        cursor = None
        last_progress = None
        expect_results = self.trace_count > 0 and self.check_count > 0

        while elapsed < timeout:
            page = await _list_check_executions(
                self.transport,
                agent_id=self.agent_id,
                tags=filter_tags,
                triggered=triggered,
                mode="evaluator",
                limit=100,
                cursor=cursor,
                include_progress=True,
            )

            # Yield new results
            for item in page.data:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    # Attach progress info to each result for convenience
                    item._progress = page.progress or CheckExecutionProgress()
                    yield item

            last_progress = page.progress

            # Move through every page before deciding the run is fully collected.
            if page.has_more:
                cursor = page.next_cursor
                if cursor is None:
                    logger.warning("Eval results page had has_more=True without a cursor")
                    return
                continue

            # Check if complete after the final page. Progress totals come from
            # the backend result query and may exceed trigger-time check_count
            # for dataset runs where each trace/check can fan out to rule rows.
            if (
                last_progress
                and last_progress.pending == 0
                and last_progress.completed >= last_progress.total
                and (last_progress.total > 0 or not expect_results)
            ):
                self.status = "completed"
                return

            # No more pages yet; wait for additional executions to appear.
            cursor = None
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        completed = last_progress.completed if last_progress else len(seen_ids)
        total = last_progress.total if last_progress else "unknown"
        raise TimeoutError(
            f"Evaluation run {self.task_id} results were not available within "
            f"{timeout}s (got {completed}/{total})"
        )

    def __repr__(self) -> str:
        return (
            f"EvalRun(task_id={self.task_id!r}, status={self.status!r}, "
            f"traces={self.trace_count}, checks={self.check_count})"
        )
