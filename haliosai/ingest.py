"""Bulk ingest and run metadata helpers for CI/data-pipeline workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from ._transport import HaliosTransport
from .types import BulkIngestResponse, IngestTaskStatus, RunMetadataResult


async def _create_run(
    transport: HaliosTransport,
    *,
    agent_id: str,
    run_tag: str,
    source: str = "dashboard",
    commit_sha: str | None = None,
    branch: str | None = None,
    pipeline_id: str | None = None,
    pipeline_url: str | None = None,
    environment: str | None = None,
    job_id: str | None = None,
    dataset_version: str | None = None,
    input_row_count: int | None = None,
    status: str = "running",
) -> RunMetadataResult:
    payload = {
        "agent_id": agent_id,
        "run_tag": run_tag,
        "source": source,
        "commit_sha": commit_sha,
        "branch": branch,
        "pipeline_id": pipeline_id,
        "pipeline_url": pipeline_url,
        "environment": environment,
        "job_id": job_id,
        "dataset_version": dataset_version,
        "input_row_count": input_row_count,
        "status": status,
    }
    resp = await transport.request("POST", f"{transport.api_prefix}/runs", json=payload)
    return RunMetadataResult.model_validate(resp.json())


async def _bulk_ingest_traces(
    transport: HaliosTransport,
    *,
    agent_id: str,
    traces: list[dict[str, Any]],
    run_tag: str | None = None,
    tags: list[str] | None = None,
    cohort_slug: str | None = None,
    idempotency_key: str | None = None,
    source: str = "runtime",
) -> BulkIngestResponse:
    payload = {
        "agent_id": agent_id,
        "run_tag": run_tag,
        "tags": tags or [],
        "cohort_slug": cohort_slug,
        "idempotency_key": idempotency_key,
        "source": source,
        "traces": traces,
    }
    resp = await transport.request("POST", f"{transport.api_prefix}/ingest/traces/bulk", json=payload)
    return BulkIngestResponse.model_validate(resp.json())


async def _bulk_ingest_check_executions(
    transport: HaliosTransport,
    *,
    agent_id: str,
    executions: list[dict[str, Any]],
    run_tag: str | None = None,
    tags: list[str] | None = None,
    cohort_slug: str | None = None,
    source: str = "harness",
) -> BulkIngestResponse:
    payload = {
        "agent_id": agent_id,
        "run_tag": run_tag,
        "tags": tags or [],
        "cohort_slug": cohort_slug,
        "source": source,
        "executions": executions,
    }
    resp = await transport.request("POST", f"{transport.api_prefix}/ingest/check-executions/bulk", json=payload)
    return BulkIngestResponse.model_validate(resp.json())


async def _get_ingest_task_status(transport: HaliosTransport, *, task_id: str) -> IngestTaskStatus:
    resp = await transport.request("GET", f"{transport.api_prefix}/ingest/tasks/{task_id}")
    return IngestTaskStatus.model_validate(resp.json())


@dataclass
class BulkIngestResult:
    task_ids: list[str] = field(default_factory=list)
    accepted_count: int = 0
    duplicate_count: int = 0
    rejected_count: int = 0
    validation_errors: list[str] = field(default_factory=list)


class BulkIngester:
    """Batch trace ingester for CI/data-engineering workflows."""

    def __init__(
        self,
        transport: HaliosTransport,
        *,
        agent_id: str,
        run_tag: str | None,
        tags: list[str] | None = None,
        cohort_slug: str | None = None,
        source: str = "ci_pipeline",
        concurrency: int = 16,
        batch_size: int = 100,
    ):
        self.transport = transport
        self.agent_id = agent_id
        self.run_tag = run_tag
        self.tags = tags or []
        self.cohort_slug = cohort_slug
        self.source = source
        self.concurrency = max(1, concurrency)
        self.batch_size = max(1, batch_size)
        self._buffer: list[dict[str, Any]] = []
        self._task_ids: list[str] = []

    async def __aenter__(self) -> "BulkIngester":
        return self

    async def __aexit__(self, *args):
        if self._buffer:
            await self._flush()

    def submit_trace(self, messages, external_trace_id=None, metadata=None) -> str:
        trace_row = {
            "external_trace_id": external_trace_id,
            "spans": [{"name": "bulk-ingest", "kind": "custom", "input": {"messages": messages}}],
            "metadata": metadata or {},
        }
        self._buffer.append(trace_row)
        token = external_trace_id or f"pending-{len(self._buffer)}"
        return token

    async def _flush(self):
        if not self._buffer:
            return
        payload = self._buffer[: self.batch_size]
        self._buffer = self._buffer[self.batch_size :]
        response = await _bulk_ingest_traces(
            self.transport,
            agent_id=self.agent_id,
            traces=payload,
            run_tag=self.run_tag,
            tags=self.tags,
            cohort_slug=self.cohort_slug,
            source=self.source,
        )
        self._task_ids.append(response.task_id)

    async def wait_for_results(self, timeout: int = 300) -> BulkIngestResult:
        while self._buffer:
            await self._flush()

        started = datetime.now(timezone.utc).timestamp()
        result = BulkIngestResult(task_ids=list(self._task_ids))

        pending = set(self._task_ids)
        while pending:
            if (datetime.now(timezone.utc).timestamp() - started) > timeout:
                break

            for task_id in list(pending):
                status = await _get_ingest_task_status(self.transport, task_id=task_id)
                if status.status in {"completed", "failed"}:
                    result.accepted_count += status.accepted_count
                    result.duplicate_count += status.duplicate_count
                    result.rejected_count += status.rejected_count
                    result.validation_errors.extend(status.errors)
                    pending.discard(task_id)

            if pending:
                await asyncio.sleep(1.0)

        return result


class BulkEvalPusher:
    """Pushes external evaluator outputs via bulk check-execution ingest."""

    def __init__(
        self,
        transport: HaliosTransport,
        *,
        agent_id: str,
        run_tag: str | None,
        tags: list[str] | None = None,
        cohort_slug: str | None = None,
        source: str = "spark",
    ):
        self.transport = transport
        self.agent_id = agent_id
        self.run_tag = run_tag
        self.tags = tags or []
        self.cohort_slug = cohort_slug
        self.source = source
        self._task_ids: list[str] = []

    async def push(self, rows: Iterable[dict], batch_size: int = 500) -> str:
        rows = list(rows)
        task_id = ""
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            response = await _bulk_ingest_check_executions(
                self.transport,
                agent_id=self.agent_id,
                executions=batch,
                run_tag=self.run_tag,
                tags=self.tags,
                cohort_slug=self.cohort_slug,
                source=self.source,
            )
            task_id = response.task_id
            self._task_ids.append(task_id)
        return task_id

    async def wait_for_completion(self, timeout: int = 300) -> BulkIngestResult:
        started = datetime.now(timezone.utc).timestamp()
        result = BulkIngestResult(task_ids=list(self._task_ids))
        pending = set(self._task_ids)

        while pending:
            if (datetime.now(timezone.utc).timestamp() - started) > timeout:
                break
            for task_id in list(pending):
                status = await _get_ingest_task_status(self.transport, task_id=task_id)
                if status.status in {"completed", "failed"}:
                    result.accepted_count += status.accepted_count
                    result.rejected_count += status.rejected_count
                    result.validation_errors.extend(status.errors)
                    pending.discard(task_id)
            if pending:
                await asyncio.sleep(1.0)

        return result
