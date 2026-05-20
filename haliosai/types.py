"""Pydantic request/response types mirroring the V2 backend schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GuardrailPolicy(str, Enum):
    """Per-guardrail policy action."""

    RECORD_ONLY = "record_only"
    BLOCK = "block"


class ViolationAction(str, Enum):
    """Resolved action after policy evaluation."""

    PASS = "pass"
    BLOCK = "block"
    ALLOW_OVERRIDE = "allow_override"


class ExecutionResult(str, Enum):
    """Execution result status codes."""

    SUCCESS = "success"
    REQUEST_BLOCKED = "request_blocked"
    RESPONSE_BLOCKED = "response_blocked"
    TIMEOUT = "timeout"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


class Violation(BaseModel):
    """Single guardrail violation."""

    check_id: str = ""
    check_name: str = ""
    validation_rule_id: str = ""
    validation_rule_name: str = ""
    message: str = ""
    severity: str = "medium"


class CheckResult(BaseModel):
    """Single check result within an evaluation."""

    check_id: str = ""
    check_name: str = ""
    validation_rule_id: str = ""
    validation_rule_name: str = ""
    triggered: bool = False
    score: float | None = None
    passed: bool | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None
    latency_ms: int = 0


class EvaluateResult(BaseModel):
    """Response from ``POST /api/<version>/evaluate`` (SDK auto-detects API version from base_url)."""

    triggered: bool = False
    action: str = "allow"
    violations: list[Violation] = Field(default_factory=list)
    check_results: list[CheckResult] = Field(default_factory=list)
    trace_id: str = ""
    span_id: str = ""
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


class SpanResponse(BaseModel):
    """Span data within a trace."""

    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    name: str | None = None
    kind: str | None = None
    status: str | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None


class TraceResponse(BaseModel):
    """Trace response (list view)."""

    trace_id: str
    agent_id: str = ""
    agent_name: str | None = None
    status: str = "active"
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    trace_stats: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finalized_at: datetime | None = None
    created_at: datetime | None = None


class TraceDetail(TraceResponse):
    """Trace response with spans included."""

    spans: list[SpanResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluations / Trigger-Eval
# ---------------------------------------------------------------------------


class TriggerEvalResult(BaseModel):
    """Response from ``POST /api/<version>/trigger-eval`` (API version is auto-detected)."""

    task_id: str = ""
    run_tag: str = ""
    status: str = "pending"
    trace_count: int = 0
    check_count: int = 0


class CheckExecutionResponse(BaseModel):
    """Check execution result record."""

    id: str = ""
    trace_id: str = ""
    span_id: str = ""
    check_id: str | None = None
    check_name: str | None = None
    task_name: str | None = None
    task_slug: str | None = None
    validation_rule_id: str = ""
    validation_rule_name: str | None = None
    mode: str = "guardrail"
    scope: str = "single"
    tags: list[str] = Field(default_factory=list)
    triggered: bool = False
    score: float | None = None
    passed: bool | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None
    latency_ms: int | None = None
    tokens_used: int = 0
    error: str | None = None
    created_at: datetime | None = None


class CheckExecutionProgress(BaseModel):
    """Progress information for check execution listing."""

    total: int = 0
    completed: int = 0
    pending: int = 0


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class PaginatedResult(BaseModel):
    """Generic paginated response wrapper."""

    data: list[Any] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


class CheckExecutionListResult(BaseModel):
    """Check execution list with progress info."""

    data: list[CheckExecutionResponse] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False
    progress: CheckExecutionProgress | None = None


# ---------------------------------------------------------------------------
# Cohorts / Runs / Bulk ingest
# ---------------------------------------------------------------------------


class CohortMatchConfig(BaseModel):
    include_tags_any: list[str] = Field(default_factory=list)
    include_tags_all: list[str] = Field(default_factory=list)
    exclude_tags_any: list[str] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)
    mode_scope: str = "all"


class CohortDefinition(BaseModel):
    id: str
    name: str
    slug: str
    type: str = "custom"
    match: CohortMatchConfig = Field(default_factory=CohortMatchConfig)
    strict_match: bool = False
    auto_stamp: bool = True
    tag_allowlist: list[str] | None = None
    require_tags_all: list[str] | None = None


class CohortValidateResult(BaseModel):
    normalized_tags: list[str] = Field(default_factory=list)
    cohort_id: str | None = None
    cohort_slug: str | None = None
    valid: bool = True
    reasons: list[str] = Field(default_factory=list)


class RunMetadataResult(BaseModel):
    id: int
    organization_id: str
    agent_id: int | None = None
    run_tag: str
    source: str = "dashboard"
    commit_sha: str | None = None
    branch: str | None = None
    pipeline_id: str | None = None
    pipeline_url: str | None = None
    environment: str | None = None
    job_id: str | None = None
    dataset_version: str | None = None
    input_row_count: int | None = None
    status: str = "running"
    triggered_at: datetime
    completed_at: datetime | None = None


class BulkIngestResponse(BaseModel):
    task_id: str
    accepted_count: int = 0
    duplicate_count: int = 0
    rejected_count: int = 0
    rejected_reasons: list[str] = Field(default_factory=list)
    unknown_trace_count: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    run_tag: str | None = None
    status: str = "pending"
    status_url: str | None = None


class IngestTaskStatus(BaseModel):
    id: str
    task_type: str
    status: str
    run_tag: str | None = None
    source: str | None = None
    accepted_count: int = 0
    duplicate_count: int = 0
    rejected_count: int = 0
    unknown_trace_count: int = 0
    errors: list[str] = Field(default_factory=list)
    error_message: str | None = None
    celery_state: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Guarded response (decorator result)
# ---------------------------------------------------------------------------


class GuardedResponse(BaseModel):
    """Response from a guarded LLM call."""

    result: ExecutionResult = ExecutionResult.SUCCESS
    final_response: Any = None
    original_response: str | None = None
    request_violations: list[Violation] = Field(default_factory=list)
    response_violations: list[Violation] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)
    error_message: str | None = None

    model_config = {"arbitrary_types_allowed": True}
