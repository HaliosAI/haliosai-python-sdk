"""Typed responses for the intentionally small Halios Python API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Violation(BaseModel):
    check_id: str = ""
    check_name: str = ""
    validation_rule_id: str = ""
    validation_rule_name: str = ""
    message: str = ""
    severity: str = "medium"


class CheckResult(BaseModel):
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
    """Result of an explicit inline request or response evaluation."""

    triggered: bool = False
    action: str = "allow"
    violations: list[Violation] = Field(default_factory=list)
    check_results: list[CheckResult] = Field(default_factory=list)
    trace_id: str = ""
    span_id: str = ""
    latency_ms: int = 0

    @property
    def blocked(self) -> bool:
        return self.triggered and self.action == "block"


class SpanDetail(BaseModel):
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

    model_config = {"extra": "allow"}


class TraceDetail(BaseModel):
    trace_id: str
    agent_id: str = ""
    agent_name: str | None = None
    status: str = "active"
    ingest_source: str = "live"
    evaluation_context: str = "unclassified"
    labels: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    spans: list[SpanDetail] = Field(default_factory=list)
    created_at: datetime | None = None
    finalized_at: datetime | None = None

    model_config = {"extra": "allow"}


class EvaluationRun(BaseModel):
    """Immutable bounded evaluation-run report."""

    run_id: str
    run_tag: str = ""
    status: str
    suite_digest: str | None = None
    attempted_trial_count: int = 0
    completed_trial_count: int = 0
    evaluated_trial_count: int = 0
    telemetry_incomplete_count: int = 0
    check_execution_error_count: int = 0
    pass_at_k: float = 0.0
    threshold: float = 0.0
    protected_failure: bool = False
    gate_passed: bool = False
    trials: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "allow"}
