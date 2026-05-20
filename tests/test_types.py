"""Tests for types / Pydantic models."""

from __future__ import annotations

from datetime import datetime

from haliosai.types import (
    CheckExecutionResponse,
    CheckResult,
    EvaluateResult,
    ExecutionResult,
    GuardedResponse,
    GuardrailPolicy,
    PaginatedResult,
    SpanResponse,
    TraceDetail,
    TraceResponse,
    Violation,
    ViolationAction,
)


class TestEnums:
    def test_guardrail_policy_values(self):
        assert GuardrailPolicy.RECORD_ONLY.value == "record_only"
        assert GuardrailPolicy.BLOCK.value == "block"

    def test_violation_action_values(self):
        assert ViolationAction.PASS.value == "pass"
        assert ViolationAction.BLOCK.value == "block"
        assert ViolationAction.ALLOW_OVERRIDE.value == "allow_override"

    def test_execution_result_values(self):
        assert ExecutionResult.SUCCESS.value == "success"
        assert ExecutionResult.REQUEST_BLOCKED.value == "request_blocked"


class TestEvaluateResult:
    def test_defaults(self):
        r = EvaluateResult()
        assert r.triggered is False
        assert r.action == "allow"
        assert r.violations == []
        assert r.check_results == []

    def test_from_dict(self):
        data = {
            "triggered": True,
            "action": "block",
            "violations": [{"check_name": "pii", "message": "found"}],
            "check_results": [],
            "trace_id": "t1",
            "span_id": "s1",
            "latency_ms": 120,
        }
        r = EvaluateResult.model_validate(data)
        assert r.triggered is True
        assert len(r.violations) == 1
        assert r.violations[0].check_name == "pii"


class TestTraceModels:
    def test_trace_response(self):
        data = {
            "trace_id": "abc",
            "agent_id": "x",
            "status": "finalized",
            "started_at": "2024-01-01T00:00:00",
            "created_at": "2024-01-01T00:00:00",
        }
        t = TraceResponse.model_validate(data)
        assert t.trace_id == "abc"
        assert t.status == "finalized"

    def test_trace_detail_inherits(self):
        data = {
            "trace_id": "abc",
            "agent_id": "x",
            "started_at": "2024-01-01T00:00:00",
            "created_at": "2024-01-01T00:00:00",
            "spans": [
                {
                    "span_id": "s1",
                    "trace_id": "abc",
                    "started_at": "2024-01-01T00:00:00",
                }
            ],
        }
        td = TraceDetail.model_validate(data)
        assert len(td.spans) == 1
        assert td.spans[0].span_id == "s1"


class TestCheckExecutionResponse:
    def test_parse(self):
        data = {
            "id": "1",
            "trace_id": "t1",
            "span_id": "s1",
            "validation_rule_id": "vr1",
            "mode": "evaluator",
            "triggered": True,
            "score": 0.9,
            "tags": ["nightly"],
            "created_at": "2024-01-01T00:00:00",
        }
        ex = CheckExecutionResponse.model_validate(data)
        assert ex.triggered is True
        assert ex.tags == ["nightly"]
        assert ex.score == 0.9


class TestPaginatedResult:
    def test_defaults(self):
        p = PaginatedResult()
        assert p.data == []
        assert p.has_more is False
        assert p.next_cursor is None


class TestGuardedResponse:
    def test_default(self):
        g = GuardedResponse()
        assert g.result == ExecutionResult.SUCCESS
        assert g.final_response is None
        assert g.request_violations == []
