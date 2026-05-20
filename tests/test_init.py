"""Tests for __init__.py — public API surface."""

from __future__ import annotations

import haliosai


class TestPublicAPI:
    def test_version(self):
        assert hasattr(haliosai, "__version__")
        assert haliosai.__version__ == "1.1.0"

    def test_client_exported(self):
        assert hasattr(haliosai, "HaliosClient")

    def test_exceptions_exported(self):
        assert hasattr(haliosai, "HaliosError")
        assert hasattr(haliosai, "ConfigError")
        assert hasattr(haliosai, "HaliosAPIError")
        assert hasattr(haliosai, "GuardrailTriggered")
        assert hasattr(haliosai, "EvaluationError")
        assert hasattr(haliosai, "TimeoutError")

    def test_guardrails_exported(self):
        assert hasattr(haliosai, "guarded")
        assert hasattr(haliosai, "extract_response_message")

    def test_tracing_exported(self):
        assert hasattr(haliosai, "TracedConversation")

    def test_evaluations_exported(self):
        assert hasattr(haliosai, "EvalRun")

    def test_types_exported(self):
        expected_types = [
            "EvaluateResult",
            "Violation",
            "CheckResult",
            "GuardrailPolicy",
            "ViolationAction",
            "ExecutionResult",
            "GuardedResponse",
            "SpanResponse",
            "TraceResponse",
            "TraceDetail",
            "CheckExecutionResponse",
            "PaginatedResult",
        ]
        for name in expected_types:
            assert hasattr(haliosai, name), f"Missing export: {name}"

    def test_all_list_complete(self):
        for name in haliosai.__all__:
            assert hasattr(haliosai, name), f"__all__ contains {name} but it's not exported"
