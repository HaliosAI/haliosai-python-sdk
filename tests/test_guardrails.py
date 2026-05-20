"""Tests for guardrails module — evaluate, validate, and @guarded decorator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from haliosai.exceptions import GuardrailTriggered
from haliosai.guardrails import (
    _evaluate,
    _evaluate_sync,
    _resolve_action,
    _validate,
    extract_response_message,
    guarded,
)
from haliosai.types import (
    EvaluateResult,
    GuardrailPolicy,
    Violation,
    ViolationAction,
)

from conftest import EVALUATE_BLOCK, EVALUATE_PASS


# ---------------------------------------------------------------------------
# evaluate / evaluate_sync
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_returns_typed_result(self, mock_transport):
        result = await _evaluate(
            mock_transport,
            agent_id="test-agent",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert isinstance(result, EvaluateResult)
        assert result.triggered is False
        assert result.trace_id == "abc123"

    @pytest.mark.asyncio
    async def test_evaluate_with_all_params(self, mock_transport):
        await _evaluate(
            mock_transport,
            agent_id="test-agent",
            messages=[{"role": "user", "content": "hi"}],
            mode="evaluator",
            trace_id="t1",
            span_id="s1",
            tags=["test"],
        )
        call = mock_transport._calls[-1]
        assert call["json"]["mode"] == "evaluator"
        assert call["json"]["trace_id"] == "t1"
        assert call["json"]["tags"] == ["test"]

    def test_evaluate_sync(self, mock_transport):
        result = _evaluate_sync(
            mock_transport,
            agent_id="test-agent",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert isinstance(result, EvaluateResult)


# ---------------------------------------------------------------------------
# validate (evaluate + raise)
# ---------------------------------------------------------------------------


class TestValidate:
    @pytest.mark.asyncio
    async def test_validate_pass(self, mock_transport):
        result = await _validate(
            mock_transport,
            agent_id="test-agent",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result.triggered is False

    @pytest.mark.asyncio
    async def test_validate_block_raises(self, mock_transport):
        mock_transport.set_response("POST /api/v1/evaluate", EVALUATE_BLOCK)
        with pytest.raises(GuardrailTriggered) as exc_info:
            await _validate(
                mock_transport,
                agent_id="test-agent",
                messages=[{"role": "user", "content": "my SSN is 123-45-6789"}],
            )
        assert exc_info.value.violation_type == "request"
        assert len(exc_info.value.violations) == 1
        assert exc_info.value.violations[0].check_name == "pii_detection"


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


class TestResolveAction:
    def test_no_violations_returns_pass(self):
        result = EvaluateResult(triggered=False, violations=[])
        assert _resolve_action(result, None) == ViolationAction.PASS

    def test_violations_no_policy_returns_block(self):
        result = EvaluateResult(
            triggered=True,
            violations=[Violation(check_name="toxicity", message="bad")],
        )
        assert _resolve_action(result, None) == ViolationAction.BLOCK

    def test_record_only_policy_allows(self):
        result = EvaluateResult(
            triggered=True,
            violations=[Violation(check_name="toxicity", message="bad")],
        )
        policies = {"toxicity": GuardrailPolicy.RECORD_ONLY}
        assert _resolve_action(result, policies) == ViolationAction.ALLOW_OVERRIDE

    def test_mixed_policies_block_wins(self):
        result = EvaluateResult(
            triggered=True,
            violations=[
                Violation(check_name="toxicity", message="bad"),
                Violation(check_name="pii", message="pii found"),
            ],
        )
        policies = {
            "toxicity": GuardrailPolicy.RECORD_ONLY,
            "pii": GuardrailPolicy.BLOCK,
        }
        assert _resolve_action(result, policies) == ViolationAction.BLOCK


# ---------------------------------------------------------------------------
# extract_response_message
# ---------------------------------------------------------------------------


class TestExtractResponseMessage:
    def test_string_input(self):
        msg = extract_response_message("hello world")
        assert msg == {"role": "assistant", "content": "hello world"}

    def test_dict_with_choices(self):
        resp = {
            "choices": [{"message": {"content": "response text", "role": "assistant"}}]
        }
        msg = extract_response_message(resp)
        assert msg["content"] == "response text"

    def test_dict_with_output_key(self):
        msg = extract_response_message({"output": "some output"})
        assert msg["content"] == "some output"

    def test_openai_like_object(self):
        class Choice:
            class Message:
                content = "from object"
                tool_calls = None
            message = Message()

        class Response:
            choices = [Choice()]

        msg = extract_response_message(Response())
        assert msg["content"] == "from object"


# ---------------------------------------------------------------------------
# @guarded decorator
# ---------------------------------------------------------------------------


class TestGuardedDecorator:
    @pytest.mark.asyncio
    async def test_guarded_concurrent_pass(self):
        import respx

        with respx.mock:
            respx.post("https://api.test.halios.ai/api/v1/evaluate").respond(
                200, json=EVALUATE_PASS
            )

            @guarded(agent_id="test-agent", api_key="test-key-123", base_url="https://api.test.halios.ai")
            async def my_llm(messages):
                return "LLM response text"

            result = await my_llm([{"role": "user", "content": "hello"}])
            assert result == "LLM response text"

    @pytest.mark.asyncio
    async def test_guarded_sequential_pass(self):
        import respx

        with respx.mock:
            respx.post("https://api.test.halios.ai/api/v1/evaluate").respond(
                200, json=EVALUATE_PASS
            )

            @guarded(
                agent_id="test-agent",
                api_key="test-key-123",
                base_url="https://api.test.halios.ai",
                concurrent=False,
            )
            async def my_llm(messages):
                return "response"

            result = await my_llm([{"role": "user", "content": "hello"}])
            assert result == "response"

    @pytest.mark.asyncio
    async def test_guarded_block_raises(self):
        """Test that a blocking evaluation raises GuardrailTriggered."""
        import respx

        with respx.mock:
            respx.post("https://api.test.halios.ai/api/v1/evaluate").respond(
                200, json=EVALUATE_BLOCK
            )

            @guarded(
                agent_id="test-agent",
                api_key="test-key-123",
                base_url="https://api.test.halios.ai",
            )
            async def my_llm(messages):
                return "response"

            with pytest.raises(GuardrailTriggered):
                await my_llm([{"role": "user", "content": "hello"}])

    @pytest.mark.asyncio
    async def test_guarded_on_violation_callback(self):
        """Test that on_violation callback is called when block happens."""
        import respx

        violations_received = []

        def on_violation(exc):
            violations_received.append(exc)

        with respx.mock:
            respx.post("https://api.test.halios.ai/api/v1/evaluate").respond(
                200, json=EVALUATE_BLOCK
            )

            @guarded(
                agent_id="test-agent",
                api_key="test-key-123",
                base_url="https://api.test.halios.ai",
                on_violation=on_violation,
            )
            async def my_llm(messages):
                return "response"

            with pytest.raises(GuardrailTriggered):
                await my_llm([{"role": "user", "content": "hello"}])

            assert len(violations_received) == 1
            assert isinstance(violations_received[0], GuardrailTriggered)
