import pytest
import time
from unittest.mock import MagicMock, patch

from haliosai import HaliosGuard, TraceContext
from haliosai.client import ScanResult


@pytest.mark.asyncio
async def test_trace_payload_and_scan_result_metadata():
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key")
    guard._ensure_http_client_for_testing()

    messages = [{"role": "user", "content": "Hello"}]
    mock_response = {
        "guardrails_triggered": 0,
        "result": [],
        "request": {"message_count": 1, "content_length": 5},
    }

    trace_ctx = TraceContext.create(trace_id="0" * 32, conversation_id="conv-xyz")

    with patch.object(guard.http_client, "post") as mock_post:
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status.return_value = None
        mock_post.return_value = mock_response_obj

        first_result = await guard.evaluate(
            messages,
            "request",
            trace_context=trace_ctx,
            span_attributes={"custom": "meta"},
        )

        first_payload = mock_post.call_args_list[0][1]["json"]
        assert first_payload["trace"]["trace_id"] == "0" * 32
        assert first_payload["trace"]["conversation_id"] == "conv-xyz"
        assert first_payload["trace"]["span_name"] == "guardrail.evaluation.request"
        assert first_payload["trace"]["span_attributes"]["custom"] == "meta"

        # ScanResult should also expose trace metadata
        assert first_result.trace_id == "0" * 32
        assert first_result.conversation_id == "conv-xyz"
        assert first_result.span_name == "guardrail.evaluation.request"
        first_span_id = first_payload["trace"]["span_id"]
        assert first_result.span_id == first_span_id

        # Subsequent response evaluation should reuse the trace and chain spans
        await guard.evaluate(
            messages + [{"role": "assistant", "content": "Hi"}],
            "response",
        )
        second_payload = mock_post.call_args_list[1][1]["json"]
        assert second_payload["trace"]["trace_id"] == "0" * 32
        assert second_payload["trace"]["parent_span_id"] == first_span_id
        assert second_payload["trace"]["span_name"] == "guardrail.evaluation.response"


@pytest.mark.asyncio
async def test_trace_context_can_be_passed_through_decorator():
    collected_payloads = []

    async def fake_llm(messages):
        return {"choices": [{"message": {"role": "assistant", "content": "Hi"}}]}

    # Decorator wrapper will strip trace_context from kwargs before calling fake_llm
    from haliosai import guarded_chat_completion

    @guarded_chat_completion(agent_id="test-agent", api_key="test-key", concurrent_guardrail_processing=True)
    async def wrapped(messages):
        return await fake_llm(messages)

    trace_ctx = TraceContext.create(trace_id="1" * 32, conversation_id="thread-1")

    with patch("haliosai.client.HaliosGuard.evaluate") as mock_evaluate:
        async def _mock_eval(self, messages, invocation_type="request", trace_context=None, span_name=None, span_attributes=None):
            collected_payloads.append(
                {
                    "invocation_type": invocation_type,
                    "trace_context": trace_context,
                    "span_name": span_name,
                }
            )
            result = ScanResult(status="safe")
            result.trace_id = trace_context.trace_id if trace_context else None
            return result

        mock_evaluate.side_effect = _mock_eval

        await wrapped([{"role": "user", "content": "Hello"}], trace_context=trace_ctx)

    assert collected_payloads
    assert collected_payloads[0]["trace_context"] == trace_ctx


@pytest.mark.asyncio
async def test_constructor_and_context_manager_create_trace():
    # Constructor with trace_id should set trace immediately
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key", trace_id=("f"*32))
    # Passing trace_id should set the trace context immediately
    assert guard.trace_context is not None
    assert guard.trace_context.trace_id == "f"*32

    # Context manager should create a session-level trace when entering context
    guard2 = HaliosGuard(agent_id="test-agent", api_key="test-key")
    guard2._ensure_http_client_for_testing()
    assert guard2.trace_context is None
    async with guard2:
        assert guard2.trace_context is not None
        # calling evaluate should pass the same trace id
        messages = [{"role": "user", "content": "Hello"}]
        mock_response = {"guardrails_triggered": 0, "result": [], "request": {"message_count": 1}}
        with patch.object(guard2.http_client, "post") as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_post.return_value = mock_response_obj

            res = await guard2.evaluate(messages, "request")
            payload = mock_post.call_args_list[0][1]["json"]
            assert payload["trace"]["trace_id"] == guard2.trace_context.trace_id


@pytest.mark.asyncio
async def test_start_and_end_span_sets_parentage_and_payload():
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key")
    guard._ensure_http_client_for_testing()

    # Start a span and ensure it becomes the last span
    span = guard.start_span("turn.1", attributes={"turn": 1})
    assert span.span_id == guard._last_span_id

    # Build a trace payload for a request and ensure the parent is the started span
    resolved_trace, span_ctx, payload = guard._build_trace_payload("request")
    assert payload["parent_span_id"] == span.span_id

    # End the span and ensure payload contains end_time and duration
    res_payload = guard.end_span(span)
    assert "end_time" in res_payload
    assert "duration_ms" in res_payload


@pytest.mark.asyncio
async def test_span_nesting_and_hierarchy():
    """Test nested spans and proper parent-child relationships."""
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key")
    guard._ensure_http_client_for_testing()

    # Start outer span
    outer_span = guard.start_span("conversation", attributes={"type": "multi-turn"})
    assert guard._span_stack == [outer_span.span_id]

    # Start inner span
    inner_span = guard.start_span("turn.1", attributes={"turn": 1})
    assert guard._span_stack == [outer_span.span_id, inner_span.span_id]
    assert guard._last_span_id == inner_span.span_id

    # Build payload should use inner span as parent
    _, _, payload = guard._build_trace_payload("request")
    assert payload["parent_span_id"] == inner_span.span_id

    # End inner span
    inner_payload = guard.end_span(inner_span)
    assert inner_payload["span_id"] == inner_span.span_id
    assert guard._span_stack == [outer_span.span_id]
    assert guard._last_span_id == outer_span.span_id

    # Build payload should now use outer span as parent
    _, _, payload2 = guard._build_trace_payload("response")
    assert payload2["parent_span_id"] == outer_span.span_id

    # End outer span
    outer_payload = guard.end_span(outer_span)
    assert outer_payload["span_id"] == outer_span.span_id
    assert guard._span_stack == []
    assert guard._last_span_id is None


@pytest.mark.asyncio
async def test_span_attributes_and_metadata():
    """Test span attributes, timing, and metadata."""
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key")

    # Test span creation with attributes
    span = guard.start_span("test.operation", attributes={"user_id": "123", "operation": "chat"})
    assert span.attributes["user_id"] == "123"
    assert span.attributes["operation"] == "chat"
    assert span.start_time is not None
    assert span.end_time is None  # Not ended yet

    # Simulate some work
    time.sleep(0.01)

    # End span and check timing
    payload = guard.end_span(span)
    assert payload["span_id"] == span.span_id
    assert payload["span_name"] == "test.operation"
    assert payload["span_attributes"]["user_id"] == "123"  # Note: uses "span_attributes" not "attributes"
    assert payload["span_attributes"]["operation"] == "chat"
    assert "start_time" in payload
    assert "end_time" in payload
    assert payload["duration_ms"] > 0

    # Test empty attributes
    span2 = guard.start_span("empty.span")
    assert span2.attributes == {}
    payload2 = guard.end_span(span2)
    assert payload2["span_attributes"] == {}


@pytest.mark.asyncio
async def test_trace_context_precedence():
    """Test precedence rules for trace_context vs trace_id in constructor."""
    # trace_id takes precedence over trace_context in constructor
    explicit_trace = TraceContext.create(trace_id="explicit123", conversation_id="conv-1")
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key",
                       trace_context=explicit_trace, trace_id="ignored456")
    assert guard.trace_context.trace_id == "ignored456"  # trace_id overrides trace_context

    # trace_id only when no trace_context
    guard2 = HaliosGuard(agent_id="test-agent", api_key="test-key", trace_id="only-trace-id")
    assert guard2.trace_context.trace_id == "only-trace-id"

    # Neither should result in None initially
    guard3 = HaliosGuard(agent_id="test-agent", api_key="test-key")
    assert guard3.trace_context is None


@pytest.mark.asyncio
async def test_automatic_span_creation_in_evaluate():
    """Test that evaluate() automatically creates spans when no explicit span is active."""
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key")
    guard._ensure_http_client_for_testing()

    messages = [{"role": "user", "content": "Hello"}]
    mock_response = {"guardrails_triggered": 0, "result": [], "request": {"message_count": 1}}

    with patch.object(guard.http_client, "post") as mock_post:
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status.return_value = None
        mock_post.return_value = mock_response_obj

        # No active span initially
        assert guard._last_span_id is None

        # Evaluate should create automatic span
        result = await guard.evaluate(messages, "request")

        # Check payload has automatic span
        payload = mock_post.call_args_list[0][1]["json"]
        assert payload["trace"]["span_name"] == "guardrail.evaluation.request"
        assert "span_id" in payload["trace"]
        assert payload["trace"]["parent_span_id"] is None  # No parent since no active span

        # Result should have span metadata
        assert result.span_name == "guardrail.evaluation.request"
        assert result.span_id == payload["trace"]["span_id"]


@pytest.mark.asyncio
async def test_trace_propagation_across_operations():
    """Test that trace context is maintained across multiple operations."""
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key", trace_id="persistent-trace")
    guard._ensure_http_client_for_testing()

    messages = [{"role": "user", "content": "Hello"}]
    mock_response = {"guardrails_triggered": 0, "result": [], "request": {"message_count": 1}}

    with patch.object(guard.http_client, "post") as mock_post:
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status.return_value = None
        mock_post.return_value = mock_response_obj

        # First evaluation
        result1 = await guard.evaluate(messages, "request")
        payload1 = mock_post.call_args_list[0][1]["json"]
        trace_id = payload1["trace"]["trace_id"]
        span_id1 = payload1["trace"]["span_id"]

        # Second evaluation should reuse same trace
        result2 = await guard.evaluate(messages, "response")
        payload2 = mock_post.call_args_list[1][1]["json"]
        assert payload2["trace"]["trace_id"] == trace_id
        assert payload2["trace"]["parent_span_id"] == span_id1  # Chains from previous span

        # Both results should have same trace_id
        assert result1.trace_id == result2.trace_id == trace_id


@pytest.mark.asyncio
async def test_span_error_handling():
    """Test error handling for span operations."""
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key")

    # Test ending non-existent span (should not raise error, just return payload)
    fake_span = MagicMock()
    fake_span.span_id = "non-existent"
    fake_span.finish = MagicMock()
    fake_span.to_payload = MagicMock(return_value={"span_id": "non-existent"})
    payload = guard.end_span(fake_span)
    assert payload["span_id"] == "non-existent"
    fake_span.finish.assert_called_once()

    # Test ending span that's not at top of stack (should still work)
    span1 = guard.start_span("span1")
    span2 = guard.start_span("span2")
    span3 = guard.start_span("span3")

    # Try to end span1 while span3 is active (should work, just remove from stack)
    guard.end_span(span1)
    assert span1.span_id not in guard._span_stack
    assert guard._last_span_id == span3.span_id  # span3 still active

    # Should be able to end span3 (most recent)
    guard.end_span(span3)
    assert guard._last_span_id == span2.span_id

    # And finally span2
    guard.end_span(span2)
    assert guard._last_span_id is None


@pytest.mark.asyncio
async def test_concurrent_spans_simulation():
    """Test behavior with multiple spans (simulating concurrent operations)."""
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key")

    # Start multiple top-level spans (simulating concurrent operations)
    span1 = guard.start_span("operation.1")
    span1_id = span1.span_id

    # Start nested span under span1
    nested1 = guard.start_span("operation.1.subtask")
    nested1_id = nested1.span_id

    # Simulate switching to another operation (end current stack)
    guard.end_span(nested1)
    guard.end_span(span1)

    # Start new operation
    span2 = guard.start_span("operation.2")
    span2_id = span2.span_id

    # Verify spans are independent
    assert span1_id != span2_id
    assert nested1_id != span2_id

    # Each span should have proper timing
    payload1 = guard.end_span(span2)
    assert payload1["duration_ms"] >= 0
    assert payload1["span_id"] == span2_id


@pytest.mark.asyncio
async def test_trace_context_creation_edge_cases():
    """Test edge cases in trace context creation."""
    # Empty trace_id should not create a trace context (empty string is falsy)
    guard = HaliosGuard(agent_id="test-agent", api_key="test-key", trace_id="")
    assert guard.trace_context is None  # Empty string doesn't create trace

    # Valid trace_id creates trace context
    guard2 = HaliosGuard(agent_id="test-agent", api_key="test-key", trace_id="valid-trace-id")
    assert guard2.trace_context.trace_id == "valid-trace-id"

    # Very long trace_id
    long_id = "a" * 100
    guard3 = HaliosGuard(agent_id="test-agent", api_key="test-key", trace_id=long_id)
    assert guard3.trace_context.trace_id == long_id

    # TraceContext.create with minimal params
    trace = TraceContext.create()
    assert trace.trace_id is not None
    assert len(trace.trace_id) == 32  # Default length

    # TraceContext.create with explicit trace_id
    trace2 = TraceContext.create(trace_id="custom123")
    assert trace2.trace_id == "custom123"
