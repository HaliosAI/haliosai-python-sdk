"""Tests for tracing module."""

from __future__ import annotations

import pytest

from haliosai.tracing import TracedConversation, _create_trace, _finalize_trace, _get_trace
from haliosai.types import TraceDetail, TraceResponse


class TestCreateTrace:
    @pytest.mark.asyncio
    async def test_create_trace(self, mock_transport):
        result = await _create_trace(
            mock_transport,
            agent_id="test-agent",
            tags=["prod"],
        )
        assert isinstance(result, TraceResponse)
        assert result.trace_id == "trace-001"

        call = mock_transport._calls[-1]
        assert call["method"] == "POST"
        assert call["json"]["agent_id"] == "test-agent"
        assert call["json"]["tags"] == ["prod"]
        assert "trace_id" in call["json"]  # auto-generated

    @pytest.mark.asyncio
    async def test_create_trace_with_explicit_id(self, mock_transport):
        await _create_trace(
            mock_transport,
            agent_id="test-agent",
            trace_id="custom-trace-id",
        )
        call = mock_transport._calls[-1]
        assert call["json"]["trace_id"] == "custom-trace-id"


class TestFinalizeTrace:
    @pytest.mark.asyncio
    async def test_finalize_trace(self, mock_transport):
        result = await _finalize_trace(
            mock_transport,
            trace_id="trace-001",
            trigger_evaluators=True,
            tags=["eval-run-1"],
        )
        assert isinstance(result, TraceResponse)
        call = mock_transport._calls[-1]
        assert "finalize" in call["path"]
        assert call["json"]["trigger_evaluators"] is True
        assert call["json"]["tags"] == ["eval-run-1"]


class TestGetTrace:
    @pytest.mark.asyncio
    async def test_get_trace(self, mock_transport):
        result = await _get_trace(mock_transport, trace_id="trace-001")
        assert isinstance(result, TraceDetail)
        assert result.trace_id == "trace-001"
        assert len(result.spans) == 1
        assert result.spans[0].name == "llm_call"


class TestTracedConversation:
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_client):
        async with mock_client.traced_conversation(tags=["test"]) as tc:
            assert tc.trace_id == "trace-001"
            assert isinstance(tc.trace, TraceResponse)

    @pytest.mark.asyncio
    async def test_trace_id_before_enter_raises(self, mock_client):
        tc = TracedConversation(client=mock_client)
        with pytest.raises(RuntimeError, match="not been entered"):
            _ = tc.trace_id

    @pytest.mark.asyncio
    async def test_finalize_on_exit(self, mock_client, mock_transport):
        async with mock_client.traced_conversation(
            trigger_evaluators=True, finalize_tags=["eval"]
        ):
            pass

        # Verify finalize was called
        finalize_calls = [
            c for c in mock_transport._calls if "finalize" in c["path"]
        ]
        assert len(finalize_calls) == 1
