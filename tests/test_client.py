"""Tests for HaliosClient."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from haliosai.client import HaliosClient
from haliosai.exceptions import ConfigError
from haliosai.types import EvaluateResult

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_init_with_explicit_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HALIOS_BASE_URL", None)
            client = HaliosClient(agent_id="my-agent", api_key="key-123")
            assert client.agent_id == "my-agent"
            assert client.api_key == "key-123"
            assert client.base_url == "https://api.halios.ai"

    def test_init_from_env(self):
        with patch.dict(os.environ, {"HALIOS_API_KEY": "env-key", "HALIOS_BASE_URL": "https://custom.api"}):
            client = HaliosClient(agent_id="agent-1")
            assert client.api_key == "env-key"
            assert client.base_url == "https://custom.api"

    def test_init_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HALIOS_API_KEY", None)
            with pytest.raises(ConfigError, match="api_key is required"):
                HaliosClient(agent_id="agent-1")

    def test_init_custom_timeout(self):
        client = HaliosClient(agent_id="a", api_key="k", timeout=60, max_retries=5)
        assert client.timeout == 60
        assert client.max_retries == 5

    def test_init_respects_api_version_in_base_url(self):
        client = HaliosClient(agent_id="a", api_key="k", base_url="https://api.halios.ai/api/v2")
        assert client.base_url == "https://api.halios.ai"
        assert client._transport.api_prefix == "/api/v2"

    def test_base_url_trailing_slash_stripped(self):
        client = HaliosClient(agent_id="a", api_key="k", base_url="https://api.test.com/")
        assert client.base_url == "https://api.test.com"


# ---------------------------------------------------------------------------
# Agent resolution
# ---------------------------------------------------------------------------


class TestAgentResolution:
    def test_resolve_from_init(self):
        client = HaliosClient(agent_id="default-agent", api_key="k")
        assert client._resolve_agent() == "default-agent"

    def test_resolve_override(self):
        client = HaliosClient(agent_id="default", api_key="k")
        assert client._resolve_agent("override") == "override"

    def test_resolve_missing_raises(self):
        client = HaliosClient(api_key="k")
        with pytest.raises(ConfigError, match="agent_id is required"):
            client._resolve_agent()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_pass(self, mock_client, mock_transport):
        result = await mock_client.evaluate(messages=[{"role": "user", "content": "hello"}])
        assert isinstance(result, EvaluateResult)
        assert result.triggered is False
        assert result.action == "allow"
        assert len(result.check_results) == 1

        # Check the transport was called correctly
        call = mock_transport._calls[-1]
        assert call["method"] == "POST"
        assert "/evaluate" in call["path"]
        assert call["json"]["agent_id"] == "test-agent"

    @pytest.mark.asyncio
    async def test_evaluate_with_tags(self, mock_client, mock_transport):
        await mock_client.evaluate(
            messages=[{"role": "user", "content": "test"}],
            tags=["prod", "v2"],
        )
        call = mock_transport._calls[-1]
        assert call["json"]["tags"] == ["prod", "v2"]

    @pytest.mark.asyncio
    async def test_evaluate_with_trace_id(self, mock_client, mock_transport):
        await mock_client.evaluate(
            messages=[{"role": "user", "content": "test"}],
            trace_id="my-trace",
            span_id="my-span",
        )
        call = mock_transport._calls[-1]
        assert call["json"]["trace_id"] == "my-trace"
        assert call["json"]["span_id"] == "my-span"

    def test_evaluate_sync(self, mock_client):
        result = mock_client.evaluate_sync(messages=[{"role": "user", "content": "hello"}])
        assert isinstance(result, EvaluateResult)
        assert result.triggered is False


# ---------------------------------------------------------------------------
# Evaluation triggers
# ---------------------------------------------------------------------------


class TestEvaluationTriggers:
    @pytest.mark.asyncio
    async def test_trigger_dataset_eval(self, mock_client, mock_transport):
        run = await mock_client.trigger_dataset_eval(
            dataset_id="dataset-1",
            dataset_version=2,
            check_ids=["check-1"],
            tags=["ci"],
            run_name="nightly",
            run_comment="main branch",
        )

        call = mock_transport._calls[-1]
        assert call["method"] == "POST"
        assert "/trigger-eval" in call["path"]
        assert call["json"]["agent_id"] == "test-agent"
        assert call["json"]["dataset_id"] == "dataset-1"
        assert call["json"]["dataset_version"] == 2
        assert call["json"]["check_ids"] == ["check-1"]
        assert call["json"]["tags"] == ["ci"]
        assert call["json"]["run_name"] == "nightly"
        assert call["json"]["run_comment"] == "main branch"
        assert run.run_tag == "run:test-abc"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_client):
        async with mock_client as c:
            assert c is mock_client
        # No error on close

    @pytest.mark.asyncio
    async def test_close_idempotent(self, mock_client):
        await mock_client.close()
        await mock_client.close()  # should not error
