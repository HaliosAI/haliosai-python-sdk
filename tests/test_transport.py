"""Tests for the HTTP transport layer."""

from __future__ import annotations

import time

import httpx
import pytest

from haliosai._transport import HaliosTransport
from haliosai.exceptions import HaliosAPIError


class TestTransportInit:
    def test_default_headers(self):
        t = HaliosTransport(base_url="https://api.test.com", api_key="key-123")
        assert t._default_headers["Authorization"] == "Bearer key-123"
        assert "haliosai-sdk" in t._default_headers["User-Agent"]

    def test_custom_headers(self):
        t = HaliosTransport(
            base_url="https://api.test.com",
            api_key="key-123",
            headers={"X-Custom": "value"},
        )
        assert t._default_headers["X-Custom"] == "value"

    def test_base_url_stripped(self):
        t = HaliosTransport(base_url="https://api.test.com/", api_key="k")
        assert t.base_url == "https://api.test.com"
        assert t.api_prefix == "/api/v1"

    def test_base_url_strips_api_v1_suffix(self):
        t = HaliosTransport(base_url="https://api.test.com/api/v1", api_key="k")
        assert t.base_url == "https://api.test.com"
        assert t.api_prefix == "/api/v1"

    def test_base_url_strips_api_v1_with_trailing_slash(self):
        t = HaliosTransport(base_url="https://api.test.com/api/v1/", api_key="k")
        assert t.base_url == "https://api.test.com"
        assert t.api_prefix == "/api/v1"

    def test_base_url_strips_api_v2_suffix(self):
        t = HaliosTransport(base_url="https://api.test.com/api/v2", api_key="k")
        assert t.base_url == "https://api.test.com"
        assert t.api_prefix == "/api/v2"


class TestTransportError:
    def test_raise_api_error_json(self):
        response = httpx.Response(
            status_code=400,
            json={"detail": "Bad request", "code": "INVALID"},
            request=httpx.Request("POST", "https://api.test.com/endpoint"),
        )
        with pytest.raises(HaliosAPIError) as exc_info:
            HaliosTransport._raise_api_error(response)
        err = exc_info.value
        assert err.status_code == 400
        assert err.detail == "Bad request"
        assert err.code == "INVALID"

    def test_raise_api_error_text(self):
        response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("GET", "https://api.test.com/fail"),
        )
        with pytest.raises(HaliosAPIError) as exc_info:
            HaliosTransport._raise_api_error(response)
        assert exc_info.value.status_code == 500


class TestTransportLifecycle:
    def test_lazy_client_creation(self):
        t = HaliosTransport(base_url="https://api.test.com", api_key="k")
        assert t._async_client is None
        assert t._sync_client is None

    @pytest.mark.asyncio
    async def test_aclose(self):
        t = HaliosTransport(base_url="https://api.test.com", api_key="k")
        # Force client creation
        _ = t._get_async_client()
        assert t._async_client is not None
        await t.aclose()
        assert t._async_client is None

    def test_close_sync(self):
        t = HaliosTransport(base_url="https://api.test.com", api_key="k")
        _ = t._get_sync_client()
        assert t._sync_client is not None
        t.close()
        assert t._sync_client is None
