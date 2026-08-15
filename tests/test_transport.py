from __future__ import annotations

import httpx
import pytest

from haliosai._transport import HaliosTransport
from haliosai.exceptions import HaliosAPIError


def test_transport_normalizes_api_prefix() -> None:
    transport = HaliosTransport("https://app.halios.ai/api/v1/", "key")
    assert transport.base_url == "https://app.halios.ai"
    assert transport.api_prefix == "/api/v1"


def test_transport_exposes_structured_error() -> None:
    response = httpx.Response(
        422,
        json={"detail": "invalid manifest", "code": "INVALID"},
        request=httpx.Request("POST", "https://app.halios.ai/api/v1/runs/evaluations"),
    )
    with pytest.raises(HaliosAPIError) as caught:
        HaliosTransport._raise_api_error(response)
    assert caught.value.status_code == 422
    assert caught.value.code == "INVALID"


def test_transport_extracts_nested_usage_limit_code() -> None:
    response = httpx.Response(
        402,
        json={
            "detail": {
                "code": "usage_limit_exceeded",
                "meter": "managed_ai_tokens",
                "billing_url": "https://app.halios.ai/settings/billing",
            }
        },
        request=httpx.Request("POST", "https://app.halios.ai/api/v1/evaluations"),
    )

    with pytest.raises(HaliosAPIError) as caught:
        HaliosTransport._raise_api_error(response)

    assert caught.value.status_code == 402
    assert caught.value.code == "usage_limit_exceeded"
    assert caught.value.detail["meter"] == "managed_ai_tokens"
