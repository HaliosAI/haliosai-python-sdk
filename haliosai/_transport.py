"""HTTP transport layer with retry, backoff, and connection pooling."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from .exceptions import HaliosAPIError

logger = logging.getLogger("haliosai")

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class HaliosTransport:
    """Async + sync HTTP transport with retry and shared connection pooling.

    Notes on API-versioning / base_url:
        - The `base_url` parameter may optionally include an API-version suffix
          (for example: ``https://api.example.com/api/v1`` or
          ``https://api.example.com/api/v2``).
        - The transport will "normalize" the supplied URL by stripping any
          known API-version suffix and storing it in ``transport.base_url``.
        - The detected API prefix (``/api/v1`` or ``/api/v2``) is exposed as
          ``transport.api_prefix`` and is used when constructing request
          paths (so the SDK will call the matching server version).
        - If no suffix is present, the SDK defaults to ``/api/v1``.

    Parameters:
        base_url: HaliosAI API base URL (may include an API-version suffix).
        api_key: Bearer token for authentication.
        timeout: Request timeout in seconds.
        max_retries: Number of retry attempts on transient errors.
        backoff_factor: Multiplier for exponential backoff between retries.
        headers: Additional headers to include on every request.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        headers: dict[str, str] | None = None,
    ):
        clean = base_url.rstrip("/")
        # Allow callers to pass base_url with the API version suffix and
        # auto-detect which API prefix to use when building paths.
        detected_prefix = "/api/v1"
        for suffix in ("/api/v2", "/api/v1"):
            if clean.endswith(suffix):
                detected_prefix = suffix
                clean = clean[: -len(suffix)]
                break
        self.base_url = clean
        # Runtime API prefix used when constructing server paths (e.g. /api/v1)
        self.api_prefix = detected_prefix
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        self._default_headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "haliosai-sdk/2.0",
            "Content-Type": "application/json",
        }
        if headers:
            self._default_headers.update(headers)

        self._async_client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    # -- Async -----------------------------------------------------------

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._default_headers,
                timeout=self.timeout,
            )
        return self._async_client

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make an async HTTP request with retry logic.

        Raises :class:`HaliosAPIError` on non-retryable failures.
        """
        client = self._get_async_client()
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(
                    method, path, json=json, params=params
                )
                if response.status_code < 400:
                    return response
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    self._raise_api_error(response)
                # Retryable status — fall through to backoff
                last_exc = HaliosAPIError(
                    f"HTTP {response.status_code}",
                    status_code=response.status_code,
                    detail=response.text,
                )
            except httpx.TransportError as exc:
                last_exc = exc

            if attempt < self.max_retries:
                delay = self.backoff_factor * (2**attempt)
                logger.debug(
                    "Retrying %s %s (attempt %d/%d) in %.1fs",
                    method, path, attempt + 1, self.max_retries, delay,
                )
                import asyncio
                await asyncio.sleep(delay)

        if isinstance(last_exc, HaliosAPIError):
            raise last_exc
        raise HaliosAPIError(
            f"Request failed after {self.max_retries + 1} attempts: {last_exc}",
            status_code=None,
        )

    # -- Sync ------------------------------------------------------------

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers=self._default_headers,
                timeout=self.timeout,
            )
        return self._sync_client

    def request_sync(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make a synchronous HTTP request with retry logic."""
        client = self._get_sync_client()
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.request(method, path, json=json, params=params)
                if response.status_code < 400:
                    return response
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    self._raise_api_error(response)
                last_exc = HaliosAPIError(
                    f"HTTP {response.status_code}",
                    status_code=response.status_code,
                    detail=response.text,
                )
            except httpx.TransportError as exc:
                last_exc = exc

            if attempt < self.max_retries:
                delay = self.backoff_factor * (2**attempt)
                logger.debug(
                    "Retrying %s %s (attempt %d/%d) in %.1fs",
                    method, path, attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)

        if isinstance(last_exc, HaliosAPIError):
            raise last_exc
        raise HaliosAPIError(
            f"Request failed after {self.max_retries + 1} attempts: {last_exc}",
            status_code=None,
        )

    # -- Helpers ---------------------------------------------------------

    @staticmethod
    def _raise_api_error(response: httpx.Response) -> None:
        try:
            body = response.json()
        except Exception:
            body = None
        detail = None
        code = None
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("error")
            code = body.get("code")
        raise HaliosAPIError(
            f"HTTP {response.status_code}: {detail or response.text}",
            status_code=response.status_code,
            detail=detail,
            code=code,
            response_body=body,
        )

    # -- Lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None

    def close(self) -> None:
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
            self._sync_client = None
