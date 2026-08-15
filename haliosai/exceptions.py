"""Public exceptions raised by the Halios Python client."""

from __future__ import annotations

from typing import Any


class HaliosError(Exception):
    """Base exception for Halios client failures."""


class ConfigError(HaliosError):
    """Raised when required client configuration is absent or invalid."""


class HaliosAPIError(HaliosError):
    """A non-success response or exhausted transport retry."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: Any = None,
        code: str | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.code = code
        self.response_body = response_body


class HaliosTimeoutError(HaliosError):
    """Raised when a bounded Halios operation does not become terminal in time."""
