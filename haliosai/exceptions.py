"""HaliosAI SDK exceptions."""

from __future__ import annotations

from typing import Any


class HaliosError(Exception):
    """Base exception for all HaliosAI SDK errors."""


class ConfigError(HaliosError):
    """Raised when SDK configuration is invalid or missing."""


class HaliosAPIError(HaliosError):
    """Raised when the HaliosAI API returns an error response.

    Attributes:
        status_code: HTTP status code from the API.
        detail: Optional detail message from the API.
        code: Optional error code from the API.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str | None = None,
        code: str | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        self.status_code = status_code
        self.detail = detail
        self.code = code
        self.response_body = response_body
        super().__init__(message)


class GuardrailTriggered(HaliosError):
    """Raised when a guardrail check blocks content.

    Attributes:
        violation_type: ``"request"`` or ``"response"``.
        violations: List of :class:`~haliosai.types.Violation` objects.
        action: The action decided by the guardrail (e.g. ``"block"``).
    """

    def __init__(
        self,
        message: str,
        *,
        violation_type: str = "request",
        violations: list[Any] | None = None,
        action: str = "block",
        scan_result: Any | None = None,
    ):
        self.violation_type = violation_type
        self.violations = violations or []
        self.action = action
        self.scan_result = scan_result
        super().__init__(message)


class EvaluationError(HaliosError):
    """Raised when an evaluation run fails."""


class TimeoutError(HaliosError):
    """Raised when an operation times out."""


class CohortTagValidationError(HaliosError):
    """Raised when a cohort/tag validation check fails."""

    def __init__(self, message: str, *, reasons: list[str] | None = None):
        self.reasons = reasons or []
        super().__init__(message)
