"""Small explicit Python client for Halios guardrails and trace evaluations.

Application telemetry belongs to stock OpenTelemetry. This package makes authenticated Halios API
calls and does not install or configure an OpenTelemetry SDK.
"""

from ._version import __version__
from .client import HaliosClient as Client
from .exceptions import ConfigError, HaliosAPIError, HaliosError, HaliosTimeoutError
from .types import CheckResult, EvaluateResult, EvaluationRun, TraceDetail, Violation

__all__ = [
    "__version__",
    "Client",
    "HaliosError",
    "ConfigError",
    "HaliosAPIError",
    "HaliosTimeoutError",
    "EvaluateResult",
    "Violation",
    "CheckResult",
    "EvaluationRun",
    "TraceDetail",
]
