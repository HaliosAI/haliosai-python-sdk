"""HaliosAI SDK — guardrails, tracing, and evaluations for LLM applications.

Usage::

    from haliosai import HaliosClient, guarded

    # Client-based API
    async with HaliosClient(agent_id="my-agent") as client:
        result = await client.evaluate(messages=[...])

    # Decorator-based API
    @guarded(agent_id="my-agent")
    async def call_llm(messages):
        return await openai.chat.completions.create(model="gpt-4o", messages=messages)
"""

from __future__ import annotations

from ._version import __version__

# Core client
from .client import HaliosClient

# Exceptions
from .exceptions import (
    ConfigError,
    CohortTagValidationError,
    EvaluationError,
    GuardrailTriggered,
    HaliosAPIError,
    HaliosError,
    TimeoutError,
)

# Guardrails (decorator + helpers)
from .guardrails import (
    extract_response_message,
    guarded,
)

# Tracing
from .tracing import TracedConversation

# Evaluations
from .evaluations import EvalRun
from .cohorts import CohortCollection, CohortValidator
from .ingest import BulkEvalPusher, BulkIngester, BulkIngestResult
from .integrations import HaliosSparkEvaluator

# Types (commonly used in user code)
from .types import (
    CohortDefinition,
    CohortValidateResult,
    CheckExecutionProgress,
    CheckExecutionResponse,
    CheckResult,
    EvaluateResult,
    ExecutionResult,
    GuardedResponse,
    IngestTaskStatus,
    GuardrailPolicy,
    PaginatedResult,
    RunMetadataResult,
    SpanResponse,
    TraceDetail,
    TraceResponse,
    Violation,
    ViolationAction,
)

__all__ = [
    # Version
    "__version__",
    # Client
    "HaliosClient",
    # Exceptions
    "HaliosError",
    "ConfigError",
    "HaliosAPIError",
    "CohortTagValidationError",
    "GuardrailTriggered",
    "EvaluationError",
    "TimeoutError",
    # Guardrails
    "guarded",
    "extract_response_message",
    # Tracing
    "TracedConversation",
    # Evaluations
    "EvalRun",
    "CohortCollection",
    "CohortValidator",
    "BulkIngester",
    "BulkEvalPusher",
    "BulkIngestResult",
    "HaliosSparkEvaluator",
    # Types
    "CohortDefinition",
    "CohortValidateResult",
    "EvaluateResult",
    "Violation",
    "CheckResult",
    "GuardrailPolicy",
    "ViolationAction",
    "ExecutionResult",
    "GuardedResponse",
    "IngestTaskStatus",
    "RunMetadataResult",
    "SpanResponse",
    "TraceResponse",
    "TraceDetail",
    "CheckExecutionResponse",
    "CheckExecutionProgress",
    "PaginatedResult",
]
