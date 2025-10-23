"""
HaliosAI SDK - AI Guardrails for LLM Applications

A powerful Python SDK for integrating AI guardrails with Large Language Model applications.
Provides simple patching, parallel processing, streaming support, and multi-agent configurations.
"""

from .client import (
    HaliosGuard,
    ExecutionResult,
    GuardedResponse,
    GuardrailViolation,
    ViolationAction,
    GuardrailPolicy,
    # Main unified decorator
    guarded_chat_completion,
    # Utility functions
)
from .config import setup_logging

# OpenAI Agents Framework Integration (optional)
try:
    from .openai import (
        HaliosInputGuardrail,
        HaliosOutputGuardrail,
    )
    _OPENAI_AGENTS_AVAILABLE = True
except ImportError:
    _OPENAI_AGENTS_AVAILABLE = False

__author__ = "HaliosLabs"
__email__ = "support@halios.ai"

__all__ = [
    # Core classes
    "HaliosGuard",
    "ExecutionResult",
    "GuardedResponse",
    "GuardrailViolation",
    "ViolationAction",
    "GuardrailPolicy",
    # Main decorator (recommended)
    "guarded_chat_completion",
    # Configuration
    "setup_logging",
]

# Add OpenAI Agents Framework guardrails if available
if _OPENAI_AGENTS_AVAILABLE:
    __all__.extend([
        "HaliosInputGuardrail",
        "HaliosOutputGuardrail",
    ])
