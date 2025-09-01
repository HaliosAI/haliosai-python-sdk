"""
HaliosAI SDK - AI Guardrails for LLM Applications

A powerful Python SDK for integrating AI guardrails with Large Language Model applications.
Provides simple patching, parallel processing, streaming support, and multi-agent configurations.
"""

from .client import (
    HaliosGuard,
    ParallelGuardedChat,
    ExecutionResult,
    # Main unified decorator
    guarded_chat_completion,
    # Legacy decorators (deprecated)
    guard,
    parallel_guarded_chat,
    streaming_guarded_chat,
    # Utility functions
    patch_openai,
    patch_anthropic,
    patch_all,
)
from .config import setup_logging

# OpenAI Agents Framework Integration (optional)
try:
    from .openai import (
        RemoteInputGuardrail,
        RemoteOutputGuardrail,
        HaliosInputGuardrail,
        HaliosOutputGuardrail,
    )
    _OPENAI_AGENTS_AVAILABLE = True
except ImportError:
    _OPENAI_AGENTS_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "HaliosLabs"
__email__ = "support@halioslabs.com"

__all__ = [
    # Core classes
    "HaliosGuard",
    "ParallelGuardedChat", 
    "ExecutionResult",
    # Main decorator (recommended)
    "guarded_chat_completion",
    # Legacy decorators
    "guard",
    "parallel_guarded_chat", 
    "streaming_guarded_chat",
    # Utility functions
    "patch_openai",
    "patch_anthropic", 
    "patch_all",
    # Configuration
    "setup_logging",
]

# Add OpenAI Agents Framework guardrails if available
if _OPENAI_AGENTS_AVAILABLE:
    __all__.extend([
        "RemoteInputGuardrail",
        "RemoteOutputGuardrail", 
        "HaliosInputGuardrail",
        "HaliosOutputGuardrail",
    ])
