"""
HaliosAI SDK - Core Client Module

This module provides the main HaliosGuard class and supporting utilities for
integrating AI guardrails with LLM applications.

CRITICAL REQUIREMENT: All message parameters must be in OpenAI-compatible format:
[
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]

Each message must have:
- "role": One of "system", "user", "assistant", or "tool"
- "content": The message text content
- Optional fields like "name", "tool_calls", etc. are also supported
"""

import asyncio
import httpx
import os
import time
import logging
import inspect
import warnings
from typing import List, Dict, Any, Callable, Optional, Union
from functools import wraps
from enum import Enum
from dataclasses import dataclass, field


# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# HTTP CLIENT POOL MANAGEMENT
# =============================================================================
# The SDK maintains a pool of HTTP clients to improve performance and reduce
# connection overhead. Clients are reused across requests to the same base URL.
#
# Benefits:
# - Reduced connection latency for subsequent requests
# - Better resource utilization
# - Automatic cleanup on application shutdown
# - Thread-safe client management
# =============================================================================

# Module-level HTTP client pool for reuse
_http_client_pool: Dict[str, httpx.AsyncClient] = {}
_http_client_pool_lock = asyncio.Lock()


async def _get_shared_http_client(base_url: str, timeout: float = 30.0) -> httpx.AsyncClient:
    """
    Get or create a shared HTTP client for the given base URL

    This function implements a connection pool pattern to reuse HTTP clients
    across multiple requests to the same base URL, improving performance
    by reducing connection overhead.

    Args:
        base_url: The base URL for which to get/create a client
        timeout: Request timeout in seconds (default: 30.0)

    Returns:
        httpx.AsyncClient instance configured for the given base URL

    Note:
        Clients are automatically cleaned up when the module is unloaded
        or when _cleanup_http_client_pool() is called explicitly.
    """
    async with _http_client_pool_lock:
        if base_url not in _http_client_pool:
            _http_client_pool[base_url] = httpx.AsyncClient(
                base_url=base_url,
                timeout=timeout
            )
            logger.debug(f"Created shared HTTP client for {base_url}")
        return _http_client_pool[base_url]


async def _cleanup_http_client_pool():
    """
    Clean up all shared HTTP clients

    This function should be called during application shutdown to properly
    close all HTTP connections and free resources. It's automatically called
    when the module is unloaded, but can be called manually for explicit cleanup.

    Note:
        After cleanup, new requests will create fresh HTTP clients as needed.
    """
    async with _http_client_pool_lock:
        for client in _http_client_pool.values():
            await client.aclose()
        _http_client_pool.clear()
        logger.debug("Cleaned up HTTP client pool")


class ExecutionResult(Enum):
    """Execution result status codes"""
    SUCCESS = "success"
    REQUEST_BLOCKED = "request_blocked"
    RESPONSE_BLOCKED = "response_blocked"
    TIMEOUT = "timeout"
    ERROR = "error"


class ViolationAction(Enum):
    """Enum representing different actions to take when violations are detected"""
    PASS = "pass"                      # No violations detected, request is clean
    BLOCK = "block"                    # Violations detected, request is blocked
    ALLOW_OVERRIDE = "allow_override"  # Violations detected but allowed (by policy, sanitization, or default permissive behavior)


class GuardrailPolicy(Enum):
    """Enum representing guardrail policy actions for specific guardrail types"""
    RECORD_ONLY = "record_only"        # Log violations but allow request to proceed
    BLOCK = "block"                    # Block request when violations are detected


@dataclass
class Violation:
    """Structured representation of a single guardrail violation"""
    guardrail_type: str
    guardrail_uuid: Optional[str] = None
    analysis: Optional[Dict] = None
    execution_time_ms: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    triggered: bool = True

    @classmethod
    def from_dict(cls, data: Dict) -> "Violation":
        """Create Violation from dictionary representation"""
        return cls(
            guardrail_type=data.get("guardrail_type", data.get("type", "unknown")),
            guardrail_uuid=data.get("guardrail_uuid", "unknown"),
            analysis=data.get("analysis", {}),
            execution_time_ms=data.get("execution_time_ms"),
            errors=data.get("errors", []),
            timestamp=data.get("timestamp"),
            triggered=data.get("triggered", True)
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for backward compatibility"""
        return {
            "guardrail_type": self.guardrail_type,
            "guardrail_uuid": self.guardrail_uuid,
            "analysis": self.analysis,
            "execution_time_ms": self.execution_time_ms,
            "errors": self.errors,
            "timestamp": self.timestamp,
            "triggered": self.triggered
        }


@dataclass
class ScanResult:
    """Detailed result from guardrail scanning with all available metadata"""
    status: str  # "safe", "blocked: {types}", or "error: {message}"
    response_id: Optional[str] = None
    request_id: Optional[str] = None
    content_hash: Optional[str] = None
    guardrails_applied: Optional[int] = None
    guardrails_triggered: Optional[int] = None
    processing_time_ms: Optional[float] = None
    timestamp: Optional[str] = None
    violations: List[Violation] = field(default_factory=list)
    agent_slug: Optional[str] = None
    agent_uuid: Optional[str] = None
    processing_mode: Optional[str] = None
    content_length: Optional[int] = None
    guardrails_configured: Optional[List[str]] = None
    message_count: Optional[int] = None

    def __post_init__(self):
        pass  # No longer need to initialize violations list

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame storage"""
        return {
            "status": self.status,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "content_hash": self.content_hash,
            "guardrails_applied": self.guardrails_applied,
            "guardrails_triggered": self.guardrails_triggered,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "agent_slug": self.agent_slug,
            "agent_uuid": self.agent_uuid,
            "processing_mode": self.processing_mode,
            "content_length": self.content_length,
            "guardrails_configured": self.guardrails_configured,
            "message_count": self.message_count,
            "violations": [v.to_dict() for v in self.violations]
        }

    def get(self, key: str, default=None):
        """Dict-like access for backward compatibility"""
        return getattr(self, key, default)

    @classmethod
    def from_evaluation_response(cls, evaluation_result: Dict, status: str) -> "ScanResult":
        """Create ScanResult from evaluation API response"""
        request_meta = evaluation_result.get("request", {})

        return cls(
            status=status,
            response_id=evaluation_result.get("response_id"),
            request_id=request_meta.get("request_id"),
            content_hash=evaluation_result.get("content_hash"),
            guardrails_applied=evaluation_result.get("guardrails_applied"),
            guardrails_triggered=evaluation_result.get("guardrails_triggered"),
            processing_time_ms=evaluation_result.get("processing_time_ms"),
            timestamp=evaluation_result.get("timestamp"),
            agent_slug=request_meta.get("agent_slug"),
            agent_uuid=request_meta.get("agent_uuid"),
            processing_mode=request_meta.get("processing_mode"),
            content_length=request_meta.get("content_length"),
            guardrails_configured=request_meta.get("guardrails_configured", []),
            message_count=request_meta.get("message_count"),
            violations=[
                Violation(
                    guardrail_uuid=r.get("guardrail_uuid"),
                    guardrail_type=r.get("guardrail_type"),
                    triggered=r.get("triggered", False),
                    analysis=r.get("analysis"),
                    execution_time_ms=r.get("execution_time_ms"),
                    errors=r.get("errors", []),
                    timestamp=r.get("timestamp")
                )
                for r in evaluation_result.get("result", [])
                if r.get("triggered", False)
            ]
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "ScanResult":
        """Create ScanResult from dictionary (for backward compatibility)"""
        violations_dict = data.get("violations", [])
        violations = [Violation.from_dict(v) if isinstance(v, dict) else v for v in violations_dict]

        return cls(
            status=data.get("status", "unknown"),
            response_id=data.get("response_id"),
            request_id=data.get("request_id"),
            content_hash=data.get("content_hash"),
            guardrails_applied=data.get("guardrails_applied"),
            guardrails_triggered=data.get("guardrails_triggered"),
            processing_time_ms=data.get("processing_time_ms"),
            timestamp=data.get("timestamp"),
            violations=violations,
            agent_slug=data.get("agent_slug"),
            agent_uuid=data.get("agent_uuid"),
            processing_mode=data.get("processing_mode"),
            content_length=data.get("content_length"),
            guardrails_configured=data.get("guardrails_configured", []),
            message_count=data.get("message_count")
        )


class GuardrailViolation(Exception):
    """Raised when guardrails block content"""

    def __init__(self,
                 violation_type: str,  # "request" or "response"
                 violations: List[Violation],
                 blocked_content: Optional[str] = None,
                 timing: Optional[Dict] = None,
                 scan_result: Optional["ScanResult"] = None):
        self.violation_type = violation_type  # "request" or "response"
        self.violations = violations
        self.blocked_content = blocked_content
        self.timing = timing or {}
        self.scan_result = scan_result  # Full context available

        # Create user-friendly message
        guardrail_types = [v.guardrail_type for v in violations]
        super().__init__(f"Content blocked by {', '.join(guardrail_types)}")


@dataclass
class GuardedResponse:
    """Response object containing execution results and metadata"""
    result: ExecutionResult
    final_response: Optional[Any] = None
    original_response: Optional[str] = None
    request_violations: List[Violation] = field(default_factory=list)
    response_violations: List[Violation] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


class HaliosGuard:
    """
    Unified HaliosAI guardrails client for LLM applications

    IMPORTANT: All message parameters must be in OpenAI-compatible format:
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    This class provides comprehensive AI guardrails with multiple integration patterns:

    Integration Patterns:
    1. Decorator Pattern (Recommended):
       @guarded_chat_completion(agent_id="your-agent")
       async def call_llm(messages): ...

    2. Context Manager Pattern:
       async with HaliosGuard(agent_id="your-agent") as guard:
           # Manual evaluation
           guard.evaluate(messages, "request")

    3. Direct Method Calls:
       guard = HaliosGuard(agent_id="your-agent")
       result = await guard.evaluate_request(messages)

    Features:
    - Sequential and parallel processing modes
    - Streaming with real-time guardrail evaluation
    - Context manager pattern for resource management
    - Direct evaluation methods for custom integrations
    - Function patching utilities (deprecated)

    Processing Modes:
    - Parallel: Guardrails run concurrently with LLM calls (faster)
    - Sequential: Guardrails complete before LLM calls (safer)

    Note:
        For new integrations, prefer the @guarded_chat_completion decorator
        over direct HaliosGuard instantiation for better maintainability.
    """

    def __init__(self, agent_id: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 parallel: bool = False, streaming: bool = False,
                 stream_buffer_size: Optional[int] = None, stream_check_interval: Optional[float] = None,
                 guardrail_timeout: float = 5.0, http_client: Optional[httpx.AsyncClient] = None,
                 guardrail_policies: Optional[Dict[str, GuardrailPolicy]] = None):
        """
        Initialize unified HaliosGuard with comprehensive configuration options

        This constructor sets up the guardrail client with all necessary configuration
        for both parallel and sequential processing modes, streaming support, and
        performance tuning options.

        Args:
            agent_id: Unique identifier for your HaliosAI agent configuration
            api_key: HaliosAI API key (defaults to HALIOS_API_KEY environment variable)
            base_url: HaliosAI API base URL (defaults to HALIOS_BASE_URL or https://api.halios.ai)
            parallel: Enable parallel processing (guardrails run concurrently with LLM calls)
                     - True: Faster execution, guardrails don't block LLM calls
                     - False: Safer execution, guardrails complete before LLM calls
            streaming: Enable real-time streaming guardrail evaluation
            stream_buffer_size: Optional character buffer size threshold for guardrail evaluation
                              If specified alone, only character-based checking is used
                              If not specified and stream_check_interval is specified, disabled
                              If neither specified, defaults to 50 characters
            stream_check_interval: Optional time interval (seconds) for guardrail checks
                                  If specified alone, only time-based checking is used
                                  If not specified and stream_buffer_size is specified, disabled
                                  If neither specified, defaults to 0.5 seconds
                                  Note: When both specified, evaluation triggers when EITHER condition is met
            guardrail_timeout: Maximum time to wait for guardrail evaluation (seconds)
            http_client: Optional pre-configured HTTP client (uses shared pool by default)
            guardrail_policies: Optional dict mapping guardrail types to actions
                              Actions: GuardrailPolicy.RECORD_ONLY, GuardrailPolicy.BLOCK
                              Example: {"sensitive-data": GuardrailPolicy.BLOCK, "hate-speech": GuardrailPolicy.RECORD_ONLY}

        Configuration Examples:
            # Basic setup with environment variables
            guard = HaliosGuard(agent_id="your-agent")

            # Parallel processing (recommended for performance)
            guard = HaliosGuard(agent_id="your-agent", parallel=True)

            # Streaming with custom buffer size
            guard = HaliosGuard(
                agent_id="your-agent",
                streaming=True,
                stream_buffer_size=100
            )

        Note:
            HTTP clients are managed automatically via a shared connection pool
            for optimal performance and resource utilization.
        """
        self.agent_id = agent_id
        self.api_key = api_key or os.getenv("HALIOS_API_KEY")
        self.base_url = base_url or os.getenv("HALIOS_BASE_URL", "https://api.halios.ai")
        self.parallel = parallel
        self.streaming = streaming
        
        # Handle either/or configuration for streaming buffer
        # If neither specified, use defaults for both
        # If only one specified, disable the other
        if stream_buffer_size is None and stream_check_interval is None:
            # Neither specified - use defaults for both
            self.stream_buffer_size = 50
            self.stream_check_interval = 0.5
        elif stream_buffer_size is not None and stream_check_interval is None:
            # Only buffer size specified - disable time-based checking
            self.stream_buffer_size = stream_buffer_size
            self.stream_check_interval = None
        elif stream_buffer_size is None and stream_check_interval is not None:
            # Only time interval specified - disable character-based checking
            self.stream_buffer_size = None
            self.stream_check_interval = stream_check_interval
        else:
            # Both specified - use both (hybrid approach)
            self.stream_buffer_size = stream_buffer_size
            self.stream_check_interval = stream_check_interval
        
        self.guardrail_timeout = guardrail_timeout
        # Validate guardrail policies
        if guardrail_policies:
            for guardrail_type, policy in guardrail_policies.items():
                if not isinstance(policy, GuardrailPolicy):
                    raise ValueError(f"Invalid guardrail policy for '{guardrail_type}': {policy}. "
                                   f"Must be a GuardrailPolicy enum value (RECORD_ONLY or BLOCK)")

        self.guardrail_policies = guardrail_policies or {}  # Default empty dict

        # HTTP client management - uses shared pool for connection reuse
        self.http_client = http_client  # Will be initialized lazily from shared pool
        self._http_client_base_url = base_url or os.getenv("HALIOS_BASE_URL", "https://api.halios.ai")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Lazily get or create shared HTTP client"""
        if self.http_client is None:
            self.http_client = await _get_shared_http_client(self._http_client_base_url, 30.0)
        return self.http_client

    async def evaluate(self, messages: List[Dict], invocation_type: str = "request") -> ScanResult:
        """
        Evaluate messages against configured guardrails

        CRITICAL: Messages must be in OpenAI-compatible format:
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        This is the core guardrail evaluation method that sends conversation messages
        to the HaliosAI API for analysis. It supports both pre-call ("request") and
        post-call ("response") evaluation modes.

        Args:
            messages: List of conversation messages in OpenAI format
                     [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
            invocation_type: Type of evaluation
                           - "request": Pre-call evaluation (before LLM response)
                           - "response": Post-call evaluation (after LLM response)

        Returns:
            ScanResult containing detailed evaluation results with all metadata

        Usage Examples:
            # Pre-call evaluation (recommended for safety)
            result = await guard.evaluate([{"role": "user", "content": "Hello"}], "request")
            print(f"Response ID: {result.response_id}")
            print(f"Violations: {len(result.violations)}")

            # Post-call evaluation (for response validation)
            result = await guard.evaluate(
                [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
                "response"
            )

        Note:
            This method automatically handles authentication, HTTP client management,
            and error handling. It will raise exceptions for network errors or API
            failures, so wrap calls in try/except blocks for production use.
        """
        logger.debug(f"Evaluating {len(messages)} messages with type={invocation_type}")

        if not self.api_key:
            logger.warning("No API key provided. Set HALIOS_API_KEY environment variable or pass api_key parameter")
            raise ValueError("API key is required for guardrail evaluation")

        url = f"{self.base_url}/api/v3/agents/{self.agent_id}/evaluate"

        payload = {
            "messages": messages,
            "invocation_type": invocation_type
        }

        headers = {
            "X-HALIOS-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        try:
            http_client = await self._get_http_client()
            response = await http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            api_result = response.json()

            logger.debug(f"Guardrail evaluation completed: {api_result.get('guardrails_triggered', 0)} triggered")

            # Create ScanResult from API response
            # For evaluate method, we don't have a simple status string, so we'll determine it
            triggered_count = api_result.get('guardrails_triggered', 0)
            if triggered_count > 0:
                # Get violation types from results
                results = api_result.get('result', [])
                violation_types = [r.get('guardrail_type', 'unknown') for r in results if r.get('triggered', False)]
                status = f"blocked: {', '.join(violation_types)}" if violation_types else "safe"
            else:
                status = "safe"

            return ScanResult.from_evaluation_response(api_result, status)

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during guardrail evaluation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during guardrail evaluation: {e}")
            raise

    def extract_messages(self, *args, **kwargs) -> List[Dict]:
        """
        Extract messages from function arguments

        Supports common patterns for passing messages to LLM functions.
        """
        # Look for 'messages' in kwargs
        if 'messages' in kwargs:
            messages = kwargs['messages']
            logger.debug(f"Extracted {len(messages)} messages from kwargs['messages']")
            return messages

        # Look for messages in first positional arg (common pattern)
        if args and isinstance(args[0], list):
            # Check if it looks like a messages list
            if all(isinstance(msg, dict) and 'role' in msg for msg in args[0]):
                messages = args[0]
                logger.debug(f"Extracted {len(messages)} messages from first positional arg")
                return messages

        # Look for common prompt fields
        for field in ['prompt', 'input', 'text']:
            if field in kwargs:
                logger.debug(f"Extracted message from {field} field")
                return [{"role": "user", "content": kwargs[field]}]

        # Look for string in first positional arg
        if args and isinstance(args[0], str):
            logger.debug("Extracted message from first string argument")
            return [{"role": "user", "content": args[0]}]

        logger.warning("No messages found in function arguments")
        return []

    def extract_response_message(self, response: Any) -> Dict:
        """
        Extract full message structure from LLM response including tool calls

        Handles various response formats from different LLM providers.
        """
        # Handle OpenAI-style response
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message'):
                message = response.choices[0].message

                # Build message dict with all relevant fields
                message_dict = {
                    "role": "assistant",
                    "content": message.content
                }

                # Add tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    message_dict["tool_calls"] = []
                    for tc in message.tool_calls:
                        tool_call_dict = {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        message_dict["tool_calls"].append(tool_call_dict)

                logger.debug(f"Extracted full message: content={len(str(message.content or '')) } chars, tool_calls={len(message_dict.get('tool_calls', []))} calls")
                return message_dict

            if hasattr(response.choices[0], 'text'):
                content = response.choices[0].text
                logger.debug(f"Extracted text response: {len(content)} chars")
                return {"role": "assistant", "content": content}

        # Handle dict response
        if isinstance(response, dict):
            if 'choices' in response:
                choice = response['choices'][0]
                message = choice.get('message', {})

                message_dict = {
                    "role": "assistant",
                    "content": message.get('content')
                }

                # Add tool calls if present
                if 'tool_calls' in message and message['tool_calls']:
                    message_dict["tool_calls"] = message['tool_calls']

                logger.debug(f"Extracted dict message: content={len(str(message.get('content', '')))} chars, tool_calls={len(message_dict.get('tool_calls', []))} calls")
                return message_dict

            if 'output' in response:
                content = response['output']
                logger.debug(f"Extracted output field: {len(content)} chars")
                return {"role": "assistant", "content": content}

            if 'text' in response:
                content = response['text']
                logger.debug(f"Extracted text field: {len(content)} chars")
                return {"role": "assistant", "content": content}

        # Handle string response
        if isinstance(response, str):
            logger.debug(f"Using string response directly: {len(response)} chars")
            return {"role": "assistant", "content": response}

        # Fallback to string conversion
        content = str(response)
        logger.debug(f"Fallback string conversion: {len(content)} chars")
        return {"role": "assistant", "content": content}

    def extract_response_content(self, response: Any) -> str:
        """
        Extract content from LLM response

        Handles various response formats from different LLM providers.
        Uses extract_response_message internally for consistent tool call handling.
        """
        message_dict = self.extract_response_message(response)
        content = message_dict.get('content', '')
        
        # If there are tool calls but no content, create a description
        tool_calls = message_dict.get('tool_calls', [])
        if tool_calls and not content:
            tool_names = [tc.get('function', {}).get('name', 'unknown') for tc in tool_calls]
            content = f"Assistant called tools: {', '.join(tool_names)}"
        
        logger.debug(f"Extracted content from response message: {len(content)} chars")
        return content

    async def check_violations(self, guardrail_result: Union[Dict, ScanResult]) -> tuple[ViolationAction, List[Violation]]:
        """
        Check guardrail violations and determine appropriate action

        Args:
            guardrail_result: Result from guardrail evaluation (dict or ScanResult)

        Returns:
            Tuple of (action, violations_list)
        """
        if not guardrail_result:
            return (ViolationAction.PASS, [])

        # Handle ScanResult input
        if isinstance(guardrail_result, ScanResult):
            # Check if any guardrails were triggered
            guardrails_triggered = guardrail_result.guardrails_triggered or 0
            if guardrails_triggered > 0 and guardrail_result.violations:
                violations = guardrail_result.violations
                
                # Check guardrail policies for each violation type
                actions = []
                for violation in violations:
                    guardrail_type = violation.guardrail_type
                    policy_action = self.guardrail_policies.get(guardrail_type)

                    if policy_action == GuardrailPolicy.RECORD_ONLY:
                        actions.append(ViolationAction.ALLOW_OVERRIDE)
                    elif policy_action == GuardrailPolicy.BLOCK:
                        actions.append(ViolationAction.BLOCK)
                    else:
                        # No policy specified, block by default
                        actions.append(ViolationAction.BLOCK)

                # Determine overall action: BLOCK takes precedence over ALLOW_OVERRIDE
                if ViolationAction.BLOCK in actions:
                    final_action = ViolationAction.BLOCK
                else:
                    final_action = ViolationAction.ALLOW_OVERRIDE

                if final_action == ViolationAction.BLOCK:
                    # Format violation details for logging
                    violation_details = []
                    for v in violations:
                        analysis = v.analysis or {}
                        detail = f"{v.guardrail_type}"
                        if 'explanation' in analysis and analysis['explanation']:
                            detail += f": {analysis['explanation']}"
                        elif 'detected_topics' in analysis and analysis['detected_topics']:
                            detail += f": detected {', '.join(analysis['detected_topics'])}"
                        elif analysis.get('flagged'):
                            detail += ": content flagged as potentially harmful"
                        violation_details.append(detail)

                    violation_summary = "; ".join(violation_details)
                    logger.warning("Blocking guardrail violations detected: %s", violation_summary)
                    return (ViolationAction.BLOCK, violations)
                else:
                    # ALLOW_OVERRIDE - violations present but allowed
                    logger.debug("Guardrail violations detected but allowed: %s",
                               [v.guardrail_type for v in violations])
                    return (ViolationAction.ALLOW_OVERRIDE, violations)

            return (ViolationAction.PASS, [])

        # Legacy dict-based result handling
        # Check for modified_messages - allow if present (content was sanitized)
        if 'modified_messages' in guardrail_result:
            logger.debug("Modified messages present - allowing request to pass with sanitized content")
            return (ViolationAction.ALLOW_OVERRIDE, [])

        # Check if any guardrails were triggered
        guardrails_triggered = guardrail_result.get('guardrails_triggered', 0)
        if guardrails_triggered > 0:
            # Find the specific violations and convert to Violation objects
            violations = []
            results = guardrail_result.get('result', [])
            for result in results:
                if result.get('triggered', False):
                    violations.append(Violation.from_dict(result))

            if violations:
                # Check guardrail policies for each violation type
                actions = []
                for violation in violations:
                    guardrail_type = violation.guardrail_type
                    policy_action = self.guardrail_policies.get(guardrail_type)

                    if policy_action == GuardrailPolicy.RECORD_ONLY:
                        actions.append(ViolationAction.ALLOW_OVERRIDE)
                    elif policy_action == GuardrailPolicy.BLOCK:
                        actions.append(ViolationAction.BLOCK)
                    else:
                        # No policy specified, block by default
                        actions.append(ViolationAction.BLOCK)

                # Determine overall action: BLOCK takes precedence over ALLOW_OVERRIDE
                if ViolationAction.BLOCK in actions:
                    final_action = ViolationAction.BLOCK
                else:
                    final_action = ViolationAction.ALLOW_OVERRIDE

                if final_action == ViolationAction.BLOCK:
                    # Format violation details for logging
                    violation_details = []
                    for v in violations:
                        analysis = v.analysis or {}
                        detail = f"{v.guardrail_type}"
                        if 'explanation' in analysis and analysis['explanation']:
                            detail += f": {analysis['explanation']}"
                        elif 'detected_topics' in analysis and analysis['detected_topics']:
                            detail += f": detected {', '.join(analysis['detected_topics'])}"
                        elif analysis.get('flagged'):
                            detail += ": content flagged as potentially harmful"
                        violation_details.append(detail)

                    violation_summary = "; ".join(violation_details)
                    logger.warning("Blocking guardrail violations detected: %s", violation_summary)
                    return (ViolationAction.BLOCK, violations)
                else:
                    # ALLOW_OVERRIDE - violations present but allowed
                    logger.debug("Guardrail violations detected but allowed: %s",
                               [v.guardrail_type for v in violations])
                    return (ViolationAction.ALLOW_OVERRIDE, violations)

        return (ViolationAction.PASS, [])

    async def validate_request(self, messages: List[Dict]) -> Union[Dict, ScanResult]:
        """
        Validate request messages against guardrails and raise exception if blocked

        This is a convenience method that combines evaluate() and check_violations()
        into a single call that automatically raises GuardrailViolation for blocked content.

        Args:
            messages: List of conversation messages in OpenAI format

        Returns:
            Dict containing evaluation results (only returned if request passes)

        Raises:
            GuardrailViolation: If guardrails block the request

        Usage:
            # Simple validation - throws exception if blocked
            await guard.validate_request(messages)

            # With custom handling
            try:
                result = await guard.validate_request(messages)
                # Request passed, continue with LLM call
            except GuardrailViolation as e:
                # Handle blocked request
                print(f"Request blocked: {e}")
        """
        evaluation_result = await self.evaluate(messages, "request")
        action, violations = await self.check_violations(evaluation_result)

        if action == ViolationAction.BLOCK:
            raise GuardrailViolation(
                violation_type="request",
                violations=violations,
                blocked_content=str(messages),
                scan_result=evaluation_result if isinstance(evaluation_result, ScanResult) else None
            )

        return evaluation_result

    async def validate_response(self, messages: List[Dict]) -> Union[Dict, ScanResult]:
        """
        Validate response messages against guardrails and raise exception if blocked

        This is a convenience method that combines evaluate() and check_violations()
        into a single call that automatically raises GuardrailViolation for blocked content.

        Args:
            messages: List of conversation messages in OpenAI format (including assistant response)

        Returns:
            Dict containing evaluation results (only returned if response passes)

        Raises:
            GuardrailViolation: If guardrails block the response

        Usage:
            # Simple validation - throws exception if blocked
            await guard.validate_response(messages)

            # With custom handling
            try:
                result = await guard.validate_response(messages)
                # Response passed
            except GuardrailViolation as e:
                # Handle blocked response
                print(f"Response blocked: {e}")
        """
        evaluation_result = await self.evaluate(messages, "response")
        action, violations = await self.check_violations(evaluation_result)

        if action == ViolationAction.BLOCK:
            raise GuardrailViolation(
                violation_type="response",
                violations=violations,
                blocked_content=str(messages),
                scan_result=evaluation_result if isinstance(evaluation_result, ScanResult) else None
            )

        return evaluation_result

    async def guarded_call_parallel(self, messages: List[Dict], llm_func: Callable,
                                   *args, **kwargs) -> GuardedResponse:
        """
        Perform guarded LLM call with parallel processing optimization

        Args:
            messages: Chat messages for guardrail evaluation
            llm_func: Async function that makes the LLM call
            *args, **kwargs: Arguments to pass to llm_func

        Returns:
            GuardedResponse with detailed timing and violation information
        """
        start_time = time.time()
        logger.debug("Starting parallel guarded call")

        # Create tasks for parallel execution
        request_guardrails_task = asyncio.create_task(
            self.evaluate(messages, "request"),
            name="request_guardrails"
        )
        llm_task = asyncio.create_task(
            llm_func(messages, *args, **kwargs),
            name="llm_call"
        )

        request_start = time.time()
        llm_start = time.time()

        # Variables to track completion
        request_evaluation = None
        llm_response = None
        request_guardrails_done = False
        llm_done = False
        request_time = 0.0
        llm_time = 0.0

        # Wait for tasks to complete, handling whichever finishes first
        pending = {request_guardrails_task, llm_task}

        try:
            while pending:
                # Wait for at least one task to complete
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self.guardrail_timeout
                )

                if not done:
                    # Timeout occurred
                    logger.warning("Operations timed out after %ss", self.guardrail_timeout)
                    for task in pending:
                        task.cancel()

                    return GuardedResponse(
                        result=ExecutionResult.TIMEOUT,
                        error_message=f"Operations timed out after {self.guardrail_timeout}s",
                        timing={
                            "total_time": time.time() - start_time,
                            "timeout": self.guardrail_timeout
                        }
                    )

                # Process completed tasks
                for task in done:
                    task_name = task.get_name()

                    try:
                        result = await task

                        if task_name == "request_guardrails":
                            request_evaluation = result
                            request_guardrails_done = True
                            request_time = time.time() - request_start

                            # Check for violations immediately
                            action, violations = await self.check_violations(request_evaluation)
                            if action == ViolationAction.BLOCK:
                                # Cancel LLM task if still running
                                if not llm_done and llm_task in pending:
                                    llm_task.cancel()
                                    pending.discard(llm_task)
                                    logger.debug("Cancelled LLM task due to request guardrail violations")

                                logger.warning(f"Request blocked: {len(violations)} violations detected")
                                return GuardedResponse(
                                    result=ExecutionResult.REQUEST_BLOCKED,
                                    request_violations=violations,
                                    timing={
                                        "request_guardrails_time": request_time,
                                        "total_time": time.time() - start_time
                                    }
                                )

                        elif task_name == "llm_call":
                            llm_response = result
                            llm_done = True
                            llm_time = time.time() - llm_start
                            logger.debug("LLM call completed in %.3fs", llm_time)

                    except asyncio.CancelledError:
                        logger.debug("Task %s was cancelled", task_name)
                        pass  # Expected when cancelling tasks
                    except Exception as e:
                        logger.error("%s failed: %s", task_name, e)
                        return GuardedResponse(
                            result=ExecutionResult.ERROR,
                            error_message="%s failed: %s" % (task_name, str(e)),
                            timing={"total_time": time.time() - start_time}
                        )

            # If we get here, both tasks completed successfully
            # Now evaluate response guardrails synchronously
            logger.debug("Evaluating response guardrails")
            response_start = time.time()

            # Extract response message for guardrail evaluation (includes tool calls)
            response_message = self.extract_response_message(llm_response)
            full_conversation = messages + [response_message]
            response_evaluation = await self.evaluate(full_conversation, "response")

            response_time = time.time() - response_start

            # Check for response violations
            action, violations = await self.check_violations(response_evaluation)
            if action == ViolationAction.BLOCK:
                logger.warning(f"Response blocked: {len(violations)} violations detected")
                return GuardedResponse(
                    result=ExecutionResult.RESPONSE_BLOCKED,
                    original_response=response_message.get("content", ""),
                    response_violations=violations,
                    timing={
                        "request_guardrails_time": request_time,
                        "llm_time": llm_time,
                        "response_guardrails_time": response_time,
                        "total_time": time.time() - start_time
                    }
                )

            # Check if response was modified
            processed_messages = response_evaluation.get("processed_messages", [])
            final_response = llm_response  # Return full response object, not just content

            if processed_messages:
                assistant_msg = next(
                    (msg for msg in reversed(processed_messages) if msg.get("role") == "assistant"),
                    None
                )
                if assistant_msg and assistant_msg.get("content") != response_message.get("content"):
                    final_response = assistant_msg["content"]  # Only return text if modified
                    logger.debug("Response was modified by guardrails")

            total_time = time.time() - start_time
            parallel_savings = max(0, request_time + llm_time - total_time)

            logger.debug(f"Parallel guarded call completed successfully in {total_time:.3f}s (saved {parallel_savings:.3f}s)")

            return GuardedResponse(
                result=ExecutionResult.SUCCESS,
                final_response=final_response,
                original_response=response_message.get("content", ""),
                timing={
                    "request_guardrails_time": request_time,
                    "llm_time": llm_time,
                    "response_guardrails_time": response_time,
                    "total_time": total_time,
                    "parallel_savings": parallel_savings
                }
            )

        except Exception as e:
            # Cancel any remaining tasks
            for task in pending:
                task.cancel()

            logger.error("Parallel guarded call failed: %s", e)
            return GuardedResponse(
                result=ExecutionResult.ERROR,
                error_message=str(e),
                timing={"total_time": time.time() - start_time}
            )

    def _extract_chunk_content(self, chunk: Any) -> str:
        """Extract content from a streaming chunk"""
        # Handle OpenAI streaming format
        if hasattr(chunk, 'choices') and chunk.choices:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                return chunk.choices[0].delta.content or ""

        # Handle dict format
        if isinstance(chunk, dict):
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                return delta.get('content', '')
            if 'content' in chunk:
                return chunk['content']
            if 'text' in chunk:
                return chunk['text']

        # Handle string
        if isinstance(chunk, str):
            return chunk

        return ""

    def _extract_response_content(self, response: Any) -> str:
        """Extract content from LLM response"""
        # Handle OpenAI-style response
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message'):
                return response.choices[0].message.content
            if hasattr(response.choices[0], 'text'):
                return response.choices[0].text

        # Handle dict response
        if isinstance(response, dict):
            if 'choices' in response:
                return response['choices'][0].get('message', {}).get('content', '')
            if 'output' in response:
                return response['output']
            if 'text' in response:
                return response['text']

        # Handle string response
        if isinstance(response, str):
            return response

        return str(response)

    async def guarded_stream_parallel(self, messages: List[Dict], llm_func: Callable,
                                     *args, **kwargs):
        """
        Stream LLM response with parallel guardrail evaluation

        Guardrails are evaluated based on:
        - Buffer size: When accumulated content reaches stream_buffer_size characters
        - Time interval: When stream_check_interval seconds have elapsed since last check
        - Whichever condition is met first triggers evaluation

        Args:
            messages: Chat messages for guardrail evaluation
            llm_func: Async generator function that yields streaming chunks
            *args, **kwargs: Arguments to pass to llm_func

        Yields:
            Dict containing streaming events with guardrail evaluation
        """
        logger.debug("Starting parallel streaming with guardrails (buffer_size=%s, check_interval=%s)",
                    self.stream_buffer_size if self.stream_buffer_size else "disabled",
                    f"{self.stream_check_interval:.2f}s" if self.stream_check_interval else "disabled")

        # Start request guardrail evaluation
        request_guardrails_task = asyncio.create_task(
            self.evaluate(messages, "request"),
            name="request_guardrails"
        )

        buffer = ""
        buffer_since_last_check = ""
        last_check_time = time.time()
        request_evaluation = None
        request_guardrails_done = False

        try:
            # Check request guardrails first (non-blocking)
            try:
                request_evaluation = await asyncio.wait_for(
                    request_guardrails_task, 
                    timeout=self.guardrail_timeout
                )
                request_guardrails_done = True
                
                # Use check_violations to respect guardrail policies
                action, violations = await self.check_violations(request_evaluation)
                if action == ViolationAction.BLOCK:
                    logger.warning("Request blocked by guardrails: %d violations", len(violations))
                    yield {
                        "type": "blocked",
                        "violations": violations,
                        "message": "Request blocked by guardrails"
                    }
                    return
            except asyncio.TimeoutError:
                logger.warning("Request guardrails evaluation timed out, proceeding with streaming")
            
            # Start LLM streaming (now we know request guardrails passed or timed out)
            async for chunk in llm_func(messages, *args, **kwargs):
                chunk_content = self._extract_chunk_content(chunk)
                if chunk_content:
                    buffer += chunk_content
                    buffer_since_last_check += chunk_content

                    # Yield the chunk for real-time display
                    yield {
                        "type": "chunk",
                        "content": chunk_content,
                        "accumulated_length": len(buffer)
                    }

                    # Check if we should evaluate guardrails based on configured thresholds
                    current_time = time.time()
                    time_since_last_check = current_time - last_check_time
                    
                    # Determine if we should check based on what's configured
                    should_check = False
                    if self.stream_buffer_size is not None and self.stream_check_interval is not None:
                        # Both configured - check if either threshold is met
                        should_check = (
                            len(buffer_since_last_check) >= self.stream_buffer_size or
                            time_since_last_check >= self.stream_check_interval
                        )
                    elif self.stream_buffer_size is not None:
                        # Only buffer size configured - check only character threshold
                        should_check = len(buffer_since_last_check) >= self.stream_buffer_size
                    elif self.stream_check_interval is not None:
                        # Only time interval configured - check only time threshold
                        should_check = time_since_last_check >= self.stream_check_interval

                    if should_check and buffer_since_last_check:
                        # Evaluate current buffer
                        buffer_messages = messages + [{"role": "assistant", "content": buffer}]
                        
                        logger.debug("Evaluating guardrails: %d chars accumulated (%.2fs since last check)",
                                   len(buffer_since_last_check), time_since_last_check)
                        
                        response_evaluation = await self.evaluate(buffer_messages, "response")

                        # Use check_violations to respect guardrail policies
                        action, violations = await self.check_violations(response_evaluation)
                        if action == ViolationAction.BLOCK:
                            logger.warning("Response blocked during streaming: %d violations", len(violations))
                            yield {
                                "type": "blocked",
                                "violations": violations,
                                "message": "Response blocked by guardrails during streaming"
                            }
                            return
                        
                        # Reset tracking for next check
                        buffer_since_last_check = ""
                        last_check_time = current_time

            # Final guardrail check on complete response if buffer has remaining content
            if buffer:
                buffer_messages = messages + [{"role": "assistant", "content": buffer}]
                response_evaluation = await self.evaluate(buffer_messages, "response")
                
                # Use check_violations to respect guardrail policies
                action, violations = await self.check_violations(response_evaluation)
                if action == ViolationAction.BLOCK:
                    logger.warning("Response blocked after completion: %d violations", len(violations))
                    yield {
                        "type": "blocked",
                        "violations": violations,
                        "message": "Response blocked by guardrails"
                    }
                    return

            # Stream completed successfully
            yield {"type": "completed"}

        except Exception as e:
            logger.error("Streaming failed: %s", e)
            yield {"type": "error", "message": str(e)}

    async def cleanup(self):
        """Clean up resources - shared HTTP clients are managed by the pool"""
        # Note: We don't close shared HTTP clients here as they're managed by the pool
        # and may be reused by other HaliosGuard instances
        pass

    def scan_text(self, text: str, detailed: bool = False) -> Union[str, ScanResult]:
        """
        Synchronous method to scan a single text string for guardrail violations.

        This method is designed for use in synchronous contexts like PySpark UDFs
        where async methods cannot be used. It uses synchronous HTTP requests
        throughout for maximum compatibility.

        Args:
            text: The text content to scan
            detailed: If True, return ScanResult object with full metadata.
                     If False, return simple status string (default: False)

        Returns:
            Union[str, ScanResult]: Either status string or detailed ScanResult object

        Usage:
            # Simple status (for PySpark UDFs)
            def scan_udf(text_col):
                return udf(lambda text: guard.scan_text(text), StringType())
            df.withColumn("status", scan_udf(col("text")))

            # Detailed results (for data storage)
            result = guard.scan_text("hello world", detailed=True)
            # result.response_id, result.violations, etc.
        """
        try:
            # Check credentials
            if not self.api_key:
                error_msg = "API key not provided. Set HALIOS_API_KEY environment variable or pass api_key parameter."
                logger.error(f"Error in scan_text: {error_msg}")
                status = f"error: {error_msg}"
                if detailed:
                    return ScanResult(status=status)
                else:
                    return status

            # Convert text to OpenAI message format
            messages = [{"role": "user", "content": text}]

            # Make synchronous HTTP request
            import httpx
            with httpx.Client(
                base_url=self.base_url,
                timeout=30.0,
                headers={"X-HALIOS-API-KEY": self.api_key}
            ) as http_client:

                # Make the API request
                url = f"{self.base_url}/api/v3/agents/{self.agent_id}/evaluate"
                payload = {
                    "messages": messages,
                    "invocation_type": "request"
                }

                response = http_client.post(url, json=payload)
                response.raise_for_status()
                evaluation_data = response.json()

                # Create ScanResult from response
                evaluation_result = ScanResult.from_evaluation_response(evaluation_data, "unknown")

                # Check violations synchronously
                action, violations = self._check_violations_sync(evaluation_result)

                if action == ViolationAction.BLOCK:
                    status = f"blocked: {', '.join([v.guardrail_type for v in violations])}"
                else:
                    status = "safe"

                if detailed:
                    evaluation_result.status = status
                    return evaluation_result
                else:
                    return status

        except GuardrailViolation as e:
            status = f"blocked: {', '.join([v.guardrail_type for v in e.violations])}"
            if detailed:
                return ScanResult(status=status, violations=e.violations)
            else:
                return status
        except Exception as ex:
            logger.error(f"Error in scan_text: {ex}")
            status = f"error: {str(ex)}"
            if detailed:
                return ScanResult(status=status)
            else:
                return status

    def _check_violations_sync(self, guardrail_result: Union[Dict, ScanResult]) -> tuple[ViolationAction, List[Violation]]:
        """
        Synchronous version of check_violations for scan_text method
        """
        if not guardrail_result:
            return (ViolationAction.PASS, [])

        # Handle ScanResult input
        if isinstance(guardrail_result, ScanResult):
            # Check if any guardrails were triggered
            guardrails_triggered = guardrail_result.guardrails_triggered or 0
            if guardrails_triggered > 0 and guardrail_result.violations:
                violations = guardrail_result.violations

                # Check guardrail policies for each violation type
                actions = []
                for violation in violations:
                    guardrail_type = violation.guardrail_type
                    policy_action = self.guardrail_policies.get(guardrail_type)

                    if policy_action == GuardrailPolicy.RECORD_ONLY:
                        actions.append(ViolationAction.ALLOW_OVERRIDE)
                    elif policy_action == GuardrailPolicy.BLOCK:
                        actions.append(ViolationAction.BLOCK)
                    else:
                        # No policy specified, block by default
                        actions.append(ViolationAction.BLOCK)

                # Determine overall action: BLOCK takes precedence over ALLOW_OVERRIDE
                if ViolationAction.BLOCK in actions:
                    final_action = ViolationAction.BLOCK
                else:
                    final_action = ViolationAction.ALLOW_OVERRIDE

                return (final_action, violations)

        return (ViolationAction.PASS, [])

    def _ensure_http_client_for_testing(self):
        """Ensure HTTP client is initialized for testing purposes"""
        if self.http_client is None:
            # For testing, create a synchronous HTTP client
            import httpx
            self.http_client = httpx.AsyncClient(
                base_url=self._http_client_base_url,
                timeout=30.0
            )

    async def __aenter__(self):
        # Initialize HTTP client when entering context
        if self.http_client is None:
            self.http_client = await _get_shared_http_client(self._http_client_base_url, 30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


def guarded_chat_completion(
    agent_id: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    concurrent_guardrail_processing: bool = True,
    streaming_guardrails: bool = False,
    stream_buffer_size: Optional[int] = None,
    stream_check_interval: Optional[float] = None,
    guardrail_timeout: float = 5.0,
    on_violation: Optional[Callable] = None,
    guardrail_policies: Optional[Dict[str, GuardrailPolicy]] = None  # Future: per-guardrail policies
):
    """
    Unified decorator for chat completion guardrails with configurable options

    Args:
        agent_id: HaliosAI agent ID
        api_key: HaliosAI API key (optional, uses HALIOS_API_KEY env var)
        base_url: HaliosAI base URL (optional, uses HALIOS_BASE_URL env var)
        concurrent_guardrail_processing: Run guardrails and LLM call simultaneously (default: True)
        streaming_guardrails: Enable streaming with real-time guardrail evaluation (default: False)
        stream_buffer_size: Optional character count threshold for guardrail checks
                          Specify alone for character-based only, or with stream_check_interval for hybrid
                          If neither specified, defaults to 50
        stream_check_interval: Optional time interval (seconds) for guardrail checks
                             Specify alone for time-based only, or with stream_buffer_size for hybrid
                             If neither specified, defaults to 0.5
        guardrail_timeout: Timeout for guardrail operations in seconds (default: 5.0)
        on_violation: Optional callback function called when violations occur (default: None)
        guardrail_policies: Optional dict mapping guardrail types to actions (default: None)
                          Actions: GuardrailPolicy.RECORD_ONLY (ALLOW_OVERRIDE), GuardrailPolicy.BLOCK. 
                          Default behavior (no policy): BLOCK.
                          Example: {"sensitive-data": GuardrailPolicy.BLOCK, "hate-speech": GuardrailPolicy.RECORD_ONLY}

    Returns:
        Decorator function that wraps async functions with guardrail protection

    Usage Examples:
        # Basic usage with concurrent processing
        @guarded_chat_completion(agent_id="your-agent-id")
        async def call_llm(messages):
            return await openai_client.chat.completions.create(...)

        # Sequential processing (useful for debugging)
        @guarded_chat_completion(agent_id="your-agent-id", concurrent_guardrail_processing=False)
        async def call_llm_sequential(messages):
            return await openai_client.chat.completions.create(...)

        # Streaming with real-time guardrails
        @guarded_chat_completion(
            agent_id="your-agent-id",
            streaming_guardrails=True,
            stream_buffer_size=100
        )
        async def stream_llm(messages):
            async for chunk in openai_client.chat.completions.create(..., stream=True):
                yield chunk

        # Custom guardrail policies
        @guarded_chat_completion(
            agent_id="your-agent-id",
            guardrail_policies={
                "sensitive-data": GuardrailPolicy.RECORD_ONLY,  # Allow violations (log but proceed)
                "hate-speech": GuardrailPolicy.BLOCK,           # Block violations
                "pii-detection": GuardrailPolicy.RECORD_ONLY    # Allow violations (log but proceed)
            }
        )
        async def call_llm_with_policies(messages):
            return await openai_client.chat.completions.create(...)
    """
    def decorator(func: Callable):
        if streaming_guardrails:
            # For streaming functions, return an async generator
            async def streaming_wrapper(*args, **kwargs):
                # Extract messages from function arguments
                messages = []
                if args and isinstance(args[0], list):
                    messages = args[0]
                elif 'messages' in kwargs:
                    messages = kwargs['messages']
                else:
                    raise ValueError("Function must receive 'messages' as first argument or keyword argument")

                # Create unified HaliosGuard instance and stream
                config = {
                    'agent_id': agent_id,
                    'api_key': api_key,
                    'base_url': base_url,
                    'parallel': concurrent_guardrail_processing,
                    'streaming': True,
                    'stream_buffer_size': stream_buffer_size,
                    'stream_check_interval': stream_check_interval,
                    'guardrail_timeout': guardrail_timeout,
                    'guardrail_policies': guardrail_policies
                }
                async with HaliosGuard(**config) as guard_client:
                    # Remove messages from args since we've extracted it and pass it separately
                    remaining_args = args[1:] if args and isinstance(args[0], list) else args
                    async for event in guard_client.guarded_stream_parallel(messages, func, *remaining_args, **kwargs):
                        # Check for violation events and throw exceptions for consistency
                        if isinstance(event, dict) and event.get('type') == 'blocked':
                            violations = event.get('violations', [])
                            if not isinstance(violations, list):
                                violations = []
                            raise GuardrailViolation(
                                violation_type="response",
                                violations=violations,
                                blocked_content=f"Streaming response blocked at {len(event.get('message', ''))}",
                                timing={"streaming": True}
                            )
                        yield event
            return streaming_wrapper
        else:
            # For non-streaming functions, return a regular async function
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract messages from function arguments
                messages = []
                if args and isinstance(args[0], list):
                    messages = args[0]
                elif 'messages' in kwargs:
                    messages = kwargs['messages']
                else:
                    raise ValueError("Function must receive 'messages' as first argument or keyword argument")

                # Use unified HaliosGuard for both concurrent and sequential processing
                config = {
                    'agent_id': agent_id,
                    'api_key': api_key,
                    'base_url': base_url,
                    'parallel': concurrent_guardrail_processing,
                    'streaming': False,
                    'guardrail_timeout': guardrail_timeout,
                    'guardrail_policies': guardrail_policies
                }
                async with HaliosGuard(**config) as guard_client:
                    if concurrent_guardrail_processing:
                        # Remove messages from args since we've extracted it and pass it separately
                        remaining_args = args[1:] if args and isinstance(args[0], list) else args
                        result = await guard_client.guarded_call_parallel(messages, func, *remaining_args, **kwargs)
                        if result.result != ExecutionResult.SUCCESS:
                            violations = result.request_violations or result.response_violations or []
                            
                            if result.result == ExecutionResult.REQUEST_BLOCKED:
                                violation = GuardrailViolation(
                                    violation_type="request",
                                    violations=violations,
                                    blocked_content=str(messages),
                                    timing=result.timing
                                )
                            elif result.result == ExecutionResult.RESPONSE_BLOCKED:
                                violation = GuardrailViolation(
                                    violation_type="response", 
                                    violations=violations,
                                    blocked_content=result.original_response,
                                    timing=result.timing
                                )
                            else:
                                raise ValueError(result.error_message or "Guardrail evaluation failed")
                            
                            # Call custom violation handler if provided
                            if on_violation:
                                on_violation(violation)
                            
                            # Raise the custom exception
                            raise violation
                        return result.final_response
                    else:
                        # Sequential processing: check request, then call LLM, then check response
                        logger.debug("Running request guardrails sequentially")
                        request_start = time.time()
                        request_result = await guard_client.evaluate(messages, "request")
                        request_time = time.time() - request_start

                        if (await guard_client.check_violations(request_result))[0] == ViolationAction.BLOCK:
                            action, violations = await guard_client.check_violations(request_result)
                            
                            violation = GuardrailViolation(
                                violation_type="request",
                                violations=violations,
                                blocked_content=str(messages),
                                timing={'request_guardrail_time': request_time}
                            )
                            
                            if on_violation:
                                on_violation(violation)
                            
                            raise violation

                        logger.debug("Request guardrails passed, calling LLM")
                        llm_start = time.time()
                        response = await func(*args, **kwargs)
                        llm_time = time.time() - llm_start

                        # Always check response guardrails synchronously
                        logger.debug("Evaluating response guardrails")
                        response_start = time.time()
                        response_message = guard_client.extract_response_message(response)
                        full_conversation = messages + [response_message]
                        response_result = await guard_client.evaluate(full_conversation, "response")
                        response_time = time.time() - response_start

                        if (await guard_client.check_violations(response_result))[0] == ViolationAction.BLOCK:
                            action, violations = await guard_client.check_violations(response_result)
                            
                            violation = GuardrailViolation(
                                violation_type="response",
                                violations=violations,
                                blocked_content=str(response_message),
                                timing={
                                    'request_guardrail_time': request_time,
                                    'llm_time': llm_time,
                                    'response_guardrail_time': response_time
                                }
                            )
                            
                            if on_violation:
                                on_violation(violation)
                            
                            raise violation

                        # Add timing info to response object
                        total_time = request_time + llm_time + response_time
                        if isinstance(response, dict):
                            # For dict responses, add timing as a key
                            if '_halios_timing' not in response:
                                response['_halios_timing'] = {}
                            response['_halios_timing'].update({
                                'request_guardrail_time': request_time,
                                'llm_time': llm_time,
                                'response_guardrail_time': response_time,
                                'total_time': total_time,
                                'mode': 'sequential'
                            })
                        else:
                            # For object responses, add as attribute
                            if not hasattr(response, '_halios_timing'):
                                response._halios_timing = {}
                            response._halios_timing.update({
                                'request_guardrail_time': request_time,
                                'llm_time': llm_time,
                                'response_guardrail_time': response_time,
                                'total_time': total_time,
                                'mode': 'sequential'
                            })

                        logger.debug("Guarded function completed successfully in %.3fs", request_time + llm_time + response_time)
                        return response
            return wrapper
    return decorator


