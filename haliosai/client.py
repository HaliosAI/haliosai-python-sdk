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
from typing import List, Dict, Any, Callable, Optional
from functools import wraps
from enum import Enum
from dataclasses import dataclass


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


@dataclass
class GuardedResponse:
    """Response object containing execution results and metadata"""
    result: ExecutionResult
    final_response: Optional[Any] = None
    original_response: Optional[str] = None
    request_violations: List[Dict] = None
    response_violations: List[Dict] = None
    timing: Dict[str, float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.request_violations is None:
            self.request_violations = []
        if self.response_violations is None:
            self.response_violations = []
        if self.timing is None:
            self.timing = {}


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
                 stream_buffer_size: int = 50, stream_check_interval: float = 0.5,
                 guardrail_timeout: float = 5.0, http_client: Optional[httpx.AsyncClient] = None):
        """
        Initialize unified HaliosGuard with comprehensive configuration options

        This constructor sets up the guardrail client with all necessary configuration
        for both parallel and sequential processing modes, streaming support, and
        performance tuning options.

        Args:
            agent_id: Unique identifier for your HaliosAI agent configuration
            api_key: HaliosAI API key (defaults to HALIOS_API_KEY environment variable)
            base_url: HaliosAI API base URL (defaults to HALIOS_BASE_URL or localhost:2000)
            parallel: Enable parallel processing (guardrails run concurrently with LLM calls)
                     - True: Faster execution, guardrails don't block LLM calls
                     - False: Safer execution, guardrails complete before LLM calls
            streaming: Enable real-time streaming guardrail evaluation
            stream_buffer_size: Character buffer size before guardrail evaluation during streaming
            stream_check_interval: Time interval between guardrail checks during streaming
            guardrail_timeout: Maximum time to wait for guardrail evaluation (seconds)
            http_client: Optional pre-configured HTTP client (uses shared pool by default)

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
        self.base_url = base_url or os.getenv("HALIOS_BASE_URL", "http://localhost:2000")
        self.parallel = parallel
        self.streaming = streaming
        self.stream_buffer_size = stream_buffer_size
        self.stream_check_interval = stream_check_interval
        self.guardrail_timeout = guardrail_timeout

        # HTTP client management - uses shared pool for connection reuse
        self.http_client = http_client  # Will be initialized lazily from shared pool
        self._http_client_base_url = base_url or os.getenv("HALIOS_BASE_URL", "http://localhost:2000")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Lazily get or create shared HTTP client"""
        if self.http_client is None:
            self.http_client = await _get_shared_http_client(self._http_client_base_url, 30.0)
        return self.http_client

    async def evaluate(self, messages: List[Dict], invocation_type: str = "request") -> Dict:
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
            Dict containing detailed evaluation results:
            {
                "guardrails_triggered": int,  # Number of guardrails that triggered
                "results": [...],             # Detailed results from each guardrail
                "execution_time": float,     # Time taken for evaluation
                "error": str|None           # Any error that occurred
            }

        Usage Examples:
            # Pre-call evaluation (recommended for safety)
            result = await guard.evaluate([{"role": "user", "content": "Hello"}], "request")

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

        logger.debug(f"Initialized unified HaliosGuard with agent_id={self.agent_id}, parallel={self.parallel}, streaming={self.streaming}")

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
            result = response.json()

            logger.debug(f"Guardrail evaluation completed: {result.get('guardrails_triggered', 0)} triggered")
            return result

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
        """
        # Handle OpenAI-style response
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message'):
                message = response.choices[0].message
                content = message.content or ""

                # If there are tool calls but no content, create a description
                if hasattr(message, 'tool_calls') and message.tool_calls and not content:
                    tool_names = [tc.function.name for tc in message.tool_calls]
                    content = f"Assistant called tools: {', '.join(tool_names)}"

                logger.debug(f"Extracted content from OpenAI message: {len(content)} chars")
                return content
            if hasattr(response.choices[0], 'text'):
                content = response.choices[0].text
                logger.debug(f"Extracted content from OpenAI text: {len(content)} chars")
                return content

        # Handle dict response
        if isinstance(response, dict):
            if 'choices' in response:
                message = response['choices'][0].get('message', {})
                content = message.get('content', '')
                if content is None:
                    # Check for tool calls when content is None
                    tool_calls = message.get('tool_calls', [])
                    if tool_calls:
                        tool_names = [call.get('function', {}).get('name', 'unknown') for call in tool_calls]
                        content = f"Assistant called tools: {', '.join(tool_names)}"
                    else:
                        content = ''
                logger.debug("Extracted content from dict response: %s chars", len(content))
                return content
            if 'output' in response:
                content = response['output']
                logger.debug("Extracted content from output field: %s chars", len(content))
                return content
            if 'text' in response:
                content = response['text']
                logger.debug("Extracted content from text field: %s chars", len(content))
                return content

        # Handle string response
        if isinstance(response, str):
            logger.debug("Using string response directly: %s chars", len(response))
            return response

        # Fallback to string conversion
        content = str(response)
        logger.debug("Converted response to string: %s chars", len(content))
        return content

    async def check_violations(self, guardrail_result: Dict) -> bool:
        """
        Check if any guardrails were triggered and should halt execution

        Args:
            guardrail_result: Result from guardrail evaluation

        Returns:
            True if violations found and execution should be halted
        """
        if not guardrail_result:
            return False

        # Check if any guardrails were triggered
        guardrails_triggered = guardrail_result.get('guardrails_triggered', 0)
        if guardrails_triggered > 0:
            # Find the specific violations
            violations = []
            results = guardrail_result.get('result', [])
            for result in results:
                if result.get('triggered', False):
                    violations.append({
                        'type': result.get('guardrail_type', 'unknown'),
                        'analysis': result.get('analysis', {}),
                        'guardrail_uuid': result.get('guardrail_uuid', 'unknown')
                    })

            if violations:
                # Format violation details for logging
                violation_details = []
                for v in violations:
                    analysis = v.get('analysis') or {}
                    detail = f"{v['type']}"
                    if 'explanation' in analysis and analysis['explanation']:
                        detail += f": {analysis['explanation']}"
                    elif 'detected_topics' in analysis and analysis['detected_topics']:
                        detail += f": detected {', '.join(analysis['detected_topics'])}"
                    elif analysis.get('flagged'):
                        detail += ": content flagged as potentially harmful"
                    violation_details.append(detail)

                violation_summary = "; ".join(violation_details)
                logger.warning("Guardrail violations detected: %s", violation_summary)
                return True

        return False

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
                            if await self.check_violations(request_evaluation):
                                # Extract violation details
                                violation_details = []
                                results = request_evaluation.get('result', [])
                                for result in results:
                                    if result.get('triggered', False):
                                        violation_details.append({
                                            'type': result.get('guardrail_type', 'unknown'),
                                            'analysis': result.get('analysis', {}),
                                            'guardrail_uuid': result.get('guardrail_uuid', 'unknown')
                                        })
                                
                                # Cancel LLM task if still running
                                if not llm_done and llm_task in pending:
                                    llm_task.cancel()
                                    pending.discard(llm_task)
                                    logger.debug("Cancelled LLM task due to request guardrail violations")

                                logger.warning(f"Request blocked: {len(violation_details)} violations detected")
                                return GuardedResponse(
                                    result=ExecutionResult.REQUEST_BLOCKED,
                                    request_violations=violation_details,
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

            # Extract response content for guardrail evaluation
            response_content = self._extract_response_content(llm_response)
            full_conversation = messages + [{"role": "assistant", "content": response_content}]
            response_evaluation = await self.evaluate(full_conversation, "response")

            response_time = time.time() - response_start

            # Check for response violations
            response_violations = response_evaluation.get("violations", [])
            if response_violations:
                logger.warning(f"Response blocked: {len(response_violations)} violations detected")
                return GuardedResponse(
                    result=ExecutionResult.RESPONSE_BLOCKED,
                    original_response=response_content,
                    response_violations=response_violations,
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
                if assistant_msg and assistant_msg.get("content") != response_content:
                    final_response = assistant_msg["content"]  # Only return text if modified
                    logger.debug("Response was modified by guardrails")

            total_time = time.time() - start_time
            parallel_savings = max(0, request_time + llm_time - total_time)

            logger.debug(f"Parallel guarded call completed successfully in {total_time:.3f}s (saved {parallel_savings:.3f}s)")

            return GuardedResponse(
                result=ExecutionResult.SUCCESS,
                final_response=final_response,
                original_response=response_content,
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

    def patch_function(self, original_func: Callable) -> Callable:
        """
        DEPRECATED: Create a guarded version of any async function

        This method is deprecated and will be removed in a future version.
        Use the @guarded_chat_completion decorator instead for new integrations.

        Migration Guide:
        OLD: guard.patch_function(your_llm_function)
        NEW: @guarded_chat_completion(agent_id="your-agent")
             async def your_llm_function(messages): ...

        Benefits of Migration:
        - Better performance with optimized guardrail evaluation
        - Cleaner code without monkey-patching
        - Easier testing and debugging
        - Future-proof API compatibility

        Args:
            original_func: Original async function to wrap with guardrails

        Returns:
            Wrapped function with guardrail protection

        Warning:
            This method may be removed in version 2.0. Migrate to decorators
            for better maintainability and performance.
        """
        warnings.warn(
            "patch_function is deprecated. Use @guarded_chat_completion decorator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        @wraps(original_func)
        async def guarded_func(*args, **kwargs):
            # Extract messages for guardrail evaluation
            messages = self.extract_messages(*args, **kwargs)
            if not messages:
                logger.debug("No messages found, calling original function without guardrails")
                return await original_func(*args, **kwargs)

            total_start = time.time()
            logger.debug("Starting guarded function call (parallel=%s)", self.parallel)

            if self.parallel:
                # Use parallel processing
                async with self:
                    result = await self.guarded_call_parallel(messages, original_func, *args, **kwargs)
                    if result.result != ExecutionResult.SUCCESS:
                        if result.result == ExecutionResult.REQUEST_BLOCKED:
                            raise ValueError(result.error_message or "Request blocked by guardrails")
                        elif result.result == ExecutionResult.RESPONSE_BLOCKED:
                            raise ValueError(result.error_message or "Response blocked by guardrails")
                        else:
                            raise ValueError(result.error_message or "Guardrail evaluation failed")
                    return result.final_response
            else:
                # Sequential execution: check request, then call LLM
                logger.debug("Running request guardrails sequentially")
                request_start = time.time()
                request_result = await self.evaluate(messages, "request")
                request_time = time.time() - request_start

                if await self.check_violations(request_result):
                    # Extract violation details for better error message
                    violations = []
                    results = request_result.get('result', [])
                    for result in results:
                        if result.get('triggered', False):
                            violations.append(result.get('guardrail_type', 'unknown'))

                    violation_types = ', '.join(violations) if violations else 'policy violation'
                    error_msg = f"Request blocked by guardrails: {violation_types} detected"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.debug("Request guardrails passed, calling LLM")
                llm_start = time.time()
                response = await original_func(*args, **kwargs)
                llm_time = time.time() - llm_start

                # Always check response guardrails synchronously
                logger.debug("Evaluating response guardrails")
                response_start = time.time()
                response_message = self.extract_response_message(response)
                full_conversation = messages + [response_message]
                response_result = await self.evaluate(full_conversation, "response")
                response_time = time.time() - response_start

                if await self.check_violations(response_result):
                    # Extract violation details for better error message
                    violations = []
                    results = response_result.get('result', [])
                    for result in results:
                        if result.get('triggered', False):
                            violations.append(result.get('guardrail_type', 'unknown'))

                    violation_types = ', '.join(violations) if violations else 'policy violation'
                    error_msg = f"Response blocked by guardrails: {violation_types} detected"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Add timing info to response object
                total_time = time.time() - total_start
                if not hasattr(response, '_halios_timing'):
                    response._halios_timing = {}

                response._halios_timing.update({
                    'request_guardrail_time': request_time,
                    'llm_time': llm_time,
                    'response_guardrail_time': response_time,
                    'total_time': total_time,
                    'mode': 'parallel' if self.parallel else 'sequential'
                })

                logger.debug("Guarded function completed successfully in %.3fs", total_time)
                return response

        return guarded_func

    def patch(self, obj, method_name: str):
        """
        Patch a method on an object/class

        Args:
            obj: Object or class to patch
            method_name: Name of method to patch
        """
        logger.debug("Patching %s.%s", obj.__class__.__name__, method_name)
        original_method = getattr(obj, method_name)
        guarded_method = self.patch_function(original_method)
        setattr(obj, method_name, guarded_method)

    def __call__(self, func: Callable) -> Callable:
        """Use HaliosGuard as a decorator"""
        return self.patch_function(func)

    async def evaluate_input_async(self, input_text: str) -> GuardedResponse:
        """
        Evaluate input text through guardrails (async version)

        Args:
            input_text: The input text to evaluate

        Returns:
            GuardedResponse: Result of guardrail evaluation
        """
        messages = [{"role": "user", "content": input_text}]
        result = await self.evaluate(messages, invocation_type="request")

        # Check if any guardrails were triggered
        triggered = result.get("guardrails_triggered", 0) > 0
        execution_result = ExecutionResult.REQUEST_BLOCKED if triggered else ExecutionResult.SUCCESS

        return GuardedResponse(
            result=execution_result,
            original_response=input_text,
            request_violations=result.get("violations", []),
            timing={"evaluation_time": result.get("evaluation_time", 0.0)}
        )

    async def evaluate_output_async(self, input_text: str, output_text: str) -> GuardedResponse:
        """
        Evaluate output text through guardrails (async version)

        Args:
            input_text: The original input text
            output_text: The output text to evaluate

        Returns:
            GuardedResponse: Result of guardrail evaluation
        """
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        result = await self.evaluate(messages, invocation_type="response")

        # Check if any guardrails were triggered
        triggered = result.get("guardrails_triggered", 0) > 0
        execution_result = ExecutionResult.RESPONSE_BLOCKED if triggered else ExecutionResult.SUCCESS

        return GuardedResponse(
            result=execution_result,
            final_response=output_text,
            original_response=output_text,
            response_violations=result.get("violations", []),
            timing={"evaluation_time": result.get("evaluation_time", 0.0)}
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

        Args:
            messages: Chat messages for guardrail evaluation
            llm_func: Async generator function that yields streaming chunks
            *args, **kwargs: Arguments to pass to llm_func

        Yields:
            Dict containing streaming events with guardrail evaluation
        """
        logger.debug("Starting parallel streaming with guardrails")

        # Start request guardrail evaluation
        request_guardrails_task = asyncio.create_task(
            self.evaluate(messages, "request"),
            name="request_guardrails"
        )

        buffer = ""
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
                
                # Check for violations immediately
                violations = [r for r in request_evaluation.get("results", []) if r.get("triggered")]
                if violations:
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

                    # Yield the chunk for real-time display
                    yield {
                        "type": "chunk",
                        "content": chunk_content,
                        "accumulated_length": len(buffer)
                    }

                    # Check buffer size for guardrail evaluation
                    if len(buffer) >= self.stream_buffer_size:
                        # Evaluate current buffer
                        buffer_messages = messages + [{"role": "assistant", "content": buffer}]
                        response_evaluation = await self.evaluate(buffer_messages, "response")

                        violations = [r for r in response_evaluation.get("results", []) if r.get("triggered")]
                        if violations:
                            logger.warning("Response blocked during streaming: %d violations", len(violations))
                            yield {
                                "type": "blocked",
                                "violations": violations,
                                "message": "Response blocked by guardrails during streaming"
                            }
                            return

            # Final guardrail check on complete response if buffer has remaining content
            if buffer:
                buffer_messages = messages + [{"role": "assistant", "content": buffer}]
                response_evaluation = await self.evaluate(buffer_messages, "response")
                
                violations = [r for r in response_evaluation.get("results", []) if r.get("triggered")]
                if violations:
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


class ParallelGuardedChat:
    """
    Legacy wrapper for HaliosGuard - DEPRECATED

    This class is deprecated. Use HaliosGuard instead.
    Maintained for backward compatibility.
    """

    def __init__(self, agent_id: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 guardrail_timeout: float = 5.0, streaming: bool = False,
                 stream_buffer_size: int = 50, stream_check_interval: float = 0.5):
        """
        Initialize ParallelGuardedChat (DEPRECATED)

        Args:
            agent_id: Agent ID for guardrail configuration
            api_key: API key (defaults to HALIOS_API_KEY env var)
            base_url: Base URL for guardrails API (defaults to HALIOS_BASE_URL env var)
            guardrail_timeout: Timeout for guardrail operations
            streaming: Enable streaming mode
            stream_buffer_size: Characters to buffer before guardrail check
            stream_check_interval: Time interval for guardrail checks
        """
        import warnings
        warnings.warn(
            "ParallelGuardedChat is deprecated. Use HaliosGuard instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Delegate to unified HaliosGuard
        self._guard = HaliosGuard(
            agent_id=agent_id,
            api_key=api_key,
            base_url=base_url,
            parallel=True,  # Always enable parallel processing for legacy compatibility
            streaming=streaming,
            stream_buffer_size=stream_buffer_size,
            stream_check_interval=stream_check_interval,
            guardrail_timeout=guardrail_timeout
        )

    async def __aenter__(self):
        await self._guard.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._guard.__aexit__(exc_type, exc_val, exc_tb)

    async def evaluate_guardrails(self, messages: List[Dict], invocation_type: str) -> Dict:
        """Legacy method - delegates to HaliosGuard.evaluate"""
        return await self._guard.evaluate(messages, invocation_type)

    async def guarded_call_parallel(self, messages: List[Dict], llm_func: Callable,
                                   *args, **kwargs) -> GuardedResponse:
        """Legacy method - delegates to HaliosGuard.guarded_call_parallel"""
        return await self._guard.guarded_call_parallel(messages, llm_func, *args, **kwargs)

    async def guarded_stream_parallel(self, messages: List[Dict], llm_stream_func: Callable,
                                     *args, **kwargs):
        """Legacy method - delegates to HaliosGuard.guarded_stream_parallel"""
        async for event in self._guard.guarded_stream_parallel(messages, llm_stream_func, *args, **kwargs):
            yield event

    def patch_function(self, func: Callable) -> Callable:
        """Legacy method - delegates to HaliosGuard.patch_function"""
        return self._guard.patch_function(func)

    def _extract_response_content(self, response: Any) -> str:
        """Legacy method - delegates to HaliosGuard._extract_response_content"""
        return self._guard._extract_response_content(response)

    def extract_messages(self, *args, **kwargs) -> List[Dict]:
        """Legacy method - delegates to HaliosGuard.extract_messages"""
        return self._guard.extract_messages(*args, **kwargs)

    def extract_response_message(self, response: Any) -> Dict:
        """Legacy method - delegates to HaliosGuard.extract_response_message"""
        return self._guard.extract_response_message(response)

    async def check_violations(self, guardrail_result: Dict) -> bool:
        """Legacy method - delegates to HaliosGuard.check_violations"""
        return await self._guard.check_violations(guardrail_result)

    # Property forwarding for backward compatibility
    @property
    def agent_id(self):
        """Forward agent_id from internal HaliosGuard"""
        return self._guard.agent_id

    @property
    def api_key(self):
        """Forward api_key from internal HaliosGuard"""
        return self._guard.api_key

    @property
    def base_url(self):
        """Forward base_url from internal HaliosGuard"""
        return self._guard.base_url

    @property
    def streaming(self):
        """Forward streaming from internal HaliosGuard"""
        return self._guard.streaming

    @property
    def stream_buffer_size(self):
        """Forward stream_buffer_size from internal HaliosGuard"""
        return self._guard.stream_buffer_size

    @property
    def stream_check_interval(self):
        """Forward stream_check_interval from internal HaliosGuard"""
        return self._guard.stream_check_interval

    @property
    def guardrail_timeout(self):
        """Forward guardrail_timeout from internal HaliosGuard"""
        return self._guard.guardrail_timeout

    @property
    def parallel(self):
        """Forward parallel from internal HaliosGuard"""
        return self._guard.parallel

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
            self.evaluate_guardrails(messages, "request"),
            name="request_guardrails"
        )
        llm_task = asyncio.create_task(
            llm_func(*args, **kwargs),
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
                            violations = [r for r in request_evaluation.get("results", []) if r.get("triggered")]
                            if violations:
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
            
            # Extract response content for guardrail evaluation
            response_content = self._extract_response_content(llm_response)
            full_conversation = messages + [{"role": "assistant", "content": response_content}]
            response_evaluation = await self.evaluate_guardrails(full_conversation, "response")
            
            response_time = time.time() - response_start
            
            # Check for response violations
            response_violations = [r for r in response_evaluation.get("results", []) if r.get("triggered")]
            if response_violations:
                logger.warning(f"Response blocked: {len(response_violations)} violations detected")
                return GuardedResponse(
                    result=ExecutionResult.RESPONSE_BLOCKED,
                    original_response=response_content,
                    response_violations=response_violations,
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
                if assistant_msg and assistant_msg.get("content") != response_content:
                    final_response = assistant_msg["content"]  # Only return text if modified
                    logger.debug("Response was modified by guardrails")
            
            total_time = time.time() - start_time
            parallel_savings = max(0, request_time + llm_time - total_time)
            
            logger.debug(f"Parallel guarded call completed successfully in {total_time:.3f}s (saved {parallel_savings:.3f}s)")
            
            return GuardedResponse(
                result=ExecutionResult.SUCCESS,
                final_response=final_response,
                original_response=response_content,
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
    
    async def guarded_stream_parallel(self, messages: List[Dict], llm_stream_func: Callable, 
                                     *args, **kwargs):
        """
        Perform guarded streaming LLM call with real-time guardrail evaluation
        
        Args:
            messages: Chat messages for guardrail evaluation
            llm_stream_func: Async generator function that yields streaming response chunks
            *args, **kwargs: Arguments to pass to llm_stream_func
        
        Yields:
            Dict with keys: 'type' ('chunk', 'violation', 'error'), 'content', 'timing', etc.
        """
        if not self.streaming:
            raise ValueError("Streaming not enabled. Set streaming=True in constructor.")
        
        start_time = time.time()
        logger.debug("Starting streaming guarded call")
        
        # Run request guardrails first
        request_start = time.time()
        try:
            request_evaluation = await self.evaluate_guardrails(messages, "request")
            request_time = time.time() - request_start
            
            # Check for request violations
            violations = [r for r in request_evaluation.get("results", []) if r.get("triggered")]
            if violations:
                logger.warning(f"Streaming request blocked: {len(violations)} violations detected")
                yield {
                    'type': 'violation',
                    'stage': 'request',
                    'violations': violations,
                    'timing': {
                        'request_guardrails_time': request_time,
                        'total_time': time.time() - start_time
                    }
                }
                return
        
        except Exception as e:
            logger.error(f"Request guardrail evaluation failed: {e}")
            yield {
                'type': 'error',
                'stage': 'request',
                'error': str(e),
                'timing': {'total_time': time.time() - start_time}
            }
            return
        
        # Start streaming LLM response
        accumulated_content = ""
        last_check_time = time.time()
        last_check_length = 0
        chunk_count = 0
        llm_start = time.time()
        
        logger.debug("Starting LLM streaming")
        
        try:
            async for chunk in llm_stream_func(messages, *args, **kwargs):
                chunk_count += 1
                
                # Extract content from chunk
                chunk_content = self._extract_chunk_content(chunk)
                accumulated_content += chunk_content
                
                # Yield the chunk immediately for real-time streaming
                yield {
                    'type': 'chunk',
                    'content': chunk_content,
                    'accumulated_length': len(accumulated_content),
                    'chunk_number': chunk_count
                }
                
                # Check if we should evaluate guardrails
                current_time = time.time()
                should_check = (
                    # Buffer size threshold
                    len(accumulated_content) - last_check_length >= self.stream_buffer_size or
                    # Time interval threshold
                    current_time - last_check_time >= self.stream_check_interval
                )
                
                if should_check and accumulated_content.strip():
                    try:
                        # Evaluate response guardrails on accumulated content
                        eval_start = time.time()
                        full_conversation = messages + [{"role": "assistant", "content": accumulated_content}]
                        response_evaluation = await self.evaluate_guardrails(full_conversation, "response")
                        eval_time = time.time() - eval_start
                        
                        # Check for violations
                        response_violations = [r for r in response_evaluation.get("results", []) if r.get("triggered")]
                        if response_violations:
                            logger.warning(f"Streaming response blocked: {len(response_violations)} violations detected")
                            yield {
                                'type': 'violation',
                                'stage': 'response',
                                'violations': response_violations,
                                'content_length': len(accumulated_content),
                                'partial_content': accumulated_content,
                                'timing': {
                                    'request_guardrails_time': request_time,
                                    'llm_time': time.time() - llm_start,
                                    'response_guardrails_time': eval_time,
                                    'total_time': time.time() - start_time
                                }
                            }
                            return
                        
                        # Update check tracking
                        last_check_time = current_time
                        last_check_length = len(accumulated_content)
                        
                        # Yield guardrail check status
                        yield {
                            'type': 'guardrail_check',
                            'status': 'passed',
                            'content_length': len(accumulated_content),
                            'check_time': eval_time
                        }
                        
                    except Exception as e:
                        # Don't stop streaming for guardrail evaluation errors
                        logger.warning(f"Guardrail evaluation error during streaming: {e}")
                        yield {
                            'type': 'warning',
                            'message': f"Guardrail evaluation failed: {str(e)}",
                            'content_length': len(accumulated_content)
                        }
            
            # Final guardrail check on complete response
            if accumulated_content.strip():
                logger.debug("Performing final guardrail check on complete response")
                eval_start = time.time()
                try:
                    full_conversation = messages + [{"role": "assistant", "content": accumulated_content}]
                    response_evaluation = await self.evaluate_guardrails(full_conversation, "response")
                    eval_time = time.time() - eval_start
                    
                    response_violations = [r for r in response_evaluation.get("results", []) if r.get("triggered")]
                    if response_violations:
                        logger.warning(f"Final response check blocked: {len(response_violations)} violations detected")
                        yield {
                            'type': 'violation',
                            'stage': 'response_final',
                            'violations': response_violations,
                            'content_length': len(accumulated_content),
                            'final_content': accumulated_content,
                            'timing': {
                                'request_guardrails_time': request_time,
                                'llm_time': time.time() - llm_start,
                                'response_guardrails_time': eval_time,
                                'total_time': time.time() - start_time
                            }
                        }
                        return
                    
                    # Check if response was modified
                    processed_messages = response_evaluation.get("processed_messages", [])
                    final_response = accumulated_content
                    
                    if processed_messages:
                        assistant_msg = next(
                            (msg for msg in reversed(processed_messages) if msg.get("role") == "assistant"), 
                            None
                        )
                        if assistant_msg and assistant_msg.get("content") != accumulated_content:
                            final_response = assistant_msg["content"]
                            logger.debug("Final response was modified by guardrails")
                    
                    # Streaming completed successfully
                    logger.debug(f"Streaming completed successfully: {chunk_count} chunks, {len(final_response)} chars")
                    yield {
                        'type': 'completed',
                        'final_content': final_response,
                        'original_content': accumulated_content,
                        'modified': final_response != accumulated_content,
                        'total_chunks': chunk_count,
                        'timing': {
                            'request_guardrails_time': request_time,
                            'llm_time': time.time() - llm_start,
                            'response_guardrails_time': eval_time,
                            'total_time': time.time() - start_time
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Final guardrail check failed: {e}")
                    yield {
                        'type': 'error',
                        'stage': 'response_final',
                        'error': str(e),
                        'partial_content': accumulated_content,
                        'timing': {'total_time': time.time() - start_time}
                    }
            
        except Exception as e:
            logger.error(f"Streaming LLM call failed: {e}")
            yield {
                'type': 'error',
                'stage': 'streaming',
                'error': str(e),
                'partial_content': accumulated_content,
                'timing': {'total_time': time.time() - start_time}
            }
    
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
    
    async def cleanup(self):
        """Clean up resources - delegates to HaliosGuard"""
        await self._guard.cleanup()

    async def __aenter__(self):
        await self._guard.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._guard.__aexit__(exc_type, exc_val, exc_tb)




# Unified decorator for all chat completion guardrail functionality
def guarded_chat_completion(
    agent_id: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    concurrent_guardrail_processing: bool = True,
    streaming_guardrails: bool = False,
    stream_buffer_size: int = 50,
    stream_check_interval: float = 0.5,
    guardrail_timeout: float = 5.0
):
    """
    Unified decorator for chat completion guardrails with configurable options

    Args:
        agent_id: HaliosAI agent ID
        api_key: HaliosAI API key (optional, uses HALIOS_API_KEY env var)
        base_url: HaliosAI base URL (optional, uses HALIOS_BASE_URL env var)
        concurrent_guardrail_processing: Run guardrails and LLM call simultaneously (default: True)
        streaming_guardrails: Enable streaming with real-time guardrail evaluation (default: False)
        stream_buffer_size: Characters to buffer before guardrail check (default: 50)
        stream_check_interval: Time interval for guardrail checks in seconds (default: 0.5)
        guardrail_timeout: Timeout for guardrail operations in seconds (default: 5.0)
        http_client: Optional shared HTTP client for reuse across calls (default: None)

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

        # Usage for streaming:
        async for event in stream_llm(messages):
            if event['type'] == 'chunk':
                print(event['content'], end='')
            elif event['type'] == 'completed':
                print("\\nStream completed!")
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
                    'guardrail_timeout': guardrail_timeout
                }
                async with HaliosGuard(**config) as guard_client:
                    # Remove messages from args since we've extracted it and pass it separately
                    remaining_args = args[1:] if args and isinstance(args[0], list) else args
                    async for event in guard_client.guarded_stream_parallel(messages, func, *remaining_args, **kwargs):
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
                    'guardrail_timeout': guardrail_timeout
                }
                async with HaliosGuard(**config) as guard_client:
                    if concurrent_guardrail_processing:
                        # Remove messages from args since we've extracted it and pass it separately
                        remaining_args = args[1:] if args and isinstance(args[0], list) else args
                        result = await guard_client.guarded_call_parallel(messages, func, *remaining_args, **kwargs)
                        if result.result != ExecutionResult.SUCCESS:
                            if result.result == ExecutionResult.REQUEST_BLOCKED:
                                raise ValueError(result.error_message or "Request blocked by guardrails")
                            elif result.result == ExecutionResult.RESPONSE_BLOCKED:
                                raise ValueError(result.error_message or "Response blocked by guardrails")
                            else:
                                raise ValueError(result.error_message or "Guardrail evaluation failed")
                        return result.final_response
                    else:
                        # Sequential processing using patch_function
                        guarded_func = guard_client.patch_function(func)
                        # For sequential processing, keep all original args since patch_function extracts messages internally
                        return await guarded_func(*args, **kwargs)
            return wrapper
    return decorator


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS - DEPRECATED
# =============================================================================
# These functions provide backward compatibility but are deprecated.
# They exist to ease migration from older SDK versions.
#
# Migration Timeline:
# - Current: Functions work but emit deprecation warnings
# - Future: Functions will be removed in a major version update
#
# Recommended Migration:
#   OLD: guard(agent_id="your-agent")
#   NEW: @guarded_chat_completion(agent_id="your-agent")
# =============================================================================

def guard(agent_id: str, api_key: Optional[str] = None, base_url: Optional[str] = None, parallel: bool = False):
    """
    DEPRECATED: Create a HaliosGuard instance

    This function creates a HaliosGuard instance for manual guardrail evaluation.
    It is deprecated in favor of the @guarded_chat_completion decorator which
    provides a more convenient and maintainable API.

    Args:
        agent_id: Agent ID for guardrail configuration
        api_key: API key (defaults to HALIOS_API_KEY env var)
        base_url: Base URL for guardrails API (defaults to HALIOS_BASE_URL env var)
        parallel: Enable parallel execution of guardrails and LLM calls

    Returns:
        HaliosGuard instance that can be used as decorator or patcher

    Migration:
        Instead of: guard_instance = guard("your-agent")
        Use: @guarded_chat_completion(agent_id="your-agent")
    """
    logger.warning("guard() is deprecated. Use guarded_chat_completion() decorator instead.")
    return HaliosGuard(agent_id, api_key, base_url, parallel)


def parallel_guarded_chat(**config):
    """
    DEPRECATED: Decorator factory for parallel guarded chat

    This function creates a decorator for parallel guardrail processing.
    It is deprecated in favor of the unified @guarded_chat_completion decorator.

    Args:
        **config: Configuration parameters including:
            - app_id: Agent ID (required)
            - api_key: API key (optional)
            - base_url: Base URL (optional)
            - streaming: Enable streaming support (optional)

    Returns:
        Decorator function that wraps async functions with parallel guardrail processing

    Migration:
        Instead of: @parallel_guarded_chat(app_id="your-agent")
        Use: @guarded_chat_completion(agent_id="your-agent", concurrent_guardrail_processing=True)
    """
    logger.warning("parallel_guarded_chat() is deprecated. Use guarded_chat_completion(concurrent_guardrail_processing=True) instead.")
    # Map to new decorator
    streaming = config.pop('streaming', False)
    app_id = config.pop('app_id')

    return guarded_chat_completion(
        agent_id=app_id,
        concurrent_guardrail_processing=True,
        streaming_guardrails=streaming,
        **config
    )


def streaming_guarded_chat(app_id: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                          stream_buffer_size: int = 50, stream_check_interval: float = 0.5,
                          guardrail_timeout: float = 5.0):
    """
    Decorator factory for streaming guarded chat (Legacy - use guarded_chat_completion instead)
    """
    logger.warning("streaming_guarded_chat() is deprecated. Use guarded_chat_completion(streaming_guardrails=True) instead.")
    
    return guarded_chat_completion(
        agent_id=app_id,
        api_key=api_key,
        base_url=base_url,
        concurrent_guardrail_processing=True,
        streaming_guardrails=True,
        stream_buffer_size=stream_buffer_size,
        stream_check_interval=stream_check_interval,
        guardrail_timeout=guardrail_timeout
    )


# =============================================================================
# LEGACY PATCHING FUNCTIONS - DEPRECATED
# =============================================================================
# These functions are deprecated in favor of the unified decorator approach.
# They perform monkey-patching of LLM client libraries, which is less maintainable
# and can cause issues with library updates.
#
# Migration Guide:
#   OLD: patch_all(agent_id="your-agent")
#   NEW: @guarded_chat_completion(agent_id="your-agent")
#
# Benefits of new approach:
# - No monkey-patching required
# - Better type safety and IDE support
# - Cleaner error handling
# - More predictable behavior
# =============================================================================

def patch_openai(guard_instance: HaliosGuard):
    """
    DEPRECATED: Patch OpenAI client with HaliosAI guardrails

    This function monkey-patches the OpenAI library to automatically apply
    guardrails to all chat completion calls. This approach is deprecated
    in favor of the @guarded_chat_completion decorator.

    Args:
        guard_instance: HaliosGuard instance to use for patching

    Warning:
        This function modifies global state and may cause issues with
        OpenAI library updates or other code that depends on the original
        OpenAI client behavior.
    """
    logger.warning("patch_openai() is deprecated. Use @guarded_chat_completion decorator instead.")
    try:
        import openai
        guard_instance.patch(openai.OpenAI.chat.completions, 'create')
        guard_instance.patch(openai.AsyncOpenAI.chat.completions, 'create')
        logger.info("OpenAI client patched with HaliosAI guardrails")
    except ImportError:
        logger.warning("OpenAI not installed, skipping OpenAI patching")


def patch_anthropic(guard_instance: HaliosGuard):
    """
    DEPRECATED: Patch Anthropic client with HaliosAI guardrails

    This function monkey-patches the Anthropic library to automatically apply
    guardrails to all message calls. This approach is deprecated in favor of
    the @guarded_chat_completion decorator.

    Args:
        guard_instance: HaliosGuard instance to use for patching

    Warning:
        This function modifies global state and may cause issues with
        Anthropic library updates or other code that depends on the original
        Anthropic client behavior.
    """
    logger.warning("patch_anthropic() is deprecated. Use @guarded_chat_completion decorator instead.")
    try:
        import anthropic
        guard_instance.patch(anthropic.Anthropic.messages, 'create')
        guard_instance.patch(anthropic.AsyncAnthropic.messages, 'create')
        logger.info("Anthropic client patched with HaliosAI guardrails")
    except ImportError:
        logger.warning("Anthropic not installed, skipping Anthropic patching")


def patch_all(app_id: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """
    DEPRECATED: Auto-patch all detected LLM clients and frameworks

    This function automatically detects and patches all available LLM client
    libraries (OpenAI, Anthropic, etc.) with HaliosAI guardrails. This approach
    is deprecated in favor of explicit decorator usage.

    Args:
        app_id: Agent ID for guardrail configuration
        api_key: API key (defaults to HALIOS_API_KEY env var)
        base_url: Base URL for guardrails API (defaults to HALIOS_BASE_URL env var)

    Returns:
        HaliosGuard instance (for advanced usage)

    Warning:
        This function performs global monkey-patching and may interfere with
        other code that depends on unmodified LLM client behavior. Use the
        @guarded_chat_completion decorator instead for better control and
        maintainability.
    """
    logger.warning("patch_all() is deprecated. Use @guarded_chat_completion decorator instead.")
    halios_guard = guard(app_id, api_key, base_url)
    patch_openai(halios_guard)
    patch_anthropic(halios_guard)

    return halios_guard
