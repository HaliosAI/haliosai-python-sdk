"""
HaliosAI SDK - Core Client Module

This module provides the main HaliosGuard class and supporting utilities for
integrating AI guardrails with LLM applications.
"""

import asyncio
import httpx
import os
import time
import logging
import inspect
from typing import List, Dict, Any, Callable, Optional
from functools import wraps
from enum import Enum
from dataclasses import dataclass


# Configure module logger
logger = logging.getLogger(__name__)


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
    Main HaliosAI guardrails client
    
    Provides simple patching and wrapper utilities for adding AI guardrails
    to LLM function calls. Supports both sequential and parallel execution modes.
    """
    
    def __init__(self, agent_id: str, api_key: str = None, base_url: str = None, parallel: bool = False):
        """
        Initialize HaliosGuard
        
        Args:
            agent_id: Agent ID for guardrail configuration
            api_key: API key (defaults to HALIOS_API_KEY env var)
            base_url: Base URL for guardrails API (defaults to HALIOS_BASE_URL env var)
            parallel: Enable parallel execution of guardrails and LLM calls
        """
        self.agent_id = agent_id
        self.api_key = api_key or os.getenv("HALIOS_API_KEY")
        self.base_url = base_url or os.getenv("HALIOS_BASE_URL", "http://localhost:2000")
        self.parallel = parallel
        
        if not self.api_key:
            logger.warning("No API key provided. Set HALIOS_API_KEY environment variable or pass api_key parameter")
        
        logger.debug(f"Initialized HaliosGuard with agent_id={agent_id}, parallel={parallel}")
        
    async def evaluate(self, messages: List[Dict], invocation_type: str = "request") -> Dict:
        """
        Evaluate messages against configured guardrails
        
        Args:
            messages: List of chat messages to evaluate
            invocation_type: Type of evaluation ("request" or "response")
            
        Returns:
            Dict containing guardrail evaluation results
        """
        logger.debug(f"Evaluating {len(messages)} messages with type={invocation_type}")
        
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
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
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
                    analysis = v.get('analysis', {})
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
    
    def patch_function(self, original_func: Callable) -> Callable:
        """
        Create a guarded version of any async function
        
        Args:
            original_func: Original async function to wrap with guardrails
            
        Returns:
            Wrapped function with guardrail protection
        """
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
                # Parallel execution: run request guardrails and LLM call simultaneously
                logger.debug("Running request guardrails and LLM call in parallel")
                request_start = time.time()
                llm_start = time.time()
                
                request_task = asyncio.create_task(self.evaluate(messages, "request"))
                llm_task = asyncio.create_task(original_func(*args, **kwargs))
                
                # Wait for both to complete
                try:
                    request_result, response = await asyncio.gather(request_task, llm_task)
                    
                    request_time = time.time() - request_start
                    llm_time = time.time() - llm_start
                    logger.debug("Parallel execution completed: request=%.3fs, llm=%.3fs", request_time, llm_time)
                    
                    # Check request violations
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
                        
                except Exception:
                    # If request guardrails fail, cancel LLM task if possible
                    if not llm_task.done():
                        llm_task.cancel()
                        logger.debug("Cancelled LLM task due to guardrail failure")
                    raise
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


class ParallelGuardedChat:
    """
    Advanced chat client with parallel guardrail processing
    
    This class provides sophisticated parallel processing where:
    1. Request guardrails run in parallel with the LLM API call
    2. If request guardrails fail first, the LLM call is aborted
    3. If LLM completes first, we wait for request guardrails (configurable timeout)
    4. Response guardrails are evaluated synchronously after LLM completion
    
    This pattern optimizes for performance while maintaining safety.
    """
    
    def __init__(self, agent_id: str, api_key: str = None, base_url: str = None, 
                 guardrail_timeout: float = 5.0, streaming: bool = False, 
                 stream_buffer_size: int = 50, stream_check_interval: float = 0.5):
        """
        Initialize ParallelGuardedChat
        
        Args:
            agent_id: Agent ID for guardrail configuration
            api_key: API key (defaults to HALIOS_API_KEY env var)
            base_url: Base URL for guardrails API (defaults to HALIOS_BASE_URL env var)
            guardrail_timeout: Timeout for guardrail operations
            streaming: Enable streaming mode
            stream_buffer_size: Characters to buffer before guardrail check
            stream_check_interval: Time interval for guardrail checks
        """
        self.agent_id = agent_id
        self.api_key = api_key or os.getenv("HALIOS_API_KEY")
        self.base_url = base_url or os.getenv("HALIOS_BASE_URL", "http://localhost:2000")
        self.guardrail_timeout = guardrail_timeout
        self.streaming = streaming
        self.stream_buffer_size = stream_buffer_size
        self.stream_check_interval = stream_check_interval
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        logger.debug("Initialized ParallelGuardedChat with agent_id=%s, streaming=%s", agent_id, streaming)
    
    async def evaluate_guardrails(self, messages: List[Dict], invocation_type: str) -> Dict:
        """
        Evaluate content against guardrails
        
        Args:
            messages: List of chat messages to evaluate
            invocation_type: Type of evaluation ("request" or "response")
            
        Returns:
            Dict containing guardrail evaluation results
        """
        url = f"{self.base_url}/api/v3/agents/{self.agent_id}/evaluate"
        
        headers = {
            "X-HALIOS-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": messages,
            "invocation_type": invocation_type
        }
        
        try:
            response = await self.http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            logger.debug("Guardrail evaluation (%s): %s triggered", invocation_type, result.get('guardrails_triggered', 0))
            return result
            
        except httpx.HTTPError as e:
            logger.error("HTTP error during guardrail evaluation: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during guardrail evaluation: %s", e)
            raise
    
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
            async for chunk in llm_stream_func(*args, **kwargs):
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
        """Clean up resources"""
        await self.http_client.aclose()
        logger.debug("ParallelGuardedChat resources cleaned up")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()




# Unified decorator for all chat completion guardrail functionality
def guarded_chat_completion(
    agent_id: str, 
    api_key: str = None, 
    base_url: str = None,
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
                
                # Create ParallelGuardedChat instance and stream
                config = {
                    'agent_id': agent_id,
                    'api_key': api_key,
                    'base_url': base_url,
                    'streaming': True,
                    'stream_buffer_size': stream_buffer_size,
                    'stream_check_interval': stream_check_interval,
                    'guardrail_timeout': guardrail_timeout
                }
                async with ParallelGuardedChat(**config) as guard_client:
                    async for event in guard_client.guarded_stream_parallel(messages, func, *args, **kwargs):
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
                
                if concurrent_guardrail_processing:
                    # Use ParallelGuardedChat for concurrent processing
                    config = {
                        'agent_id': agent_id,
                        'api_key': api_key,
                        'base_url': base_url,
                        'streaming': False,
                        'guardrail_timeout': guardrail_timeout
                    }
                    async with ParallelGuardedChat(**config) as guard_client:
                        return await guard_client.guarded_call_parallel(messages, func, *args, **kwargs)
                else:
                    # Use HaliosGuard for sequential processing
                    guard_instance = HaliosGuard(agent_id, api_key, base_url, parallel=False)
                    guarded_func = guard_instance.patch_function(func)
                    return await guarded_func(*args, **kwargs)
            return wrapper
    return decorator


# Legacy compatibility functions - these will be deprecated
def guard(agent_id: str, api_key: str = None, base_url: str = None, parallel: bool = False):
    """
    Create a HaliosGuard instance (Legacy - use guarded_chat_completion decorator instead)
    
    Args:
        agent_id: Agent ID for guardrail configuration
        api_key: API key (defaults to HALIOS_API_KEY env var)
        base_url: Base URL for guardrails API (defaults to HALIOS_BASE_URL env var)
        parallel: Enable parallel execution of guardrails and LLM calls
        
    Returns:
        HaliosGuard instance that can be used as decorator or patcher
    """
    logger.warning("guard() is deprecated. Use guarded_chat_completion() decorator instead.")
    return HaliosGuard(agent_id, api_key, base_url, parallel)


def parallel_guarded_chat(**config):
    """
    Decorator factory for parallel guarded chat (Legacy - use guarded_chat_completion instead)
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


def streaming_guarded_chat(app_id: str, api_key: str = None, base_url: str = None,
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


# Utility functions for client patching
def patch_openai(guard_instance: HaliosGuard):
    """
    Patch OpenAI client with HaliosAI guardrails
    
    Args:
        guard_instance: HaliosGuard instance to use for patching
    """
    try:
        import openai
        guard_instance.patch(openai.OpenAI.chat.completions, 'create')
        guard_instance.patch(openai.AsyncOpenAI.chat.completions, 'create')
        logger.info("OpenAI client patched with HaliosAI guardrails")
    except ImportError:
        logger.warning("OpenAI not installed, skipping OpenAI patching")


def patch_anthropic(guard_instance: HaliosGuard):
    """
    Patch Anthropic client with HaliosAI guardrails
    
    Args:
        guard_instance: HaliosGuard instance to use for patching
    """
    try:
        import anthropic
        guard_instance.patch(anthropic.Anthropic.messages, 'create')
        guard_instance.patch(anthropic.AsyncAnthropic.messages, 'create')
        logger.info("Anthropic client patched with HaliosAI guardrails")
    except ImportError:
        logger.warning("Anthropic not installed, skipping Anthropic patching")


def patch_all(app_id: str, api_key: str = None, base_url: str = None):
    """
    Auto-patch all detected LLM clients and frameworks
    
    Args:
        app_id: Agent ID for guardrail configuration
        api_key: API key (defaults to HALIOS_API_KEY env var)
        base_url: Base URL for guardrails API (defaults to HALIOS_BASE_URL env var)
        
    Returns:
        Dict containing the guard instance and any framework patchers
    """
    halios_guard = guard(app_id, api_key, base_url)
    patch_openai(halios_guard)
    patch_anthropic(halios_guard)
    
    return halios_guard
