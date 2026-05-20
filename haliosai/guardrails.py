"""Guardrail evaluation, decorator, and streaming helpers."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from functools import wraps
from typing import Any, Callable

from ._transport import HaliosTransport
from .exceptions import GuardrailTriggered
from .types import (
    CheckResult,
    EvaluateResult,
    ExecutionResult,
    GuardedResponse,
    GuardrailPolicy,
    Violation,
    ViolationAction,
)

logger = logging.getLogger("haliosai")


# ---------------------------------------------------------------------------
# Core evaluate (async + sync)
# ---------------------------------------------------------------------------


async def _evaluate(
    transport: HaliosTransport,
    *,
    agent_id: str,
    messages: list[dict[str, Any]],
    mode: str = "guardrail",
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    tags: list[str] | None = None,
) -> EvaluateResult:
    """Call ``POST /api/<version>/evaluate`` and return typed result.

    The SDK will use the runtime API prefix detected from the client's
    `base_url` (``transport.api_prefix``), so requests may be sent to
    ``/api/v1`` or ``/api/v2`` depending on configuration.
    
    If ``trace_id`` is provided but ``span_id`` is not, a unique span_id
    is automatically generated. This ensures each evaluate() call within
    a trace gets its own span for proper observability.
    
    Use ``parent_span_id`` to create span hierarchies (e.g., link output
    span to input span for the same conversation turn).
    """
    # Auto-generate span_id if trace_id provided but span_id is not
    if trace_id and not span_id:
        span_id = secrets.token_hex(8)
    
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "messages": messages,
        "mode": mode,
    }
    if trace_id:
        payload["trace_id"] = trace_id
    if span_id:
        payload["span_id"] = span_id
    if parent_span_id:
        payload["parent_span_id"] = parent_span_id
    if tags:
        payload["tags"] = tags

    resp = await transport.request("POST", f"{transport.api_prefix}/evaluate", json=payload)
    return EvaluateResult.model_validate(resp.json())


def _evaluate_sync(
    transport: HaliosTransport,
    *,
    agent_id: str,
    messages: list[dict[str, Any]],
    mode: str = "guardrail",
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    tags: list[str] | None = None,
) -> EvaluateResult:
    """Synchronous version of evaluate for notebooks / PySpark."""
    # Auto-generate span_id if trace_id provided but span_id is not
    if trace_id and not span_id:
        span_id = secrets.token_hex(8)
    
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "messages": messages,
        "mode": mode,
    }
    if trace_id:
        payload["trace_id"] = trace_id
    if span_id:
        payload["span_id"] = span_id
    if parent_span_id:
        payload["parent_span_id"] = parent_span_id
    if tags:
        payload["tags"] = tags

    resp = transport.request_sync("POST", f"{transport.api_prefix}/evaluate", json=payload)
    return EvaluateResult.model_validate(resp.json())


# ---------------------------------------------------------------------------
# Validate helper (evaluate + raise on trigger)
# ---------------------------------------------------------------------------


async def _validate(
    transport: HaliosTransport,
    *,
    agent_id: str,
    messages: list[dict[str, Any]],
    invocation_type: str = "request",
    mode: str = "guardrail",
    trace_id: str | None = None,
    span_id: str | None = None,
    tags: list[str] | None = None,
    guardrail_policies: dict[str, GuardrailPolicy] | None = None,
) -> EvaluateResult:
    """Evaluate and raise :class:`GuardrailTriggered` if blocked."""
    result = await _evaluate(
        transport,
        agent_id=agent_id,
        messages=messages,
        mode=mode,
        trace_id=trace_id,
        span_id=span_id,
        tags=tags,
    )
    action = _resolve_action(result, guardrail_policies)
    if action == ViolationAction.BLOCK:
        raise GuardrailTriggered(
            f"Content blocked by guardrails: {[v.check_name for v in result.violations]}",
            violation_type=invocation_type,
            violations=result.violations,
            action=result.action,
            scan_result=result,
        )
    return result


# ---------------------------------------------------------------------------
# Violation / policy resolution
# ---------------------------------------------------------------------------


def _resolve_action(
    result: EvaluateResult,
    policies: dict[str, GuardrailPolicy] | None = None,
) -> ViolationAction:
    """Determine the final action given evaluate result and per-check policies."""
    if not result.triggered or not result.violations:
        return ViolationAction.PASS

    if not policies:
        return ViolationAction.BLOCK

    actions: list[ViolationAction] = []
    for v in result.violations:
        policy = policies.get(v.check_name)
        if policy == GuardrailPolicy.RECORD_ONLY:
            actions.append(ViolationAction.ALLOW_OVERRIDE)
        else:
            actions.append(ViolationAction.BLOCK)

    if ViolationAction.BLOCK in actions:
        return ViolationAction.BLOCK
    return ViolationAction.ALLOW_OVERRIDE


# ---------------------------------------------------------------------------
# Response extraction helpers (ported from V1)
# ---------------------------------------------------------------------------


def extract_response_message(response: Any) -> dict[str, Any]:
    """Extract an OpenAI-format message dict from an LLM response object."""
    # OpenAI response object
    if hasattr(response, "choices") and response.choices:
        if hasattr(response.choices[0], "message"):
            msg = response.choices[0].message
            d: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return d
        if hasattr(response.choices[0], "text"):
            return {"role": "assistant", "content": response.choices[0].text}

    # Dict response
    if isinstance(response, dict):
        if "choices" in response:
            choice = response["choices"][0]
            msg_dict = choice.get("message", {})
            d = {"role": "assistant", "content": msg_dict.get("content")}
            if "tool_calls" in msg_dict:
                d["tool_calls"] = msg_dict["tool_calls"]
            return d
        for key in ("output", "text", "content"):
            if key in response:
                return {"role": "assistant", "content": response[key]}

    if isinstance(response, str):
        return {"role": "assistant", "content": response}

    return {"role": "assistant", "content": str(response)}


# ---------------------------------------------------------------------------
# @guarded decorator
# ---------------------------------------------------------------------------


def guarded(
    agent_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
    concurrent: bool = True,
    guardrail_timeout: float = 5.0,
    guardrail_policies: dict[str, GuardrailPolicy] | None = None,
    on_violation: Callable | None = None,
    streaming: bool = False,
    stream_buffer_size: int | None = None,
    stream_check_interval: float | None = None,
    tags: list[str] | None = None,
):
    """Decorator that wraps an async LLM function with guardrail evaluation.

    Usage::

        @guarded(agent_id="my-agent")
        async def call_llm(messages):
            return await openai.chat.completions.create(model="gpt-4o", messages=messages)

    Parameters:
        agent_id: HaliosAI agent slug or UUID.
        concurrent: If True, run guardrails in parallel with the LLM call.
        streaming: If True, treat the wrapped function as an async generator.
        guardrail_policies: Per-check policy overrides.
        on_violation: Optional callback invoked with the ``GuardrailTriggered`` exception.
        tags: Optional tags to attach to every evaluation from this decorator.
    """

    def decorator(func: Callable):
        if streaming:
            return _streaming_wrapper(
                func,
                agent_id=agent_id,
                api_key=api_key,
                base_url=base_url,
                guardrail_timeout=guardrail_timeout,
                guardrail_policies=guardrail_policies,
                on_violation=on_violation,
                stream_buffer_size=stream_buffer_size,
                stream_check_interval=stream_check_interval,
                tags=tags,
            )
        if concurrent:
            return _parallel_wrapper(
                func,
                agent_id=agent_id,
                api_key=api_key,
                base_url=base_url,
                guardrail_timeout=guardrail_timeout,
                guardrail_policies=guardrail_policies,
                on_violation=on_violation,
                tags=tags,
            )
        return _sequential_wrapper(
            func,
            agent_id=agent_id,
            api_key=api_key,
            base_url=base_url,
            guardrail_timeout=guardrail_timeout,
            guardrail_policies=guardrail_policies,
            on_violation=on_violation,
            tags=tags,
        )

    return decorator


# ---------------------------------------------------------------------------
# Internal wrapper implementations
# ---------------------------------------------------------------------------


def _make_transport(
    agent_id: str,
    api_key: str | None,
    base_url: str | None,
) -> HaliosTransport:
    """Build a transport from decorator kwargs (resolved lazily)."""
    import os

    return HaliosTransport(
        base_url=(base_url or os.getenv("HALIOS_BASE_URL", "https://api.halios.ai")).rstrip("/"),
        api_key=api_key or os.getenv("HALIOS_API_KEY", ""),
    )


def _extract_messages(args, kwargs) -> list[dict[str, Any]]:
    if args and isinstance(args[0], list):
        return args[0]
    if "messages" in kwargs:
        return kwargs["messages"]
    raise ValueError("Wrapped function must receive 'messages' as first arg or keyword arg")


def _parallel_wrapper(func, *, agent_id, api_key, base_url, guardrail_timeout, guardrail_policies, on_violation, tags):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        messages = _extract_messages(args, kwargs)
        transport = _make_transport(agent_id, api_key, base_url)

        try:
            # Launch request guardrails + LLM in parallel
            guardrail_task = asyncio.create_task(
                _evaluate(transport, agent_id=agent_id, messages=messages, tags=tags)
            )
            llm_task = asyncio.create_task(func(*args, **kwargs))

            # Wait for request guardrails first (with timeout)
            try:
                req_result = await asyncio.wait_for(guardrail_task, timeout=guardrail_timeout)
                action = _resolve_action(req_result, guardrail_policies)
                if action == ViolationAction.BLOCK:
                    llm_task.cancel()
                    exc = GuardrailTriggered(
                        f"Request blocked: {[v.check_name for v in req_result.violations]}",
                        violation_type="request",
                        violations=req_result.violations,
                        action=req_result.action,
                        scan_result=req_result,
                    )
                    if on_violation:
                        on_violation(exc)
                    raise exc
            except asyncio.TimeoutError:
                logger.warning("Request guardrails timed out after %.1fs", guardrail_timeout)

            # Wait for LLM
            llm_response = await llm_task

            # Post-response guardrails
            response_msg = extract_response_message(llm_response)
            full_conv = messages + [response_msg]
            resp_result = await _evaluate(
                transport, agent_id=agent_id, messages=full_conv, mode="guardrail", tags=tags
            )
            action = _resolve_action(resp_result, guardrail_policies)
            if action == ViolationAction.BLOCK:
                exc = GuardrailTriggered(
                    f"Response blocked: {[v.check_name for v in resp_result.violations]}",
                    violation_type="response",
                    violations=resp_result.violations,
                    action=resp_result.action,
                    scan_result=resp_result,
                )
                if on_violation:
                    on_violation(exc)
                raise exc

            return llm_response
        finally:
            await transport.aclose()

    return wrapper


def _sequential_wrapper(func, *, agent_id, api_key, base_url, guardrail_timeout, guardrail_policies, on_violation, tags):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        messages = _extract_messages(args, kwargs)
        transport = _make_transport(agent_id, api_key, base_url)

        try:
            # 1. Request guardrails
            req_result = await _evaluate(
                transport, agent_id=agent_id, messages=messages, tags=tags
            )
            action = _resolve_action(req_result, guardrail_policies)
            if action == ViolationAction.BLOCK:
                exc = GuardrailTriggered(
                    f"Request blocked: {[v.check_name for v in req_result.violations]}",
                    violation_type="request",
                    violations=req_result.violations,
                    action=req_result.action,
                    scan_result=req_result,
                )
                if on_violation:
                    on_violation(exc)
                raise exc

            # 2. LLM call
            llm_response = await func(*args, **kwargs)

            # 3. Response guardrails
            response_msg = extract_response_message(llm_response)
            full_conv = messages + [response_msg]
            resp_result = await _evaluate(
                transport, agent_id=agent_id, messages=full_conv, mode="guardrail", tags=tags
            )
            action = _resolve_action(resp_result, guardrail_policies)
            if action == ViolationAction.BLOCK:
                exc = GuardrailTriggered(
                    f"Response blocked: {[v.check_name for v in resp_result.violations]}",
                    violation_type="response",
                    violations=resp_result.violations,
                    action=resp_result.action,
                    scan_result=resp_result,
                )
                if on_violation:
                    on_violation(exc)
                raise exc

            return llm_response
        finally:
            await transport.aclose()

    return wrapper


def _streaming_wrapper(func, *, agent_id, api_key, base_url, guardrail_timeout, guardrail_policies, on_violation, stream_buffer_size, stream_check_interval, tags):
    buf_size = stream_buffer_size if stream_buffer_size is not None else 50
    check_interval = stream_check_interval if stream_check_interval is not None else 0.5

    async def wrapper(*args, **kwargs):
        messages = _extract_messages(args, kwargs)
        transport = _make_transport(agent_id, api_key, base_url)

        try:
            # Request guardrails
            try:
                req_result = await asyncio.wait_for(
                    _evaluate(transport, agent_id=agent_id, messages=messages, tags=tags),
                    timeout=guardrail_timeout,
                )
                action = _resolve_action(req_result, guardrail_policies)
                if action == ViolationAction.BLOCK:
                    exc = GuardrailTriggered(
                        f"Request blocked: {[v.check_name for v in req_result.violations]}",
                        violation_type="request",
                        violations=req_result.violations,
                        action=req_result.action,
                    )
                    if on_violation:
                        on_violation(exc)
                    raise exc
            except asyncio.TimeoutError:
                logger.warning("Request guardrails timed out, proceeding with stream")

            # Stream with periodic guardrail checks
            buffer = ""
            buffer_since_check = ""
            last_check = time.time()

            remaining_args = args
            async for chunk in func(*remaining_args, **kwargs):
                content = _extract_chunk_content(chunk)
                if content:
                    buffer += content
                    buffer_since_check += content

                yield chunk

                now = time.time()
                should_check = (
                    len(buffer_since_check) >= buf_size
                    or (now - last_check) >= check_interval
                )
                if should_check and buffer_since_check:
                    resp_result = await _evaluate(
                        transport,
                        agent_id=agent_id,
                        messages=messages + [{"role": "assistant", "content": buffer}],
                        mode="guardrail",
                        tags=tags,
                    )
                    action = _resolve_action(resp_result, guardrail_policies)
                    if action == ViolationAction.BLOCK:
                        exc = GuardrailTriggered(
                            "Response blocked during streaming",
                            violation_type="response",
                            violations=resp_result.violations,
                            action=resp_result.action,
                        )
                        if on_violation:
                            on_violation(exc)
                        raise exc
                    buffer_since_check = ""
                    last_check = now

            # Final check
            if buffer:
                resp_result = await _evaluate(
                    transport,
                    agent_id=agent_id,
                    messages=messages + [{"role": "assistant", "content": buffer}],
                    mode="guardrail",
                    tags=tags,
                )
                action = _resolve_action(resp_result, guardrail_policies)
                if action == ViolationAction.BLOCK:
                    exc = GuardrailTriggered(
                        "Response blocked after stream completion",
                        violation_type="response",
                        violations=resp_result.violations,
                        action=resp_result.action,
                    )
                    if on_violation:
                        on_violation(exc)
                    raise exc
        finally:
            await transport.aclose()

    return wrapper


def _extract_chunk_content(chunk: Any) -> str:
    """Extract text content from a streaming chunk."""
    if hasattr(chunk, "choices") and chunk.choices:
        if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
            return chunk.choices[0].delta.content or ""
    if isinstance(chunk, dict):
        if "choices" in chunk and chunk["choices"]:
            return chunk["choices"][0].get("delta", {}).get("content", "")
        return chunk.get("content", chunk.get("text", ""))
    if isinstance(chunk, str):
        return chunk
    return ""
