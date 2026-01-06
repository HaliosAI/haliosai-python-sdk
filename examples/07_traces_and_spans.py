"""
Trace and Span Propagation Examples

This example demonstrates different ways to use traces and spans with HaliosGuard:

1. Decorator Pattern with Explicit Trace Passing:
   - Use @guarded_chat_completion decorator
   - Pass trace_context explicitly to each call
   - Suitable for simple cases or when you control trace lifecycle

2. Context Manager with Constructor Trace ID and Explicit Spans:
   - Construct HaliosGuard with trace_id for session-level trace
   - Use start_span/end_span for per-turn or per-operation spans
   - Best for multi-turn conversations with detailed observability

3. Automatic Spans (SDK-Managed):
   - Let the SDK create traces and spans automatically
   - No explicit span management needed
   - Good for quick integration or when you don't need custom spans

Prerequisites:
- HALIOS_API_KEY, HALIOS_AGENT_ID, OPENAI_API_KEY must be set in the environment
- Optional: HALIOS_BASE_URL to point at a non-default API host

Run:
    python examples/07_traces_and_spans.py
"""

import asyncio
import os
from contextlib import asynccontextmanager

from openai import AsyncOpenAI

from haliosai import HaliosGuard, ExecutionResult, GuardrailViolation, TraceContext, guarded_chat_completion

HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID")
HALIOS_API_KEY = os.getenv("HALIOS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REQUIRED = {"HALIOS_AGENT_ID": HALIOS_AGENT_ID, "HALIOS_API_KEY": HALIOS_API_KEY, "OPENAI_API_KEY": OPENAI_API_KEY}
missing = [name for name, value in REQUIRED.items() if not value]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


@guarded_chat_completion(
    agent_id=HALIOS_AGENT_ID,
    api_key=HALIOS_API_KEY,
    concurrent_guardrail_processing=True,
)
async def ask_agent(messages):
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=120,
    )


async def ask_agent_raw(messages):
    """Raw LLM call used with HaliosGuard context manager."""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=120,
    )


def build_trace_context(conversation_id: str) -> TraceContext:
    """Helper to create a trace context for a conversation/session."""
    return TraceContext.create(conversation_id=conversation_id)


@asynccontextmanager
async def trace_session(conversation_id: str):
    """Context manager that keeps a trace across multiple turns."""
    trace = build_trace_context(conversation_id=conversation_id)
    try:
        yield trace
    finally:
        # No teardown needed now, but this keeps the pattern extensible.
        pass


async def main():
    # Example 1: Decorator Pattern with Explicit Trace Passing
    print("=== Example 1: Decorator Pattern with Explicit Trace Passing ===")
    async with trace_session(conversation_id="demo-session-001") as trace_context:
        # Turn 1
        history = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Give me two bullet points about Mars exploration."},
        ]

        try:
            first = await ask_agent(history, trace_context=trace_context)
            first_content = first.choices[0].message.content
            print("Turn 1:\n", first_content)
            history.append({"role": "assistant", "content": first_content})

            # Turn 2 reuses the same trace_context so all guardrail calls share the trace
            history.append({"role": "user", "content": "Give me one follow-up fact."})
            second = await ask_agent(history, trace_context=trace_context)
            second_content = second.choices[0].message.content
            print("\nTurn 2:\n", second_content)
        except GuardrailViolation as violation:
            print(f"Blocked at stage={violation.violation_type}: {len(violation.violations)} violation(s)")
            for v in violation.violations:
                print(" -", v.guardrail_type)

    # Example 2: Context Manager with Constructor Trace ID and Explicit Spans
    print("\n=== Example 2: Context Manager with Constructor Trace ID and Explicit Spans ===")
    # Construct guard with a stable trace id for this session
    # IMPORTANT: trace_id must be a valid W3C trace-id (32 lowercase hex characters)
    # For session tracking, use a hash of your session ID or generate a valid hex string
    import hashlib
    session_id = "my-app-session-abc-123"
    guard_trace_id = hashlib.sha256(session_id.encode()).hexdigest()[:32]  # First 32 chars of hash
    print(f"Using custom trace_id for session '{session_id}': {guard_trace_id}")
    
    async with HaliosGuard(agent_id=HALIOS_AGENT_ID, api_key=HALIOS_API_KEY, parallel=True, trace_id=guard_trace_id) as guard:
        # Start a higher-level turn span (start_span pushes to stack)
        turn_span = guard.start_span("turn.1", attributes={"turn": 1})
        print("Started span:", turn_span.span_id)

        # Turn 1
        history = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Give me two bullet points about Mars exploration."},
        ]

        try:
            res = await guard.guarded_call_parallel(history, ask_agent_raw)

            if res.result == ExecutionResult.SUCCESS:
                resp = res.final_response
                try:
                    content = resp.choices[0].message.content
                except Exception:
                    # Fallback for dict-like responses
                    content = (
                        resp.get("choices", [{}])[0].get("message", {}).get("content")
                        if isinstance(resp, dict)
                        else str(resp)
                    )

                print("Context (constructor trace) - Turn 1:\n", content)
                history.append({"role": "assistant", "content": content})

                # End turn span and log payload
                payload = guard.end_span(turn_span)
                print("Ended span payload:", payload)

                # Turn 2 with explicit per-turn span
                turn2 = guard.start_span("turn.2", attributes={"turn": 2})
                history.append({"role": "user", "content": "Give me one follow-up fact."})
                res2 = await guard.guarded_call_parallel(history, ask_agent_raw)
                if res2.result == ExecutionResult.SUCCESS:
                    resp2 = res2.final_response
                    try:
                        content2 = resp2.choices[0].message.content
                    except Exception:
                        content2 = (
                            resp2.get("choices", [{}])[0].get("message", {}).get("content")
                            if isinstance(resp2, dict)
                            else str(resp2)
                        )
                    print("\nContext (constructor trace) - Turn 2:\n", content2)

                    payload2 = guard.end_span(turn2)
                    print("Ended span payload:", payload2)
                else:
                    print("Turn 2 blocked or error:", res2.result)
            else:
                print("Turn 1 blocked or error:", res.result)
        except GuardrailViolation as violation:
            print(f"Blocked at stage={violation.violation_type}: {len(violation.violations)} violation(s)")
            for v in violation.violations:
                print(" -", v.guardrail_type)

    # Example 3: Automatic Spans (SDK-Managed)
    print("\n=== Example 3: Automatic Spans (SDK-Managed) ===")
    async with HaliosGuard(agent_id=HALIOS_AGENT_ID, api_key=HALIOS_API_KEY, parallel=True) as guard_auto:
        # Ensure session-level trace exists
        print("Auto session trace id:", guard_auto.trace_context.trace_id)

        # Evaluate request (SDK will create a short-lived request span)
        req_res = await guard_auto.evaluate(history, "request")
        print("Automatic request span:", req_res.span_name, req_res.span_id)

        # Simulate response evaluation (would normally follow an LLM call)
        resp_res = await guard_auto.evaluate(history + [{"role": "assistant", "content": "Hi"}], "response")
        print("Automatic response span:", resp_res.span_name, "parent:", resp_res.parent_span_id)


if __name__ == "__main__":
    asyncio.run(main())
