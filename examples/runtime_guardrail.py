"""Explicit request/response guardrails at an application boundary."""

import asyncio
import os

import haliosai


async def application_agent(messages: list[dict[str, str]]) -> str:
    """Replace with the application's real agent call."""
    return f"Assistant response to: {messages[-1]['content']}"


async def guarded_answer(messages: list[dict[str, str]]) -> str:
    async with haliosai.Client(agent_id=os.getenv("HALIOS_AGENT_ID", "support-agent")) as client:
        request_check = await client.evaluate_request(messages)
        if request_check.blocked:
            raise PermissionError("Halios blocked the request")

        response = await application_agent(messages)
        response_check = await client.evaluate_response(
            [*messages, {"role": "assistant", "content": response}]
        )
        if response_check.blocked:
            raise PermissionError("Halios blocked the response")
        return response


if __name__ == "__main__":
    sample_messages = [{"role": "user", "content": "Hello, can you check my order status?"}]
    if os.getenv("HALIOS_API_KEY"):
        result = asyncio.run(guarded_answer(sample_messages))
        print("Guarded response:", result)
    else:
        print("Set HALIOS_API_KEY to run inline guardrail checks against Halios cloud.")
