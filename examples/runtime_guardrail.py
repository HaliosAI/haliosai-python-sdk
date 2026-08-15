"""Explicit request/response guardrails at an application boundary."""

import haliosai


async def application_agent(_messages: list[dict[str, str]]) -> str:
    """Replace with the application's real agent call."""
    raise NotImplementedError


async def guarded_answer(messages: list[dict[str, str]]) -> str:
    async with haliosai.Client(agent_id="support-agent") as client:
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
