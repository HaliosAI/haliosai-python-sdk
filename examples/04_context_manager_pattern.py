#!/usr/bin/env python3
"""
HaliosAI SDK - Example 5: Context Manager Pattern

Shows minimal code changes needed to add guardrails using async context manager.
"""

import asyncio
import os
from openai import AsyncOpenAI
from haliosai import HaliosGuard

# Configuration
HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID", "demo-agent-context")

async def main():
    """Demonstrate minimal integration with real OpenAI calls"""
    print("🔐 HaliosAI Context Manager - Minimal Integration")
    print("=" * 50)

    # Minimal change: wrap your LLM calls with guardrails
    async with HaliosGuard(agent_id=HALIOS_AGENT_ID) as guard:
        client = AsyncOpenAI()

        messages = [{"role": "user", "content": "Explain quantum computing simply"}]

        # Evaluate request (minimal addition)
        req_result = await guard.evaluate(messages, "request")
        if req_result.get("guardrails_triggered", 0) > 0:
            print("🚫 Request blocked")
            return

        # Your existing LLM call (unchanged)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200
        )

        assistant_message = response.choices[0].message.content
        print(f"🤖 Assistant: {assistant_message}")

        # Evaluate response (minimal addition)
        resp_messages = messages + [{"role": "assistant", "content": assistant_message}]
        resp_result = await guard.evaluate(resp_messages, "response")
        if resp_result.get("guardrails_triggered", 0) > 0:
            print("🚫 Response blocked")
        else:
            print("✅ Response passed guardrails")

if __name__ == "__main__":
    # Set up demo environment if real credentials not provided
    if not os.getenv("HALIOS_API_KEY"):
        os.environ["HALIOS_API_KEY"] = "demo-key"
        os.environ["HALIOS_BASE_URL"] = "https://httpbin.org/status/200"
        print("🧪 Demo mode - using mock API responses")

    asyncio.run(main())
