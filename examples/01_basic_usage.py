#!/usr/bin/env python3
"""
HaliosAI SDK - Basic Usage

Demonstrates:
- Decorator pattern for guardrails
- Request/response evaluation
- Exception handling for blocked content

Setup Required:
1. Create agent in HaliosAI dashboard for YOUR use case
2. Configure guardrails appropriate for your agent's persona
3. Set environment variables:
   export HALIOS_API_KEY="your-key"
   export HALIOS_AGENT_ID="your-agent-id"
   export OPENAI_API_KEY="your-openai-key"

💡 For interactive testing: See ../halios_sdk_test/halios_demo.py

⚠️  Update test messages below to match YOUR agent's persona!
"""

import asyncio
import os
from openai import AsyncOpenAI
from haliosai import guarded_chat_completion, GuardrailViolation

# Validate required environment variables
REQUIRED_VARS = ["HALIOS_API_KEY", "HALIOS_AGENT_ID", "OPENAI_API_KEY"]
missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID")

@guarded_chat_completion(agent_id=HALIOS_AGENT_ID)
async def call_llm(messages):
    """Basic LLM call with guardrails"""
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100
    )
    return response

async def main():
    # 👇 CUSTOMIZE THESE MESSAGES FOR YOUR AGENT'S PERSONA
    test_messages = [
        {"role": "user", "content": "Hello, can you help me?"}
    ]
    
    try:
        response = await call_llm(test_messages)
        content = response.choices[0].message.content
        print(f"✓ Response: {content}")
    except GuardrailViolation as e:
        print(f"✗ Blocked: {e.violation_type} - {len(e.violations)} violation(s)")

if __name__ == "__main__":
    asyncio.run(main())
