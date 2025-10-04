#!/usr/bin/env python3
"""
HaliosAI SDK - Context Manager Pattern

Demonstrates:
- Manual guardrail evaluation using context manager
- Separate request/response validation
- Convenience methods (validate_request, validate_response)

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
from haliosai import HaliosGuard, GuardrailViolation

# Validate required environment variables
REQUIRED_VARS = ["HALIOS_API_KEY", "HALIOS_AGENT_ID", "OPENAI_API_KEY"]
missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID")

async def main():
    # 👇 CUSTOMIZE THESE MESSAGES FOR YOUR AGENT'S PERSONA
    request_messages = [{"role": "user", "content": "Hello, how can you help?"}]
    
    async with HaliosGuard(agent_id=HALIOS_AGENT_ID) as guard:
        try:
            # Validate request
            await guard.validate_request(request_messages)
            print("✓ Request passed")
            
            # Call LLM
            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=request_messages,
                max_tokens=100
            )
            
            # Validate response
            response_content = response.choices[0].message.content
            full_conversation = request_messages + [{"role": "assistant", "content": response_content}]
            await guard.validate_response(full_conversation)
            print("✓ Response passed")
            print(f"Response: {response_content}")
            
        except GuardrailViolation as e:
            print(f"✗ Blocked: {e.violation_type} - {len(e.violations)} violation(s)")

if __name__ == "__main__":
    asyncio.run(main())
