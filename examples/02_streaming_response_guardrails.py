#!/usr/bin/env python3
"""
HaliosAI SDK - Streaming with Real-time Guardrails

Demonstrates:
- Streaming LLM responses with real-time guardrail evaluation
- Character-based buffering (stream_buffer_size)
- Time-based buffering (stream_check_interval)
- Hybrid buffering (both)

Setup Required:
1. Create agent in HaliosAI dashboard for YOUR use case
2. Configure guardrails appropriate for your agent's persona
3. Set environment variables:
   export HALIOS_API_KEY="your-key"
   export HALIOS_AGENT_ID="your-agent-id"
   export OPENAI_API_KEY="your-openai-key"

Buffering Options:
- stream_buffer_size=100, stream_check_interval=None  # Character-based only
- stream_buffer_size=None, stream_check_interval=2.0  # Time-based only
- stream_buffer_size=50, stream_check_interval=0.5    # Hybrid (default)

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

# Streaming with real-time guardrails
@guarded_chat_completion(
    agent_id=HALIOS_AGENT_ID,
    streaming_guardrails=True,
    stream_buffer_size=50,        # Check every 50 characters
    stream_check_interval=2.0      # OR every 2 seconds
)
async def stream_llm(messages):
    """Stream LLM response with real-time guardrail checks"""
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        stream=True
    )
    async for chunk in stream:
        yield chunk

async def main():
    # 👇 CUSTOMIZE THESE MESSAGES FOR YOUR AGENT'S PERSONA
    test_messages = [
        {"role": "user", "content": "Explain how you can help me."}
    ]
    
    try:
        print("Streaming: ", end="", flush=True)
        async for event in stream_llm(test_messages):
            if isinstance(event, dict) and event.get('type') == 'chunk':
                print(event['content'], end='', flush=True)
            elif isinstance(event, dict) and event.get('type') == 'completed':
                print("\n✓ Stream completed")
    except GuardrailViolation as e:
        print(f"\n✗ Blocked: {e.violation_type} - {len(e.violations)} violation(s)")

if __name__ == "__main__":
    asyncio.run(main())
