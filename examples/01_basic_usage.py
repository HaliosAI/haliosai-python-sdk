#!/usr/bin/env python3
"""
HaliosAI SDK - Example 1: Basic Usage

equirements:
    pip install haliosai
    pip install openai

Environment Variables:
    HALIOS_API_KEY: Your HaliosAI API key
    HALIOS_AGENT_ID: Your agent ID
    OPENAI_API_KEY: Your OpenAI API key

"""

import asyncio
import os
from openai import AsyncOpenAI
from haliosai import guarded_chat_completion

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "demo-key")
HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID", "demo-agent")

# =============================================================================
# SUPPORTED FUNCTION SIGNATURES for decorator
# =============================================================================
"""
Supported function signatures for @guarded_chat_completion:

1. Basic: messages as first positional argument
   @guarded_chat_completion(agent_id="...")
   async def func(messages): ...

2. With additional parameters:
   @guarded_chat_completion(agent_id="...")
   async def func(messages, model="gpt-4", temperature=0.7): ...

3. Messages as keyword argument:
   @guarded_chat_completion(agent_id="...")
   async def func(**kwargs):  # messages in kwargs

4. Tool calling support:
   @guarded_chat_completion(agent_id="...")
   async def func(messages, tools=None): ...

5. Streaming (returns async generator):
   @guarded_chat_completion(agent_id="...", streaming_guardrails=True)
   async def func(messages): yield chunk
"""

# =============================================================================
# PARALLEL PROCESSING (DEFAULT) - Guardrails run concurrently with LLM
# =============================================================================

@guarded_chat_completion(agent_id=HALIOS_AGENT_ID)
async def parallel_call(messages):
    """Parallel processing: Guardrails run at the same time as LLM call"""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100
    )

# =============================================================================
# SEQUENTIAL PROCESSING - Guardrails run before LLM call
# =============================================================================

@guarded_chat_completion(agent_id=HALIOS_AGENT_ID, concurrent_guardrail_processing=False)
async def sequential_call(messages):
    """Sequential processing: Guardrails run before LLM call"""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100
    )

async def main():
    """Run both parallel and sequential examples"""
    print("🚀 HaliosAI Basic Usage - Parallel vs Sequential")
    print("=" * 50)

    # Single example message
    messages = [{"role": "user", "content": "Hello! How are you?"}]

    # Test Parallel Processing (Default)
    print("\n1️⃣  PARALLEL PROCESSING (Default)")
    print("   Guardrails run concurrently with LLM call")
    print("-" * 40)

    try:
        response = await parallel_call(messages)
        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
            print(f"✅ Parallel: {content}")
        else:
            print("✅ Parallel: Response received")
    except Exception as e:
        print(f"❌ Parallel Error: {e}")

    # Test Sequential Processing
    print("\n2️⃣  SEQUENTIAL PROCESSING")
    print("   Guardrails run before LLM call")
    print("-" * 40)

    try:
        response = await sequential_call(messages)
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            print(f"✅ Sequential: {content}")
        else:
            print("✅ Sequential: Response received")
    except ValueError as e:
            print(f"❌ Sequential Error: {e}")
    except Exception as e:
        print(f"❌ Sequential Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("✨ Examples completed!")
    print("\nKey Differences:")
    print("• Parallel: Faster (concurrent execution)")
    print("• Sequential: Safer (guardrails complete first)")
    print("• Both protect against input/output violations")

if __name__ == "__main__":
    asyncio.run(main())
