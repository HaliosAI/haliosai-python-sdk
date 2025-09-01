#!/usr/bin/env python3
"""
Migration Guide: Old vs New HaliosAI Decorator Syntax

This example shows how to migrate from the old multiple decorators 
to the new unified guarded_chat_completion decorator.
"""

import asyncio
from typing import List, Dict, Any

# OLD SYNTAX (still works but deprecated)
print("📚 OLD SYNTAX Examples (deprecated):")

# Example 1: Basic guard decorator
try:
    from haliosai import guard
    
    print("\n1️⃣  OLD: @guard decorator")
    print("""
    from haliosai import guard

    my_guard = guard(app_id="your-app-id", parallel=True)

    @my_guard
    async def call_llm(messages):
        return await openai_client.chat.completions.create(...)
    """)
except ImportError:
    print("❌ guard not available")

# Example 2: Parallel guarded chat
try:
    from haliosai import parallel_guarded_chat
    
    print("\n2️⃣  OLD: @parallel_guarded_chat decorator")
    print("""
    from haliosai import parallel_guarded_chat

    @parallel_guarded_chat(app_id="your-app-id", parallel=True)
    async def fast_llm_call(messages):
        return await openai_client.chat.completions.create(...)
    """)
except ImportError:
    print("❌ parallel_guarded_chat not available")

# Example 3: Streaming guarded chat
try:
    from haliosai import streaming_guarded_chat
    
    print("\n3️⃣  OLD: @streaming_guarded_chat decorator")
    print("""
    from haliosai import streaming_guarded_chat

    @streaming_guarded_chat(app_id="your-app-id", stream_buffer_size=100)
    async def stream_llm_call(messages):
        async for chunk in openai_client.chat.completions.create(..., stream=True):
            yield chunk
    """)
except ImportError:
    print("❌ streaming_guarded_chat not available")

print("\n" + "="*70)

# NEW UNIFIED SYNTAX (recommended)
print("\n🚀 NEW UNIFIED SYNTAX (recommended):")

try:
    from haliosai import guarded_chat_completion
    
    print("\n✨ UNIFIED: @guarded_chat_completion decorator")
    
    # Example 1: Basic usage (replaces @guard)
    print("\n1️⃣  NEW: Basic usage with concurrent processing (default)")
    print("""
    from haliosai import guarded_chat_completion

    @guarded_chat_completion(app_id="your-app-id")
    async def call_llm(messages):
        return await openai_client.chat.completions.create(...)
    """)
    
    # Example 2: Sequential processing (replaces @guard with parallel=False)
    print("\n2️⃣  NEW: Sequential processing (for debugging)")
    print("""
    @guarded_chat_completion(
        app_id="your-app-id", 
        concurrent_guardrail_processing=False
    )
    async def debug_llm_call(messages):
        return await openai_client.chat.completions.create(...)
    """)
    
    # Example 3: Streaming (replaces @streaming_guarded_chat)
    print("\n3️⃣  NEW: Streaming with real-time guardrails")
    print("""
    @guarded_chat_completion(
        app_id="your-app-id",
        streaming_guardrails=True,
        stream_buffer_size=100
    )
    async def stream_llm_call(messages):
        async for chunk in openai_client.chat.completions.create(..., stream=True):
            yield chunk
    """)
    
    # Example 4: All options
    print("\n4️⃣  NEW: All configuration options")
    print("""
    @guarded_chat_completion(
        app_id="your-app-id",
        api_key="your-api-key",  # Optional, uses env var
        base_url="https://api.halioslabs.com",  # Optional
        concurrent_guardrail_processing=True,  # Default: True
        streaming_guardrails=False,  # Default: False
        stream_buffer_size=50,  # Default: 50
        stream_check_interval=0.5,  # Default: 0.5 seconds
        guardrail_timeout=5.0  # Default: 5.0 seconds
    )
    async def fully_configured_llm_call(messages):
        return await openai_client.chat.completions.create(...)
    """)
    
except ImportError as e:
    print(f"❌ guarded_chat_completion not available: {e}")

print("\n" + "="*70)
print("\n📋 MIGRATION SUMMARY:")
print("""
OLD DECORATORS → NEW UNIFIED DECORATOR

@guard(app_id="...", parallel=True)
→ @guarded_chat_completion(app_id="...")

@guard(app_id="...", parallel=False) 
→ @guarded_chat_completion(app_id="...", concurrent_guardrail_processing=False)

@parallel_guarded_chat(app_id="...", parallel=True)
→ @guarded_chat_completion(app_id="...")

@streaming_guarded_chat(app_id="...", stream_buffer_size=100)
→ @guarded_chat_completion(app_id="...", streaming_guardrails=True, stream_buffer_size=100)

BENEFITS:
✅ Single, consistent API
✅ Clear parameter names  
✅ Better discoverability
✅ Fewer imports to remember
✅ Backward compatibility maintained
""")

print("\n🎯 RECOMMENDATIONS:")
print("""
1. Use guarded_chat_completion for all new code
2. Old decorators still work but will show deprecation warnings
3. concurrent_guardrail_processing=True is the default (best performance)
4. Use concurrent_guardrail_processing=False only for debugging
5. Enable streaming_guardrails=True for streaming use cases
""")

async def demo_new_syntax():
    """Quick demo of the new syntax"""
    print("\n🧪 QUICK DEMO:")
    
    # Mock LLM function
    async def mock_llm(messages):
        await asyncio.sleep(0.1)
        return {"choices": [{"message": {"content": "Hello from HaliosAI!"}}]}
    
    # Use the new decorator
    @guarded_chat_completion(
        app_id="demo-app",
        api_key="demo-key", 
        base_url="https://httpbin.org/status/200"  # Mock endpoint
    )
    async def protected_llm_call(messages):
        return await mock_llm(messages)
    
    try:
        messages = [{"role": "user", "content": "Hello!"}]
        result = await protected_llm_call(messages)
        print(f"✅ Demo successful! Result type: {type(result)}")
    except Exception as e:
        print(f"Demo result (expected error with mock endpoint): {type(e).__name__}")

if __name__ == "__main__":
    asyncio.run(demo_new_syntax())
