#!/usr/bin/env python3
"""
Test script for the unified guarded_chat_completion decorator
"""

import asyncio
import logging
from typing import List, Dict, Any

# Test the new unified decorator
try:
    from haliosai import guarded_chat_completion
    print("✅ Successfully imported guarded_chat_completion")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Mock LLM function for testing
async def mock_llm_call(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mock LLM function that returns a simple response"""
    await asyncio.sleep(0.1)  # Simulate API call
    return {
        "choices": [{
            "message": {
                "role": "assistant", 
                "content": "Hello! This is a test response."
            }
        }]
    }

# Mock streaming LLM function
async def mock_streaming_llm_call(messages: List[Dict[str, Any]]):
    """Mock streaming LLM function"""
    chunks = ["Hello", " ", "world", "! ", "This", " is", " streaming", "."]
    for chunk in chunks:
        await asyncio.sleep(0.05)
        yield {
            "choices": [{
                "delta": {"content": chunk}
            }]
        }

async def test_basic_decorator():
    """Test basic usage with concurrent processing"""
    print("\n🧪 Testing basic decorator (concurrent processing)...")
    
    @guarded_chat_completion(
        app_id="test-app-id",
        api_key="test-key",
        base_url="https://httpbin.org/status/200"  # Mock endpoint that returns 200
    )
    async def call_llm(messages):
        return await mock_llm_call(messages)
    
    messages = [{"role": "user", "content": "Hello test!"}]
    
    try:
        result = await call_llm(messages)
        print(f"✅ Basic decorator test passed: {result}")
    except Exception as e:
        print(f"❌ Basic decorator test failed: {e}")

async def test_sequential_processing():
    """Test sequential processing (for debugging)"""
    print("\n🧪 Testing sequential processing...")
    
    @guarded_chat_completion(
        app_id="test-app-id",
        api_key="test-key", 
        base_url="https://httpbin.org/status/200",
        concurrent_guardrail_processing=False
    )
    async def call_llm_sequential(messages):
        return await mock_llm_call(messages)
    
    messages = [{"role": "user", "content": "Hello sequential test!"}]
    
    try:
        result = await call_llm_sequential(messages)
        print(f"✅ Sequential processing test passed: {result}")
    except Exception as e:
        print(f"❌ Sequential processing test failed: {e}")

async def test_streaming():
    """Test streaming with guardrails"""
    print("\n🧪 Testing streaming with guardrails...")
    
    @guarded_chat_completion(
        app_id="test-app-id",
        api_key="test-key",
        base_url="https://httpbin.org/status/200",
        streaming_guardrails=True,
        stream_buffer_size=10
    )
    async def stream_llm(messages):
        async for chunk in mock_streaming_llm_call(messages):
            yield chunk
    
    messages = [{"role": "user", "content": "Hello streaming test!"}]
    
    try:
        print("📡 Streaming events:")
        async for event in stream_llm(messages):
            if event.get('type') == 'chunk':
                print(f"  📝 Chunk: {event.get('content', '')}", end='')
            elif event.get('type') == 'completed':
                print(f"\n  ✅ Stream completed!")
            else:
                print(f"  📊 Event: {event}")
        print("✅ Streaming test passed")
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")

async def main():
    """Run all tests"""
    print("🚀 Testing HaliosAI unified decorator...")
    
    await test_basic_decorator()
    await test_sequential_processing()
    await test_streaming()
    
    print("\n✨ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
