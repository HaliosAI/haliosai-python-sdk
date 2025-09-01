#!/usr/bin/env python3
"""
Test tool calls functionality
"""
import asyncio
import os
from haliosai import HaliosGuard

async def test_tool_calls():
    """Test sending tool calls in messages"""
    
    # Set up guard with demo configuration
    guard = HaliosGuard(
        agent_id=os.getenv("HALIOS_AGENT_ID", "demo-agent"),
        api_key=os.getenv("HALIOS_API_KEY", "demo-key")
    )
    
    # Test messages with tool_calls
    messages = [
        {
            "role": "user",
            "content": "Just say hello to me"
        }
    ]
    
    # Test messages with tool calls (like what would come from assistant)
    messages_with_tools = [
        {
            "role": "user", 
            "content": "What's the weather like?"
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123", 
            "content": '{"temperature": "72F", "condition": "sunny"}'
        }
    ]
    
    print("🧪 Testing messages without tool calls:")
    print(f"Messages: {messages}")
    
    try:
        result = await guard.evaluate(messages, "request")
        print(f"✅ API call successful")
        print(f"Response: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🧪 Testing messages with tool calls:")
    print(f"Messages: {messages_with_tools}")
    
    try:
        result = await guard.evaluate(messages_with_tools, "response")
        print(f"✅ API call successful")
        print(f"Response: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tool_calls())
