#!/usr/bin/env python3
"""
Test tool calls are being sent to API correctly
"""
import asyncio
import os
from haliosai import HaliosGuard

async def test_tool_calls_api_payload():
    """Test that tool calls are being sent in the API payload"""
    
    guard = HaliosGuard(
        agent_id=os.getenv("HALIOS_AGENT_ID", "demo-agent"),
        api_key=os.getenv("HALIOS_API_KEY", "demo-key"), 
        base_url=os.getenv("HALIOS_BASE_URL", "http://localhost:2000")
    )
    
    # Test conversation with tool calls (simulate what LLM would send)
    messages_with_tools = [
        {
            "role": "user",
            "content": "Calculate the result of 15 * 8 + 42"
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function", 
                    "function": {
                        "name": "calculate_math",
                        "arguments": '{"expression":"15 * 8 + 42"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"result": 162}'
        }
    ]
    
    print("🧪 Testing API payload with tool calls...")
    print(f"Messages being sent to API:")
    for i, msg in enumerate(messages_with_tools):
        print(f"   {i+1}. Role: {msg['role']}")
        print(f"      Content: {msg.get('content')}")
        if msg.get('tool_calls'):
            print(f"      Tool calls: {len(msg['tool_calls'])} calls")
            for tc in msg['tool_calls']:
                print(f"        • {tc['function']['name']}({tc['function']['arguments']})")
        if msg.get('tool_call_id'):
            print(f"      Tool call ID: {msg['tool_call_id']}")
    
    try:
        result = await guard.evaluate(messages_with_tools, "response")
        print(f"\n✅ API call successful!")
        print(f"Messages processed: {result.get('request', {}).get('message_count', 'unknown')}")
        print(f"Content length: {result.get('request', {}).get('content_length', 'unknown')} chars")
        print(f"Guardrails triggered: {result.get('guardrails_triggered', 0)}")
        
        # The key thing is that the API accepted and processed the tool calls
        if result.get('request', {}).get('message_count', 0) == 3:
            print("🔧 ✅ Tool calls were properly processed by the API!")
        else:
            print("❌ Unexpected message count - tool calls may not have been processed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tool_calls_api_payload())
