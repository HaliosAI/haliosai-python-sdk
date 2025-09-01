#!/usr/bin/env python3
"""
Test tool calls preservation with guardrails
"""
import asyncio
import os
from haliosai import guarded_chat_completion

# Mock LLM response that includes tool calls
class MockResponse:
    def __init__(self):
        self.choices = [MockChoice()]

class MockChoice:
    def __init__(self):
        self.message = MockMessage()

class MockMessage:
    def __init__(self):
        self.content = None
        self.tool_calls = [MockToolCall()]

class MockToolCall:
    def __init__(self):
        self.id = "call_123"
        self.type = "function"
        self.function = MockFunction()

class MockFunction:
    def __init__(self):
        self.name = "calculate_math"
        self.arguments = '{"expression":"15 * 8 + 42"}'

@guarded_chat_completion(
    agent_id=os.getenv("HALIOS_AGENT_ID", "demo-agent"),
    api_key=os.getenv("HALIOS_API_KEY", "demo-key"),
    base_url=os.getenv("HALIOS_BASE_URL", "http://localhost:2000")
)
async def mock_llm_with_tools(messages):
    """Mock LLM that returns tool calls"""
    print(f"📧 LLM received messages: {len(messages)}")
    for i, msg in enumerate(messages):
        print(f"   {i+1}. {msg.get('role', 'unknown')}: {str(msg.get('content', 'None'))[:50]}...")
        if msg.get('tool_calls'):
            print(f"      🔧 Tool calls: {len(msg['tool_calls'])} calls")
    
    # Return a mock response with tool calls
    return MockResponse()

async def test_tool_calls_preservation():
    """Test that tool calls are preserved through guardrails"""
    
    messages = [{"role": "user", "content": "Calculate 15 * 8 + 42"}]
    
    print("🧪 Testing tool calls preservation...")
    print(f"Input messages: {messages}")
    
    try:
        result = await mock_llm_with_tools(messages)
        print(f"✅ Function completed successfully")
        print(f"Result type: {type(result)}")
        
        # Check if it has tool calls
        if hasattr(result, 'choices') and result.choices:
            message = result.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"🔧 Tool calls preserved: {len(message.tool_calls)} calls")
                for tc in message.tool_calls:
                    print(f"   • {tc.function.name}({tc.function.arguments})")
            else:
                print("❌ No tool calls found in result")
        else:
            print("❌ No choices in result")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tool_calls_preservation())
