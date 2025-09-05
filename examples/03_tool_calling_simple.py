#!/usr/bin/env python3
"""
HaliosAI SDK - Example 3: Simple Tool Calling with Guardrails

This example demonstrates how to use HaliosAI guardrails with OpenAI function/tool calling.
Shows the minimal post-processing required for tool calling workflows.

Requirements:
    pip install haliosai openai

Environment Variables:
    HALIOS_API_KEY: Your HaliosAI API key  
    HALIOS_BASE_URL: Your HaliosAI base URL (optional)
    HALIOS_AGENT_ID: Your HaliosAI agent ID
    OPENAI_API_KEY: Your OpenAI API key
"""

import asyncio
import json
import os
from openai import AsyncOpenAI

from haliosai import guarded_chat_completion

# Configuration validation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID")

if not OPENAI_API_KEY or not HALIOS_AGENT_ID:
    print("❌ Missing required environment variables:")
    print("   - OPENAI_API_KEY")
    print("   - HALIOS_AGENT_ID")
    exit(1)

# Define simple tools
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country, e.g. 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "calculate_math",
            "description": "Perform basic mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

def execute_tool_call(tool_call):
    """Simulate tool execution"""
    function_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return {"error": "Invalid arguments JSON"}
    
    # Simulate tool responses
    if function_name == "get_weather":
        location = arguments.get("location", "Unknown")
        unit = arguments.get("unit", "fahrenheit")
        return {
            "location": location,
            "temperature": "72°F" if unit == "fahrenheit" else "22°C",
            "condition": "Sunny",
            "humidity": "45%"
        }
    elif function_name == "calculate_math":
        expression = arguments.get("expression", "")
        try:
            # Simple evaluation (in real code, use safer approach)
            result = eval(expression.replace("^", "**"))
            return {"expression": expression, "result": result}
        except:
            return {"error": "Invalid mathematical expression"}
    else:
        return {"error": f"Unknown function: {function_name}"}

# Guarded chat completion with tools
@guarded_chat_completion(agent_id=HALIOS_AGENT_ID)
async def call_openai_with_tools(messages, tools=None):
    """Make OpenAI API call with tool calling support and automatic guardrails"""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.1
    }
    
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    
    # The decorator handles guardrails automatically
    response = await client.chat.completions.create(**kwargs)
    return response

async def simple_tool_calling_demo():
    """Simplified tool calling demo (based on working debug_tools.py pattern)"""
    print("🔧 HaliosAI Tool Calling Demo - Simplified Version")
    print("=" * 60)
    
    test_messages = [
        "What's the weather like in San Francisco?",
        "Calculate 15 * 7 + 3",
        "Hello, how are you today?"  # No tools needed
    ]
    
    for i, user_message in enumerate(test_messages, 1):
        print(f"\n{i}️⃣  Testing: {user_message}")
        print("-" * 40)
        
        # Start conversation
        messages = [{"role": "user", "content": user_message}]
        
        try:
            # Step 1: Initial call with tools
            response = await call_openai_with_tools(messages, AVAILABLE_TOOLS)
            message = response.choices[0].message
            
            # Add assistant's response to conversation
            # Convert tool_calls to serializable format for guardrails
            tool_calls_dict = None
            if message.tool_calls:
                tool_calls_dict = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            
            messages.append({
                "role": "assistant", 
                "content": message.content,
                "tool_calls": tool_calls_dict
            })
            
            # Step 2: Check if tools were called
            if message.tool_calls:
                print(f"🔧 OpenAI wants to use {len(message.tool_calls)} tool(s)")
                
                # Execute tools and add results
                for tool_call in message.tool_calls:
                    print(f"   ▶ Calling: {tool_call.function.name}")
                    
                    result = execute_tool_call(tool_call)
                    print(f"   ◀ Result: {result}")
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                # Step 3: Final call for natural language response
                print("🔄 Getting final response...")
                final_response = await call_openai_with_tools(messages)
                final_content = final_response.choices[0].message.content
                print(f"✅ Final answer: {final_content}")
                
            else:
                # No tools needed
                print(f"✅ Direct response: {message.content}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✨ Tool calling demo completed!")

if __name__ == "__main__":
    asyncio.run(simple_tool_calling_demo())
