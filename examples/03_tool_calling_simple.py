#!/usr/bin/env python3
"""
HaliosAI SDK - Example 3: Simple Tool Calling with Guardrails

This example demonstrates how to use HaliosAI guardrails with Gemini function/tool calling.
Based on the auto_demo_tools.py structure but using the HaliosAI SDK.

Requirements:
    pip install haliosai openai

Environment Variables:
    HALIOS_API_KEY: Your HaliosAI API key  
    HALIOS_BASE_URL: Your HaliosAI base URL (optional)
    HALIOS_AGENT_ID: Your HaliosAI agent ID
    GEMINI_API_KEY: Your Gemini API key
"""

import asyncio
import json
import os
import time
from openai import AsyncOpenAI

from haliosai import guarded_chat_completion

# Configuration validation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID")

if not GEMINI_API_KEY or not HALIOS_AGENT_ID:
    print("❌ Missing required environment variables:")
    print("   - GEMINI_API_KEY")
    print("   - HALIOS_AGENT_ID")
    exit(1)

# Define simple tools (same as auto_demo_tools.py)
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information (simulated)",
            "parameters": {
                "type": "object", 
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def execute_tool_call(tool_call):
    """Simulate tool execution (from auto_demo_tools.py)"""
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
    elif function_name == "search_web":
        query = arguments.get("query", "")
        return {
            "query": query,
            "results": [
                f"Search result 1 for '{query}'",
                f"Search result 2 for '{query}'",
                f"Search result 3 for '{query}'"
            ]
        }
    else:
        return {"error": f"Unknown function: {function_name}"}

# Guarded chat completion with tools
@guarded_chat_completion(
    agent_id=HALIOS_AGENT_ID
)
async def call_gemini_with_tools(messages, tools=None):
    """
    Make Gemini API call with tool calling support and automatic guardrails
    (Based on auto_demo_tools.py structure)
    """
    client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    kwargs = {
        "model": "gemini-2.0-flash",
        "messages": messages,
        "max_tokens": 300
    }
    
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    
    response = await client.chat.completions.create(**kwargs)
    return response

async def chat_with_tools(user_message):
    """
    Perform a chat completion with tool calling and guardrails
    (Simplified version of auto_demo_tools.py)
    """
    print(f"\n🤖 Processing: {user_message}")
    print("="*60)
    
    messages = [{"role": "user", "content": user_message}]
    
    try:
        start_time = time.time()
        
        # First call - may include tool calls
        print("📞 Making initial API call with tools...")
        response = await call_gemini_with_tools(messages, AVAILABLE_TOOLS)
        
        # Handle GuardedResponse from the decorator
        from haliosai import ExecutionResult
        
        if hasattr(response, 'result'):
            if response.result == ExecutionResult.SUCCESS:
                # Extract the actual OpenAI response
                actual_response = response.final_response
                if isinstance(actual_response, str):
                    print(f"✅ Response: {actual_response}")
                    return
                else:
                    # Handle dict or OpenAI response object
                    if hasattr(actual_response, 'choices'):
                        choice = actual_response.choices[0]
                    elif isinstance(actual_response, dict) and 'choices' in actual_response:
                        choice = type('Choice', (), actual_response['choices'][0])()
                        choice.message = type('Message', (), actual_response['choices'][0]['message'])()
                        choice.message.content = actual_response['choices'][0]['message'].get('content')
                        choice.message.tool_calls = actual_response['choices'][0]['message'].get('tool_calls')
                    else:
                        print(f"✅ Response: {str(actual_response)}")
                        return
            elif response.result == ExecutionResult.REQUEST_BLOCKED:
                print(f"🚫 Request blocked by guardrails: {response.request_violations}")
                return
            elif response.result == ExecutionResult.RESPONSE_BLOCKED:
                print(f"🚫 Response blocked by guardrails: {response.response_violations}")
                return
            else:
                print(f"❌ Error: {response.error_message}")
                return
        else:
            # Direct response without guardrails
            choice = response.choices[0]
        
        assistant_message = choice.message
        
        # Add assistant response to conversation
        # Fix missing tool call IDs (Gemini compatibility issue)
        tool_calls_for_message = None
        if assistant_message.tool_calls:
            tool_calls_for_message = []
            for tc in assistant_message.tool_calls:
                # Generate ID if missing (Gemini compatibility fix)
                tool_call_id = tc.id if tc.id else f"call_{hash(tc.function.name + tc.function.arguments) & 0x7FFFFFFF:08x}"
                
                # Use model_dump instead of deprecated dict()
                if hasattr(tc, 'model_dump'):
                    tool_call_dict = tc.model_dump()
                elif hasattr(tc, 'dict'):
                    tool_call_dict = tc.dict()
                else:
                    # Fallback for basic objects
                    tool_call_dict = {
                        "id": tool_call_id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                
                # Ensure ID is set
                tool_call_dict["id"] = tool_call_id
                tool_calls_for_message.append(tool_call_dict)
        
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": tool_calls_for_message
        })
        
        # Check if model wants to use tools
        if assistant_message.tool_calls:
            print(f"\n🔧 Model requested {len(assistant_message.tool_calls)} tool call(s):")
            
            for i, tool_call in enumerate(assistant_message.tool_calls):
                print(f"   • {tool_call.function.name}({tool_call.function.arguments})")
                
                # Execute the tool call
                tool_result = execute_tool_call(tool_call)
                print(f"   → Result: {json.dumps(tool_result, indent=2)}")
                
                # Use the same ID we generated for the assistant message
                tool_call_id = tool_calls_for_message[i]["id"] if tool_calls_for_message else (
                    tool_call.id if tool_call.id else f"call_{hash(tool_call.function.name + tool_call.function.arguments) & 0x7FFFFFFF:08x}"
                )
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result)
                })
            
            # Make follow-up call with tool results
            print("\n📞 Making follow-up call with tool results...")
            final_response = await call_gemini_with_tools(messages)
            
            # Handle final response
            if hasattr(final_response, 'result') and final_response.result == ExecutionResult.SUCCESS:
                if isinstance(final_response.final_response, str):
                    final_content = final_response.final_response
                else:
                    final_content = final_response.final_response.choices[0].message.content
            else:
                final_content = final_response.choices[0].message.content
                
            print(f"\n✅ Final Response:\n{final_content}")
        else:
            # No tools needed
            print(f"\n✅ Response (no tools needed):\n{assistant_message.content}")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total conversation time: {total_time:.2f}s")
        
    except Exception as e:
        if "blocked by guardrails" in str(e):
            print(f"🚫 Blocked by guardrails: {e}")
        else:
            print(f"💥 Error: {e}")

async def main():
    """Run the tool calling demo"""
    print("🛡️  HaliosAI SDK - Tool Calling Demo")
    print("="*60)
    
    print("🔧 Available tools:")
    for tool in AVAILABLE_TOOLS:
        func = tool["function"]
        print(f"   • {func['name']}: {func['description']}")
    
    # Test cases that should trigger tool usage
    test_cases = [
        "What's the weather like in San Francisco?",
        "Calculate the result of 15 * 8 + 42",
        "Search for information about quantum computing",
        "Just say hello to me",  # No tools needed
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}/{len(test_cases)}")
        await chat_with_tools(test_case)
        
        if i < len(test_cases):
            await asyncio.sleep(1)  # Brief pause between tests
    
    print(f"\n{'='*60}")
    print("✅ Tool calling demo completed!")
    print("\n📋 This demo tested:")
    print("   • Request guardrails with tool definitions")
    print("   • Response guardrails with tool calls")
    print("   • Multi-turn conversations with tool results")
    print("   • Both tool-using and non-tool conversations")

if __name__ == "__main__":
    asyncio.run(main())
