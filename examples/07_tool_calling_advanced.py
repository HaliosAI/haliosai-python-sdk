#!/usr/bin/env python3
"""
HaliosAI SDK - Advanced Tool Calling with Comprehensive Guardrails

This example demonstrates comprehensive guardrail evaluation including:
- Request validation (initial user message)
- Tool call result validation (NEW: prevents sensitive data leakage)
- Response validation (final LLM response)

Uses context manager pattern for fine-grained control over each step.

Key Security Enhancement:
- Tool results are evaluated as "requests" before being added to conversation
- Fail-fast: Any violation blocks the entire operation
- Tool results can be modified by guardrails if needed

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

from haliosai import HaliosGuard, GuardrailViolation

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

async def call_openai_with_tools(messages, tools=None):
    """Make OpenAI API call with tool calling support (no guardrails - handled externally)"""
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

    response = await client.chat.completions.create(**kwargs)
    return response

async def advanced_tool_calling_demo():
    """Advanced tool calling demo with comprehensive guardrail evaluation"""
    print("🔧 HaliosAI Advanced Tool Calling Demo - Comprehensive Guardrails")
    print("=" * 70)

    test_messages = [
        "What's the weather like in San Francisco?",
        "Calculate 15 * 7 + 3",
        "Hello, how are you today?"  # No tools needed
    ]

    async with HaliosGuard(agent_id=HALIOS_AGENT_ID) as guard:
        for i, user_message in enumerate(test_messages, 1):
            print(f"\n{i}️⃣  Testing: {user_message}")
            print("-" * 50)

            # Start conversation
            messages = [{"role": "user", "content": user_message}]

            try:
                # ========================================
                # Step 1: Validate initial request
                # ========================================
                print("🔍 Validating request...")
                await guard.validate_request(messages)
                print("✓ Request passed guardrails")

                # ========================================
                # Step 2: Initial LLM call with tools
                # ========================================
                print("🤖 Getting LLM response with tools...")
                response = await call_openai_with_tools(messages, AVAILABLE_TOOLS)
                message = response.choices[0].message

                # Convert tool_calls to serializable format
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

                # Add assistant's response to conversation
                assistant_message = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls_dict
                }
                messages.append(assistant_message)

                # ========================================
                # Step 3: Check if tools were called
                # ========================================
                if message.tool_calls:
                    print(f"🔧 LLM wants to use {len(message.tool_calls)} tool(s)")

                    # ========================================
                    # Step 4: Execute tools and validate results
                    # ========================================
                    for tool_call in message.tool_calls:
                        print(f"   ▶ Executing: {tool_call.function.name}")

                        # Execute the tool
                        result = execute_tool_call(tool_call)
                        print(f"   ◀ Raw result: {result}")

                        # ========================================
                        # CRITICAL: Validate tool result as "request"
                        # ========================================
                        print("   🔍 Validating tool result...")
                        tool_result_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        }

                        # Evaluate tool result as a "request" to check for sensitive data
                        # This prevents tools from returning sensitive information
                        await guard.validate_request([tool_result_message])
                        print("   ✓ Tool result passed guardrails")

                        # Add validated tool result to conversation
                        messages.append(tool_result_message)

                    # ========================================
                    # Step 5: Get final response with validated tool results
                    # ========================================
                    print("🔄 Getting final response...")
                    final_response = await call_openai_with_tools(messages)
                    final_message = final_response.choices[0].message

                    # ========================================
                    # Step 6: Validate final response
                    # ========================================
                    print("🔍 Validating final response...")
                    final_conversation = messages + [{
                        "role": "assistant",
                        "content": final_message.content
                    }]
                    await guard.validate_response(final_conversation)
                    print("✓ Final response passed guardrails")

                    print(f"✅ Final answer: {final_message.content}")

                else:
                    # ========================================
                    # No tools needed - just validate final response
                    # ========================================
                    print("🔍 Validating direct response...")
                    await guard.validate_response(messages)
                    print("✓ Direct response passed guardrails")
                    print(f"✅ Direct response: {message.content}")

            except GuardrailViolation as e:
                print(f"✗ BLOCKED at {e.violation_type}: {len(e.violations)} violation(s)")
                for violation in e.violations:
                    print(f"   - {violation.get('type', 'unknown')}: {violation.get('message', 'no details')}")
            except Exception as e:
                print(f"❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✨ Advanced tool calling demo completed!")
    print("   ✓ Request validation")
    print("   ✓ Tool result validation")
    print("   ✓ Response validation")
    print("   ✓ Fail-fast behavior")

if __name__ == "__main__":
    asyncio.run(advanced_tool_calling_demo())