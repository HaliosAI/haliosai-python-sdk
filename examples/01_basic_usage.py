#!/usr/bin/env python3
"""
HaliosAI SDK - Example 1: Basic Usage

This is the simplest possible example showing how to protect an LLM call 
with guardrails using the unified decorator syntax.

Requirements:
    pip install haliosai
    pip install openai  # For this example

Environment Variables:
    HALIOS_API_KEY: Your HaliosAI API key  
    HALIOS_AGENT_ID: Your agent ID (or pass as parameter)
    GEMINI_API_KEY: Your Gemini API key (optional, for real testing)
"""

import asyncio
import os
from openai import AsyncOpenAI
from haliosai import guarded_chat_completion

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "demo-key")
HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID", "demo-agent-basic")

# Simple protected LLM call using the unified decorator
@guarded_chat_completion(agent_id=HALIOS_AGENT_ID)
async def protected_gemini_call(messages):
    """
    Make a Gemini API call protected by HaliosAI guardrails
    
    The decorator automatically handles:
    - REQUEST guardrails: Evaluated before sending to Gemini
    - RESPONSE guardrails: Evaluated after receiving response
    - Concurrent processing: Guardrails run in parallel with LLM call
    
    Args:
        messages: List of {"role": "...", "content": "..."} messages
        
    Returns:
        Gemini response object (compatible with OpenAI format)
    """
    client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    response = await client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    
    return response

async def main():
    """Run basic example with different types of messages"""
    print("🚀 HaliosAI Basic Example - Protected LLM Calls")
    print("=" * 60)
    
    # Test cases with different content types
    test_cases = [
        {
            "name": "Safe Message",
            "messages": [
                {"role": "user", "content": "Hello! Can you help me write a professional email?"}
            ]
        },
        {
            "name": "Creative Request", 
            "messages": [
                {"role": "user", "content": "Write a short story about a robot learning to paint"}
            ]
        },
        {
            "name": "Technical Question",
            "messages": [
                {"role": "user", "content": "Explain how machine learning models work in simple terms"}
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣  Testing: {test_case['name']}")
        print(f"📝 Message: {test_case['messages'][0]['content']}")
        print("-" * 40)
        
        try:
            # Call the protected function
            response = await protected_gemini_call(test_case['messages'])
            
            # Handle the response based on its type
            if hasattr(response, 'result'):
                # GuardedResponse object
                from haliosai import ExecutionResult
                
                if response.result == ExecutionResult.SUCCESS:
                    print("✅ SUCCESS: Response received and passed guardrails")
                    if response.final_response:
                        # Extract content from the response
                        if hasattr(response.final_response, 'choices'):
                            # OpenAI-style response object
                            content = response.final_response.choices[0].message.content
                        elif isinstance(response.final_response, str):
                            # String response
                            content = response.final_response
                        elif isinstance(response.final_response, dict):
                            # Dict response - try to extract content
                            if 'choices' in response.final_response:
                                content = response.final_response['choices'][0]['message']['content']
                            else:
                                content = str(response.final_response)
                        else:
                            content = str(response.final_response)
                        
                        print(f"🤖 Response: {content[:100]}{'...' if len(content) > 100 else ''}")
                    else:
                        print("🤖 Response: (empty or null response)")
                elif response.result == ExecutionResult.REQUEST_BLOCKED:
                    print("🚫 REQUEST BLOCKED: Message violates input guardrails")
                    print(f"⚠️  Violations: {response.request_violations}")
                elif response.result == ExecutionResult.RESPONSE_BLOCKED:
                    print("🚫 RESPONSE BLOCKED: LLM response violates output guardrails")
                    print(f"⚠️  Violations: {response.response_violations}")
                elif response.result == ExecutionResult.ERROR:
                    print(f"❌ ERROR: {response.error_message}")
                
                # Show timing information
                if response.timing:
                    print(f"⏱️  Total time: {response.timing['total_time']:.3f}s")
            else:
                # Direct response (fallback mode)
                print("✅ SUCCESS: Direct response received")
                if hasattr(response, 'choices'):
                    content = response.choices[0].message.content
                    print(f"🤖 Response: {content[:100]}{'...' if len(content) > 100 else ''}")
                else:
                    print(f"🤖 Response: {str(response)[:100]}...")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print()
    
    print("=" * 60)
    print("✨ Basic example completed!")
    print("\n💡 Key Features Demonstrated:")
    print("• Simple decorator syntax: @guarded_chat_completion(agent_id=...)")
    print("• Automatic request and response guardrail evaluation")
    print("• Concurrent processing (guardrails run parallel to LLM)")
    print("• Detailed violation reporting and timing metrics")
    print("• Graceful error handling")

if __name__ == "__main__":
    # Set up demo environment if real credentials not provided
    if not os.getenv("HALIOS_API_KEY"):
        os.environ["HALIOS_API_KEY"] = "demo-key"
        os.environ["HALIOS_BASE_URL"] = "https://httpbin.org/status/200"
        print("🧪 Demo mode: Using mock endpoints (errors expected)")
        print("   Set HALIOS_API_KEY environment variable for real usage\n")
    
    asyncio.run(main())
