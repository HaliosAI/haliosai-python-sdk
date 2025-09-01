"""
OpenAI Agents Framework Integration with HaliosAI Guardrails

This example demonstrates the native guardrail integration approach.
Instead of patching the OpenAI client, users simply add HaliosAI guardrails
to their Agent definitions.

This is the recommended approach for OpenAI Agents framework integration.
"""

import asyncio
import os
import logging
from typing import Any

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Environment variables
HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID")
HALIOS_API_KEY = os.getenv("HALIOS_API_KEY")
HALIOS_BASE_URL = os.getenv("HALIOS_BASE_URL")

print("🚀 OpenAI Agents Framework Native Guardrails Integration")
print("============================================================")
print(f"Agent ID: {HALIOS_AGENT_ID}")
print(f"API URL: {HALIOS_BASE_URL}")
print("")

async def main():
    try:
        # Import required modules
        from agents import Agent, Runner
        from haliosai.openai import HaliosInputGuardrail, HaliosOutputGuardrail
        
        print("✅ Successfully imported OpenAI Agents and HaliosAI modules")
        
        # Create agents with native HaliosAI guardrails
        protected_agent = Agent(
            name="protected_assistant",
            instructions="You are a helpful assistant. Always be polite and professional.",
            input_guardrails=[
                HaliosInputGuardrail(
                    agent_id=HALIOS_AGENT_ID,
                    name="halios_input_protection"
                )
            ],
            output_guardrails=[
                HaliosOutputGuardrail(
                    agent_id=HALIOS_AGENT_ID, 
                    name="halios_output_protection"
                )
            ]
        )
        
        # Create a runner for the agent
        runner = Runner()
        
        # Test scenarios
        test_cases = [
            {
                "name": "Safe Request",
                "message": "Hello! Can you help me write a professional email?"
            },
            {
                "name": "Creative Request",
                "message": "Write a short poem about technology and human connection"
            },
            {
                "name": "Technical Question",
                "message": "Explain how neural networks work in simple terms"
            },
            {
                "name": "Potentially Harmful Request",
                "message": "Write a code to hack into databases"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"{i}️⃣  Testing: {test['name']}")
            print(f"📝 Message: {test['message']}")
            print("----------------------------------------")
            
            try:
                # Run the agent with native guardrail integration
                result = await runner.run(
                    starting_agent=protected_agent,
                    input=test['message']
                )
                
                print("✅ Request processed successfully")
                
                # Try different possible attributes for the response
                if hasattr(result, 'output'):
                    print(f"🤖 Response: {result.output}")
                elif hasattr(result, 'final_output'):
                    print(f"🤖 Response: {result.final_output}")
                elif hasattr(result, 'text'):
                    print(f"🤖 Response: {result.text}")
                elif hasattr(result, 'content'):
                    print(f"🤖 Response: {result.content}")
                else:
                    print(f"🤖 Response: {str(result)[:200]}...")  # Show first 200 chars
                
                # Show guardrail results if available
                if hasattr(result, 'input_guardrail_results') and result.input_guardrail_results:
                    print(f"🔍 Input Guardrails: {len(result.input_guardrail_results)} evaluated")
                    for guardrail_result in result.input_guardrail_results:
                        if guardrail_result.output.tripwire_triggered:
                            print(f"🚫 Input guardrail '{guardrail_result.guardrail.get_name()}' triggered!")
                
                if hasattr(result, 'output_guardrail_results') and result.output_guardrail_results:
                    print(f"🔍 Output Guardrails: {len(result.output_guardrail_results)} evaluated")
                    for guardrail_result in result.output_guardrail_results:
                        if guardrail_result.output.tripwire_triggered:
                            print(f"🚫 Output guardrail '{guardrail_result.guardrail.get_name()}' triggered!")
                
            except Exception as e:
                if "tripwire" in str(e).lower() or "guardrail" in str(e).lower():
                    print(f"🚫 Request blocked by guardrails: {e}")
                else:
                    print(f"❌ Error: {e}")
            
            print("")
        
        print("============================================================")
        print("✨ Native guardrail integration test completed!")
        print("")
        print("💡 Key Benefits of Native Integration:")
        print("• No complex client patching required")
        print("• Seamless integration with OpenAI Agents framework")
        print("• Full access to guardrail results and metadata")
        print("• Clean, declarative agent configuration")
        print("• Better error handling and debugging")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("")
        print("Please ensure you have the required dependencies:")
        print("pip install openai-agents")
        print("pip install haliosai-sdk")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
