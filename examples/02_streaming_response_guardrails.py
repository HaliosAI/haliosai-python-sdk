#!/usr/bin/env python3
"""
HaliosAI SDK - Example 2: Streaming with Real-time Guardrails

This example demonstrates streaming responses with real-time guardrail evaluation.
The decorator buffers streaming content and evaluates it incrementally.

Requirements:
    pip install haliosai
    pip install openai  # For this example

Environment Variables:
    HALIOS_API_KEY: Your HaliosAI API key  
    HALIOS_APP_ID: Your application ID
    OPENAI_API_KEY: Your OpenAI API key (for real testing)
"""

import asyncio
import os
from openai import AsyncOpenAI
from haliosai import guarded_chat_completion

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "demo-key")
HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID", "demo-agent-streaming")

# Streaming LLM call with real-time guardrails
@guarded_chat_completion(
    agent_id=HALIOS_AGENT_ID,
    streaming_guardrails=True,
    stream_buffer_size=50,  # Check guardrails every 50 characters
    stream_check_interval=0.5,  # Check every 500ms
    guardrail_timeout=3.0  # 3 second timeout for guardrail checks
)
async def streaming_openai_call(messages):
    """
    Make a streaming OpenAI call with real-time guardrail monitoring
    
    The decorator automatically:
    - Evaluates request guardrails before starting the stream
    - Buffers streaming content and checks guardrails incrementally
    - Can halt streaming if violations are detected
    - Provides real-time feedback events
    
    Args:
        messages: List of message objects
        
    Yields:
        OpenAI streaming response chunks
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=800,
        temperature=0.7,
        stream=True
    )
    
    async for chunk in stream:
        yield chunk

async def demo_streaming():
    """Demonstrate streaming with different scenarios"""
    print("🌊 HaliosAI Streaming Example - Real-time Guardrails")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Creative Writing",
            "messages": [
                {"role": "user", "content": "Write a creative story about a time traveler who discovers an ancient library. Make it engaging and descriptive."}
            ]
        },
        {
            "name": "Technical Explanation", 
            "messages": [
                {"role": "user", "content": "Explain how blockchain technology works, including concepts like decentralization, consensus mechanisms, and smart contracts."}
            ]
        },
        {
            "name": "Code Generation",
            "messages": [
                {"role": "user", "content": "Write a Python function that implements a binary search algorithm with detailed comments."}
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣  Streaming Test: {test_case['name']}")
        print(f"📝 Query: {test_case['messages'][0]['content'][:80]}...")
        print("🌊 Streaming response:")
        print("-" * 40)
        
        try:
            content_buffer = ""
            chunk_count = 0
            
            async for event in streaming_openai_call(test_case['messages']):
                chunk_count += 1
                
                # Handle different event types from the guardrail decorator
                if isinstance(event, dict):
                    event_type = event.get('type', 'unknown')
                    
                    if event_type == 'chunk':
                        # Content chunk - display it
                        chunk_content = event.get('content', '')
                        if chunk_content:
                            print(chunk_content, end='', flush=True)
                            content_buffer += chunk_content
                    
                    elif event_type == 'guardrail_check':
                        # Guardrail evaluation happened
                        print(f"\n🛡️  [Guardrail check at {len(content_buffer)} chars]", end='')
                    
                    elif event_type == 'warning':
                        # Guardrail warning
                        print(f"\n⚠️  Warning: {event.get('message', 'Unknown warning')}")
                    
                    elif event_type == 'violation':
                        # Guardrail violation - streaming stopped
                        print(f"\n🚫 VIOLATION DETECTED: {event.get('violation_type', 'Unknown')}")
                        print(f"📋 Details: {event.get('details', 'No details')}")
                        break
                    
                    elif event_type == 'completed':
                        # Stream completed successfully
                        print(f"\n✅ Stream completed successfully!")
                        timing = event.get('timing', {})
                        if timing:
                            print(f"⏱️  Total time: {timing.get('total_time', 0):.3f}s")
                            print(f"📊 Chunks processed: {chunk_count}")
                    
                    elif event_type == 'error':
                        # Error occurred
                        print(f"\n❌ Error: {event.get('error', 'Unknown error')}")
                        break
                
                else:
                    # Handle raw OpenAI chunks (fallback mode)
                    if hasattr(event, 'choices') and event.choices:
                        delta = event.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            print(delta.content, end='', flush=True)
                            content_buffer += delta.content
                            chunk_count += 1
            
            print(f"\n📈 Final stats: {len(content_buffer)} characters, {chunk_count} chunks")
            
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
        
        print("\n" + "=" * 40)
        
        # Brief pause between tests
        await asyncio.sleep(1)
    
    print("\n✨ Streaming example completed!")
    print("\n💡 Key Features Demonstrated:")
    print("• Real-time streaming with guardrail monitoring")
    print("• Configurable buffer size and check intervals")
    print("• Incremental content evaluation")
    print("• Violation detection and stream interruption")
    print("• Detailed event handling and user feedback")

async def demo_configuration_options():
    """Show different streaming configuration options"""
    print("\n⚙️  Streaming Configuration Examples")
    print("=" * 60)
    
    # Example 1: Fast checking (small buffer, frequent checks)
    @guarded_chat_completion(
        agent_id=HALIOS_AGENT_ID,
        streaming_guardrails=True,
        stream_buffer_size=20,  # Check every 20 characters
        stream_check_interval=0.2  # Check every 200ms
    )
    async def fast_checking_stream(messages):
        # Mock streaming for demo
        chunks = ["This ", "is ", "a ", "fast ", "checking ", "stream ", "example."]
        for chunk in chunks:
            await asyncio.sleep(0.1)
            yield {"choices": [{"delta": {"content": chunk}}]}
    
    # Example 2: Balanced checking
    @guarded_chat_completion(
        agent_id=HALIOS_AGENT_ID,
        streaming_guardrails=True,
        stream_buffer_size=100,  # Check every 100 characters  
        stream_check_interval=1.0  # Check every 1 second
    )
    async def balanced_stream(messages):
        # Mock streaming for demo
        chunks = ["This is a balanced streaming example with moderate checking frequency. ",
                 "It provides good performance while maintaining safety. ",
                 "The buffer size and interval are optimized for typical use cases."]
        for chunk in chunks:
            await asyncio.sleep(0.3)
            yield {"choices": [{"delta": {"content": chunk}}]}
    
    print("1️⃣  Fast Checking Configuration:")
    print("   • stream_buffer_size=20, stream_check_interval=0.2")
    print("   • Use for: High-risk content, maximum safety")
    
    print("\n2️⃣  Balanced Configuration:")
    print("   • stream_buffer_size=100, stream_check_interval=1.0") 
    print("   • Use for: General purpose, good performance/safety balance")

async def main():
    """Run all streaming examples"""
    await demo_streaming()
    await demo_configuration_options()

if __name__ == "__main__":
    # Set up demo environment if real credentials not provided
    if not os.getenv("HALIOS_API_KEY"):
        os.environ["HALIOS_API_KEY"] = "demo-key"
        os.environ["HALIOS_BASE_URL"] = "https://httpbin.org/status/200"
        print("🧪 Demo mode: Using mock endpoints (guardrail errors expected)")
        print("   Set HALIOS_API_KEY and OPENAI_API_KEY for real usage\n")
    
    asyncio.run(main())
