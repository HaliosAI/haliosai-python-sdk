#!/usr/bin/env python3
"""
HaliosAI SDK - Interactive Chatbot

Demonstrates:
- Real interactive chat session
- Natural conversation flow with guardrails
- Guardrail policies (RECORD_ONLY vs BLOCK)
- Works with ANY agent configuration

Setup Required:
1. Create agent in HaliosAI dashboard for YOUR use case
2. Configure guardrails appropriate for your agent's persona
3. Set environment variables:
   export HALIOS_API_KEY="your-key"
   export HALIOS_AGENT_ID="your-agent-id"
   export OPENAI_API_KEY="your-openai-key"

Usage:
    python examples/06_interactive_chatbot.py
    
Then chat naturally with messages relevant to YOUR agent's domain!
Type 'quit' or 'exit' to end the session.

💡 This is the BEST way to explore guardrails with your specific agent.
"""

import asyncio
import os
from openai import AsyncOpenAI, OpenAIError
from haliosai import guarded_chat_completion, GuardrailViolation, GuardrailPolicy

# Validate required environment variables
REQUIRED_VARS = ["HALIOS_API_KEY", "HALIOS_AGENT_ID", "OPENAI_API_KEY"]
missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID")

@guarded_chat_completion(
    agent_id=HALIOS_AGENT_ID,
    guardrail_policies={
        "sensitive-data": GuardrailPolicy.RECORD_ONLY,  # Log but allow
        "hate-speech": GuardrailPolicy.BLOCK,           # Block hate speech
        "toxicity": GuardrailPolicy.BLOCK               # Block toxic content
    }
)
async def chat_with_ai(messages):
    client = AsyncOpenAI(timeout=30.0)
    response = await client.chat.completions.create(
        model='gpt-4',
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content

async def chatbot():
    """Simple chatbot with guardrails"""
    print("🤖 HaliosAI Interactive Chatbot")
    print("Chat naturally - guardrails protect your conversation")
    print("Type 'quit' or 'exit' to end session\n")

    # 👇 CUSTOMIZE THIS SYSTEM PROMPT FOR YOUR AGENT'S PERSONA
    system_prompt = """You are a helpful assistant. Provide clear, concise responses."""

    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_input})

            # Get AI response with guardrails
            try:
                ai_response = await chat_with_ai(conversation_history)
                print(f"🤖 Assistant: {ai_response}")

                # Add AI response to conversation history
                conversation_history.append({"role": "assistant", "content": ai_response})

            except GuardrailViolation as e:
                print(f"🚫 Content blocked: {e}")
                # Don't add blocked content to conversation history

            except (OpenAIError, ValueError) as e:
                print(f"❌ API Error: {e}")
                # Remove the last user message on error
                conversation_history.pop()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

async def main():
    """Run the chatbot"""
    await chatbot()

if __name__ == "__main__":
    asyncio.run(main())
