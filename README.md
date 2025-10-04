# HaliosAI SDK

[![PyPI version](https://img.shields.io/pypi/v/haliosai.svg)](https://pypi.org/project/haliosai/)
[![Python Support](https://img.shields.io/pypi/pyversions/haliosai.svg)](https://pypi.org/project/haliosai/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**HaliosAI : Ship Reliable AI Agents Faster!** 🚀🚀🚀

HaliosAI SDK helps you catch tricky AI agent failures before they reach users. It supports both offline and live guardrail checks, streaming response validation, parallel processing, and multi-agent setups. Integration is seamless - just add a simple decorator to your code. HaliosAI instantly plugs into your agent workflows, making it easy to add safety and reliability without changing your architecture.

## Features

- 🛡️ **Easy Integration**: Simple decorators and patchers for existing AI agent code
- ⚡ **Parallel Processing**: Run guardrails and agent calls simultaneously for optimal performance
- 🌊 **Streaming Support**: Real-time guardrail evaluation for streaming responses
- 🤖 **Multi-Agent Support**: Per-agent guardrail profiles for complex AI systems
- 🔧 **Framework Support**: Built-in support for OpenAI, Anthropic, and OpenAI Agents
- 📊 **Detailed Timing**: Performance metrics and execution insights
- 🚨 **Violation Handling**: Automatic blocking and detailed error reporting

## Installation

```bash
pip install haliosai
```

For specific LLM providers:
```bash
pip install haliosai[openai]        # For OpenAI support
pip install haliosai[agents]        # For OpenAI Agents support
pip install haliosai[all]           # For all providers
```

## Prerequisites

1. **Get your API key**: Visit [console.halios.ai](https://console.halios.ai) to obtain your HaliosAI API key
2. **Create an agent**: Follow the [documentation](https://docs.halios.ai) to create your first agent and configure guardrails
3. **Keep your agent_id handy**: You'll need it for SDK integration

Set required environment variables:
```bash
export HALIOS_API_KEY="your-api-key"
export HALIOS_AGENT_ID="your-agent-id"
export OPENAI_API_KEY="your-openai-key"  # For OpenAI examples
```

## Quick Start

### Basic Usage (Decorator Pattern)

```python
import asyncio
import os
from openai import AsyncOpenAI
from haliosai import guarded_chat_completion, GuardrailViolation

# Validate required environment variables
REQUIRED_VARS = ["HALIOS_API_KEY", "HALIOS_AGENT_ID", "OPENAI_API_KEY"]
missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID")

@guarded_chat_completion(agent_id=HALIOS_AGENT_ID)
async def call_llm(messages):
    """LLM call with automatic guardrail evaluation"""
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100
    )
    return response

async def main():
    # Customize messages for your agent's persona
    messages = [{"role": "user", "content": "Hello, can you help me?"}]
    
    try:
        response = await call_llm(messages)
        content = response.choices[0].message.content
        print(f"✓ Response: {content}")
    except GuardrailViolation as e:
        print(f"✗ Blocked: {e.violation_type} - {len(e.violations)} violation(s)")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage (Context Manager Pattern)

For fine-grained control over guardrail evaluation:

```python
import asyncio
import os
from openai import AsyncOpenAI
from haliosai import HaliosGuard, GuardrailViolation

HALIOS_AGENT_ID = os.getenv("HALIOS_AGENT_ID")

async def main():
    messages = [{"role": "user", "content": "Hello, how can you help?"}]
    
    async with HaliosGuard(agent_id=HALIOS_AGENT_ID) as guard:
        try:
            # Step 1: Validate request
            await guard.validate_request(messages)
            print("✓ Request passed")
            
            # Step 2: Call LLM
            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100
            )
            
            # Step 3: Validate response
            response_message = response.choices[0].message
            full_conversation = messages + [{"role": "assistant", "content": response_message.content}]
            await guard.validate_response(full_conversation)
            
            print("✓ Response passed")
            print(f"Response: {response_message.content}")
            
        except GuardrailViolation as e:
            print(f"✗ Blocked: {e.violation_type} - {len(e.violations)} violation(s)")

if __name__ == "__main__":
    asyncio.run(main())
```

## OpenAI Agents Framework Integration

For native integration with OpenAI Agents framework:

```python
from openai import AsyncOpenAI
from agents import Agent
from haliosai import RemoteInputGuardrail, RemoteOutputGuardrail

# Create guardrails
input_guardrail = RemoteInputGuardrail(agent_id="your-agent-id")
output_guardrail = RemoteOutputGuardrail(agent_id="your-agent-id")

# Create agent with guardrails
agent = Agent(
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    input_guardrails=[input_guardrail],
    output_guardrails=[output_guardrail]
)

# Use the agent normally - guardrails run automatically
client = AsyncOpenAI()
runner = await client.beta.agents.get_agent_runner(agent)
result = await runner.run(
    starting_agent=agent,
    input="Write a professional email"
)
```

## Examples

Check out the `examples/` directory for complete working examples.

### 🚀 Recommended Starting Point

**`06_interactive_chatbot.py`** - Interactive chat session
- Works with ANY agent configuration
- Type your own messages relevant to your agent's persona
- See guardrails in action in real-time
- Best way to explore the SDK!

### 📚 SDK Mechanics

**`01_basic_usage.py`** - Simple decorator pattern
- Shows basic `@guarded_chat_completion` usage
- Request/response guardrail evaluation
- Exception handling

**`02_streaming_response_guardrails.py`** - Streaming responses
- Real-time streaming with guardrails
- Character-based and time-based buffering
- Hybrid buffering modes

**`03_tool_calling_simple.py`** - Tool/function calling
- Guardrails for function calling scenarios
- Tool invocation tracking

**`04_context_manager_pattern.py`** - Manual control
- Context manager for explicit guardrail calls
- Separate request/response validation

**`05_tool_calling_advanced.py`** - Advanced tool calling with comprehensive guardrails
- Request validation
- Tool result validation (prevents data leakage)
- Response validation
- Context manager pattern for fine-grained control

**`05_openai_agents_guardrails_integration.py`** - OpenAI Agents framework
- Integration with OpenAI Agents SDK
- Multi-agent workflows



## Note
Currently, HaliosAI SDK supports OpenAI and OpenAI Agents frameworks natively. Other providers (e.g. Anthropic and Gemini) can be integrated using their OpenAI-compatible APIs via OpenAI SDK. Support for additional frameworks is coming soon.

This is beta release. API and features may change. Please report any issues or feedback on GitHub.

## Requirements

- Python 3.9+
- httpx >= 0.24.0
- typing-extensions >= 4.0.0

### Optional Dependencies

- openai >= 1.0.0 (for OpenAI integration)
- anthropic >= 0.25.0 (for Anthropic integration)
- openai-agents >= 0.1.0 (for OpenAI Agents integration)

## Documentation

- 📖 **Full Documentation**: [docs.halios.ai](https://docs.halios.ai)

## Support

- 🌐 **Website**: [halios.ai](https://halios.ai)
- 📧 **Email**: support@halios.ai
- � **Issues**: [GitHub Issues](https://github.com/HaliosAI/haliosai-python-sdk/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/HaliosAI/haliosai-python-sdk/discussions)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
