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

## Quick Start

### Basic Usage

```python
import asyncio
from haliosai import guarded_chat_completion

# Basic usage with concurrent guardrail processing (default)
@guarded_chat_completion(agent_id="your-agent-id")
async def call_llm(messages):
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response

# Use the guarded function
messages = [{"role": "user", "content": "Hello!"}]
response = await call_llm(messages)
```

### Configuration

Set your API key as an environment variable:
```bash
export HALIOS_API_KEY="your-api-key"
```

Or pass it directly:
```python
@guarded_chat_completion(
    agent_id="your-agent-id",
    api_key="your-api-key"
)
async def call_llm(messages):
    # Your agent implementation
    pass
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

Check out the `examples/` directory for complete working examples:

## Advanced Usage

### Streaming Response Guardrails Support

```python
@guarded_chat_completion(
    agent_id="your-agent-id",
    streaming_guardrails=True,
    stream_buffer_size=100
)
async def stream_llm_call(messages):
    async for chunk in openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    ):
        yield chunk

# Handle streaming events
async for event in stream_llm_call(messages):
    if event['type'] == 'chunk':
        print(event['content'], end='')
    elif event['type'] == 'violation':
        print(f"Content blocked: {event['violations']}")
        break
```

### Performance Optimization

```python
# Sequential processing (for debugging)
@guarded_chat_completion(
    agent_id="your-agent-id", 
    concurrent_guardrail_processing=False
)
async def debug_llm_call(messages):
    return await openai_client.chat.completions.create(...)

# Custom timeout settings
@guarded_chat_completion(
    agent_id="your-agent-id",
    guardrail_timeout=10.0  # Increase timeout for slow networks
)
async def slow_network_call(messages):
    return await openai_client.chat.completions.create(...)
```

### Error Handling

```python
from haliosai import guarded_chat_completion, ExecutionResult

@guarded_chat_completion(agent_id="your-agent-id")
async def protected_agent_call(messages):
    return await agent_call(messages)

# Better approach: Check execution result instead of catching exceptions
result = await protected_agent_call(messages)

if hasattr(result, '_halios_execution_result'):
    execution_result = result._halios_execution_result
    
    if execution_result.result == ExecutionResult.REQUEST_BLOCKED:
        print(f"Request blocked: {execution_result.request_violations}")
        # Handle blocked request appropriately
    elif execution_result.result == ExecutionResult.RESPONSE_BLOCKED:
        print(f"Response blocked: {execution_result.response_violations}")
        # Handle blocked response appropriately
    elif execution_result.result == ExecutionResult.SUCCESS:
        print("Agent call completed successfully")
        # Use the response normally
else:
    # Fallback: handle the legacy ValueError approach
    try:
        response = await protected_agent_call(messages)
    except ValueError as e:
        if "blocked by guardrails" in str(e):
            print(f"Content blocked: {e}")
            # Handle blocked content appropriately
        else:
            raise
```

## Note
Currently, HaliosAI SDK supports OpenAI and OpenAI Agents frameworks natively. Other providers (e.g. Anthropic and Gemini) can be integrated using their OpenAI-compatible APIs via OpenAI SDK. Support for additional frameworks is coming soon.

This is beta release. API and features may change. Please report any issues or feedback on GitHub.

## Requirements

- Python 3.8+
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
