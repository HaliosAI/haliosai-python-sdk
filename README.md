# HaliosAI SDK

[![PyPI version](https://badge.fury.io/py/haliosai.svg)](https://badge.fury.io/py/haliosai)
[![Python Support](https://img.shields.io/pypi/pyversions/haliosai.svg)](https://pypi.org/project/haliosai/)
[![License: MIT### Error Handling

```python
from haliosai import guarded_chat_completion

@guarded_chat_completion(app_id="your-app-id")
async def protected_llm_call(messages):
    return await llm_call(messages)

try:
    response = await protected_llm_call(messages)
except ValueError as e:
    if "blocked by guardrails" in str(e):
        print(f"Content blocked: {e}")
        # Handle blocked content appropriately
    else:
        raise
```lds.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HaliosAI** is a powerful Python SDK for integrating AI guardrails with Large Language Model (LLM) applications. It provides simple patching, parallel processing, streaming support, and multi-agent configurations to help you build safer AI applications.

## Features

- 🛡️ **Easy Integration**: Simple decorators and patchers for existing LLM code
- ⚡ **Parallel Processing**: Run guardrails and LLM calls simultaneously for optimal performance
- 🌊 **Streaming Support**: Real-time guardrail evaluation for streaming responses
- 🤖 **Multi-Agent Support**: Per-agent guardrail profiles for complex AI systems
- 🔧 **Framework Support**: Built-in support for OpenAI, Anthropic, and OpenAI Agents
- 📊 **Detailed Timing**: Performance metrics and execution insights
- 🚨 **Violation Handling**: Automatic blocking and detailed error reporting

## Quick Start

### Installation

```bash
pip install haliosai
```

For specific LLM providers:
```bash
pip install haliosai[openai]        # For OpenAI support
pip install haliosai[anthropic]     # For Anthropic support  
pip install haliosai[agents]        # For OpenAI Agents support
pip install haliosai[all]           # For all providers
```

### Basic Usage

#### Simple Decorator Pattern

```python
import asyncio
from haliosai import guarded_chat_completion

# Basic usage with concurrent guardrail processing (default)
@guarded_chat_completion(app_id="your-app-id")
async def call_llm(messages):
    # Your LLM call here
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response

# Use the guarded function
messages = [{"role": "user", "content": "Hello!"}]
response = await call_llm(messages)
```

#### Sequential Processing (for debugging)

```python
from haliosai import guarded_chat_completion

@guarded_chat_completion(
    app_id="your-app-id", 
    concurrent_guardrail_processing=False
)
async def debug_llm_call(messages):
    # Guardrails run sequentially (easier to debug)
    return await openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

#### Streaming Support

```python
from haliosai import guarded_chat_completion

@guarded_chat_completion(
    app_id="your-app-id",
    streaming_guardrails=True,
    stream_buffer_size=100
)
async def stream_llm_call(messages):
    # Your streaming LLM implementation
    async for chunk in openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    ):
        yield chunk

# Use streaming with real-time guardrails
async for event in stream_llm_call(messages):
    if event['type'] == 'chunk':
        print(event['content'], end='')
    elif event['type'] == 'completed':
        print("\\nStream completed!")
```

### Framework Integration

#### OpenAI Agents Framework

```python
from haliosai import patch_openai_agents

# Single agent mode
with patch_openai_agents(app_id="your-app-id") as patcher:
    # All OpenAI calls are automatically guarded
    result = await runner.run(agent, message)
```

#### Multi-Agent Systems

```python
from haliosai import patch_openai_agents_multi

# Configure different guardrail profiles for different agents
agent_config = {
    'orchestrator': {
        'app_id': 'app-orchestrator', 
        'description': 'Main coordination agent'
    },
    'translator': {
        'app_id': 'app-translation', 
        'description': 'Translation specialist'
    },
    'synthesizer': {
        'app_id': 'app-synthesis', 
        'description': 'Content synthesis agent'
    }
}

with patch_openai_agents_multi(agent_config) as patcher:
    # Different agents automatically use their configured guardrail profiles
    result = await multi_agent_workflow()
```

#### Auto-Patch All Clients

```python
from haliosai import patch_all

# Automatically patch OpenAI, Anthropic, and Agents
guard_instance = patch_all(app_id="your-app-id")

# Now all LLM calls are automatically protected
response1 = await openai_client.chat.completions.create(...)
response2 = await anthropic_client.messages.create(...)
```

## Configuration

### Environment Variables

```bash
export HALIOS_API_KEY="your-api-key"
export HALIOS_BASE_URL="https://api.halioslabs.com"  # Optional
export HALIOS_LOG_LEVEL="INFO"  # Optional: DEBUG, INFO, WARNING, ERROR
```

### Programmatic Configuration

```python
from haliosai import guarded_chat_completion

@guarded_chat_completion(
    app_id="your-app-id",
    api_key="your-api-key",  # Or set HALIOS_API_KEY env var
    base_url="https://api.halioslabs.com",  # Optional
    concurrent_guardrail_processing=True,  # Enable parallel processing
    streaming_guardrails=False,  # Enable for streaming use cases
    stream_buffer_size=50,  # Characters before guardrail check
    guardrail_timeout=5.0  # Timeout for guardrail operations
)
async def configured_llm_call(messages):
    return await openai_client.chat.completions.create(...)
```

## Advanced Usage

### Custom Response Handling

```python
from haliosai import ParallelGuardedChat, ExecutionResult

async def advanced_llm_call():
    async with ParallelGuardedChat(app_id="your-app-id") as guard:
        result = await guard.guarded_call_parallel(
            messages=[{"role": "user", "content": "Hello!"}],
            llm_func=your_llm_function
        )
        
        if result.result == ExecutionResult.SUCCESS:
            print(f"Success! Response: {result.final_response}")
            print(f"Timing: {result.timing}")
        elif result.result == ExecutionResult.REQUEST_BLOCKED:
            print(f"Request blocked: {result.request_violations}")
        elif result.result == ExecutionResult.RESPONSE_BLOCKED:
            print(f"Response blocked: {result.response_violations}")
```

### Direct Client Patching

```python
from haliosai import patch_openai, patch_anthropic
import openai
import anthropic

# Create guard using the new decorator syntax (or legacy guard() function)
from haliosai import HaliosGuard
my_guard = HaliosGuard(app_id="your-app-id")

# Patch specific clients
patch_openai(my_guard)
patch_anthropic(my_guard)

# All subsequent calls are protected
response = await openai.AsyncOpenAI().chat.completions.create(...)
```

## Performance

HaliosAI is designed for production use with minimal performance impact:

- **Parallel Mode**: Guardrails run simultaneously with LLM calls, saving up to 80% of guardrail overhead
- **Efficient Networking**: Persistent HTTP connections and optimized request handling
- **Smart Streaming**: Configurable buffer sizes and check intervals for optimal streaming performance
- **Context-Aware**: Intelligent agent detection for multi-agent systems

## Error Handling

```python
from haliosai import guard

@guard(app_id="your-app-id")
async def protected_llm_call(messages):
    return await llm_call(messages)

try:
    response = await protected_llm_call(messages)
except ValueError as e:
    if "blocked by guardrails" in str(e):
        print(f"Content blocked: {e}")
        # Handle blocked content appropriately
    else:
        raise
```

## Logging

HaliosAI provides detailed logging for debugging and monitoring:

```python
import logging
from haliosai.config import setup_logging

# Configure logging
setup_logging("DEBUG")

# Or use environment variable
# export HALIOS_LOG_LEVEL=DEBUG
```

## Requirements

- Python 3.8+
- httpx >= 0.24.0
- typing-extensions >= 4.0.0

### Optional Dependencies

- openai >= 1.0.0 (for OpenAI integration)
- anthropic >= 0.25.0 (for Anthropic integration)
- openai-agents >= 0.1.0 (for OpenAI Agents integration)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: support@halioslabs.com
- 📖 Documentation: https://docs.halioslabs.com
- 🐛 Issues: https://github.com/halioslabs/haliosai-sdk/issues
- 💬 Discussions: https://github.com/halioslabs/haliosai-sdk/discussions

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.
