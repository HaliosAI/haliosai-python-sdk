# HaliosAI SDK Examples

This directory contains examples demonstrating the HaliosAI SDK functionality with increasing complexity.

## Quick Start

1. Install the SDK:
   ```bash
   pip install haliosai
   ```

2. Set environment variables:
   ```bash
   export HALIOS_API_KEY="your-api-key"
   export HALIOS_APP_ID="your-app-id"
   export HALIOS_BASE_URL="https://api.halioslabs.com"  # Optional
   export OPENAI_API_KEY="your-openai-key"  # For OpenAI examples
   export GEMINI_API_KEY="your-gemini-key"  # For Gemini examples
   ```

3. Run any example:
   ```bash
   python examples/01_basic_usage.py
   ```

## Examples Overview

### 1. Basic Usage (`01_basic_usage.py`)
**Complexity: Beginner**

Demonstrates the new unified `guarded_chat_completion` decorator with different modes:
- Basic concurrent processing (recommended)
- Sequential processing (for debugging)
- Streaming with real-time guardrails
- Full configuration options

**Key Concepts:**
- `@guarded_chat_completion()` decorator
- `concurrent_guardrail_processing` parameter
- `streaming_guardrails` parameter
- Environment variable configuration

### 2. Streaming Chat (`02_streaming.py`)
**Complexity: Beginner-Intermediate**

Shows real-time guardrail evaluation during streaming responses:
- Buffer-based guardrail checking
- Configurable check intervals
- Stream interruption on violations
- Event-driven streaming architecture

**Key Concepts:**
- Real-time guardrail evaluation
- Stream buffering and checking
- Event types: `chunk`, `completed`, `error`
- Stream safety mechanisms

### 3. Tool Calling (`03_tool_calling.py`)
**Complexity: Intermediate**

Demonstrates guardrails with Gemini function/tool calling (based on auto_demo_tools.py):
- Simple tool definitions (weather, math calculations)
- Basic guardrail protection for tool usage
- Gemini API integration with OpenAI-compatible interface
- Clear error handling and response processing

**Key Concepts:**
- Tool definitions and function calling
- Gemini API integration
- Basic guardrail protection
- Simple response handling

### 4. Multi-Agent Systems (`04_multi_agent_systems.py`)
**Complexity: Advanced**

Shows per-agent guardrail profiles for complex AI systems:
- Different guardrail configurations per agent type
- Automatic agent detection and routing
- Multi-agent workflows with handoffs
- Context-aware guardrail application

**Key Concepts:**
- Per-agent app_id configuration
- Agent detection and routing
- Multi-agent coordination
- Context-aware guardrails

### 5. Performance Comparison (`05_performance.py`)
**Complexity: Intermediate**

Compares different execution modes and their performance:
- Sequential vs concurrent processing
- Timing analysis and benchmarks
- Performance optimization strategies
- Best practice recommendations

**Key Concepts:**
- Performance measurement
- Execution mode comparison
- Optimization strategies
- Timing analysis

### 6. Advanced Configuration (`06_advanced_config.py`)
**Complexity: Advanced**

Demonstrates advanced SDK configuration options:
- Custom timeout settings
- Retry mechanisms
- Error handling strategies
- Advanced logging configuration

**Key Concepts:**
- Advanced configuration options
- Error handling and retries
- Custom logging setup
- Production deployment patterns


## Common Patterns

### Basic Decorator Usage
```python
from haliosai import guarded_chat_completion

@guarded_chat_completion(app_id="your-app-id")
async def call_llm(messages):
    return await openai_client.chat.completions.create(...)
```

### Streaming with Guardrails
```python
@guarded_chat_completion(
    app_id="your-app-id",
    streaming_guardrails=True,
    stream_buffer_size=100
)
async def stream_llm(messages):
    async for chunk in openai_client.chat.completions.create(..., stream=True):
        yield chunk
```

### Multi-Agent Configuration
```python
from haliosai import patch_openai_agents_multi

agent_config = {
    'translator': {'app_id': 'app-translation'},
    'writer': {'app_id': 'app-creative'}
}

with patch_openai_agents_multi(agent_config):
    # Agents automatically use appropriate guardrail profiles
    result = await runner.run(agent, message)
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HALIOS_API_KEY` | Your HaliosAI API key | Yes |
| `HALIOS_BASE_URL` | HaliosAI service URL | No (has default) |
| `OPENAI_API_KEY` | OpenAI API key (for OpenAI examples) | For OpenAI examples |
| `HALIOS_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No (default: INFO) |

## Running Examples

All examples can be run independently:

```bash
# Basic usage
PYTHONPATH=. python examples/01_basic_usage.py

# Streaming
PYTHONPATH=. python examples/02_streaming.py

# Tool calling (requires OpenAI API key)
PYTHONPATH=. python examples/03_tool_calling.py

# Multi-agent (requires openai-agents package)
PYTHONPATH=. python examples/04_multi_agent_systems.py

# Performance comparison
PYTHONPATH=. python examples/05_performance.py

# Advanced configuration
PYTHONPATH=. python examples/06_advanced_config.py
```

## Key SDK Features Demonstrated

1. **Unified Decorator API**: Single `guarded_chat_completion` decorator for all use cases
2. **Concurrent Processing**: Guardrails run in parallel with LLM calls for optimal performance
3. **Streaming Support**: Real-time guardrail evaluation during streaming responses
4. **Multi-Agent Support**: Per-agent guardrail profiles for complex AI systems
5. **Tool Integration**: Seamless integration with OpenAI function calling
6. **Framework Support**: Built-in support for OpenAI Agents framework
7. **Performance Optimization**: Configurable timeouts, buffering, and retry mechanisms
8. **Production Ready**: Comprehensive error handling and logging

## Best Practices

1. **Use concurrent processing** (`concurrent_guardrail_processing=True`) for production
2. **Set appropriate timeouts** based on your application requirements
3. **Configure per-agent guardrails** for multi-agent systems
4. **Use streaming guardrails** for real-time applications
5. **Set up proper logging** for debugging and monitoring
6. **Handle guardrail violations** gracefully in your application logic

## Support

- 📧 Email: support@halioslabs.com
- 📖 Documentation: https://docs.halioslabs.com
- 🐛 Issues: https://github.com/halioslabs/haliosai-sdk/issues
