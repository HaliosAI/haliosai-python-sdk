# HaliosAI SDK Examples

This directory contains practical examples demonstrating various HaliosAI SDK features and integration patterns. Each example builds on the previous one, starting with basic usage and progressing to advanced scenarios.

## Prerequisites

Before running these examples, ensure you have:

1. **HaliosAI SDK installed**:
   ```bash
   pip install haliosai
   ```

2. **Required environment variables**:
   ```bash
   export HALIOS_API_KEY="your-halios-api-key"
   export HALIOS_AGENT_ID="your-agent-id"
   export HALIOS_BASE_URL="https://api.halioslabs.com"  # Optional
   ```

3. **LLM provider API keys** (depending on examples):
   ```bash
   export GEMINI_API_KEY="your-gemini-key"     # For Gemini examples
   export OPENAI_API_KEY="your-openai-key"     # For OpenAI examples
   ```

## Examples Overview

### 1. Basic Usage (`01_basic_usage.py`)
**Complexity: Beginner** | **Provider: Gemini**

Demonstrates the fundamental usage of HaliosAI guardrails with a simple decorator approach. Shows how to protect LLM calls with automatic request and response guardrail evaluation.

**Key Features:**
- Simple `@guarded_chat_completion` decorator usage
- Concurrent guardrail processing (guardrails run parallel to LLM calls)
- Detailed violation reporting and timing metrics
- Error handling for different scenarios

**Run it:**
```bash
python 01_basic_usage.py
```

### 2. Streaming Guardrails (`02_streaming_guardrails.py`)
**Complexity: Intermediate** | **Provider: OpenAI**

Shows real-time guardrail evaluation during streaming responses. The SDK buffers streaming content and evaluates it incrementally, allowing for early detection of violations.

**Key Features:**
- Real-time guardrail monitoring during streaming
- Configurable buffer sizes and check intervals
- Stream interruption on guardrail violations
- Event-driven architecture with detailed feedback

**Run it:**
```bash
python 02_streaming_guardrails.py
```

### 3. Tool Calling (`03_tool_calling_simple.py`)
**Complexity: Intermediate** | **Provider: Gemini**

Demonstrates guardrail protection for function/tool calling scenarios. Shows how to safely handle tool invocations while maintaining guardrail coverage.

**Key Features:**
- Tool definition and function calling support
- Guardrail evaluation for tool-based interactions
- Simulated tool execution with weather, math, and search functions
- Integration with Gemini's tool calling capabilities

**Run it:**
```bash
python 03_tool_calling_simple.py
```

### 4. OpenAI Agents Integration (`04_openai_agents_guardrails_integration.py`)
**Complexity: Advanced** | **Provider: OpenAI Agents**

Demonstrates native integration with the OpenAI Agents framework. Instead of patching clients, this shows how to add HaliosAI guardrails directly to Agent definitions.

**Key Features:**
- Native guardrail integration with OpenAI Agents
- Declarative agent configuration
- Input and output guardrail support
- Clean framework integration without client patching

**Run it:**
```bash
python 04_openai_agents_guardrails_integration.py
```

## Running Examples

Each example can be run independently:

```bash
# Navigate to examples directory
cd examples

# Run any example
python 01_basic_usage.py
python 02_streaming_guardrails.py
python 03_tool_calling_simple.py
python 04_openai_agents_guardrails_integration.py
```

## Demo Mode

If you don't have real API keys, most examples will run in demo mode with mock responses. You'll see mock guardrail evaluations and simulated LLM responses.

## Learning Path

1. **Start here**: `01_basic_usage.py` - Learn the fundamentals
2. **Then try**: `02_streaming_guardrails.py` - Add streaming support
3. **Next**: `03_tool_calling_simple.py` - Handle tool calling scenarios
4. **Finally**: `04_openai_agents_guardrails_integration.py` - Framework integration

## Troubleshooting

- **Import errors**: Ensure `haliosai` is installed and virtual environment is activated
- **API errors**: Check your environment variables and API key validity
- **Network issues**: Verify `HALIOS_BASE_URL` is correct (if set)
- **Provider errors**: Ensure you have the correct API keys for the LLM provider

## Need Help?

- Check the main [HaliosAI SDK README](../README.md) for detailed documentation
- Visit [docs.halioslabs.com](https://docs.halioslabs.com) for comprehensive guides
- Contact support at [support@halioslabs.com](mailto:support@halioslabs.com)
