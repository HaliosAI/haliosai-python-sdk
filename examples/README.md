# HaliosAI SDK Examples

## Prerequisites

All examples require:

1. **Create your agent** in HaliosAI dashboard for YOUR specific use case
   - Define agent persona (e.g., "HR support bot", "E-commerce customer service", etc.)
   - Configure appropriate guardrails for your domain

2. **Set environment variables:**
   ```bash
   export HALIOS_API_KEY="your-halios-api-key"
   export HALIOS_AGENT_ID="your-agent-id"
   export OPENAI_API_KEY="your-openai-key"
   ```

## Examples

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

**`05_tool_calling_advanced.py`** - Advanced tool calling with comprehensive guardrails
- Request validation
- Tool result validation (prevents data leakage)
- Response validation
- Context manager pattern for fine-grained control

**`04_context_manager_pattern.py`** - Manual control
- Context manager for explicit guardrail calls
- Separate request/response validation

**`05_openai_agents_guardrails_integration.py`** - OpenAI Agents framework
- Integration with OpenAI Agents SDK
- Multi-agent workflows

## Running Examples

⚠️  **Important:** Update test messages in each example to match YOUR agent's persona!

```bash
# Interactive (recommended!)
python examples/06_interactive_chatbot.py

# Basic usage
python examples/01_basic_usage.py

# Streaming
python examples/02_streaming_response_guardrails.py
```

## Customization

Each example includes placeholder messages. **You must customize these** to match your agent's configured persona:

- ❌ DON'T use generic messages with a specialized agent
- ✅ DO use messages relevant to your agent's domain

Example:
- HR Agent → "I need help with my benefits"
- E-commerce → "Where is my order?"
- Code Assistant → "How do I implement binary search?"
```bash
python 01_basic_usage.py
```

### 2. Streaming Guardrails (`02_streaming_response_guardrails.py`)
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

### 4. Context Manager Pattern (`04_context_manager_pattern.py`)
**Complexity: Intermediate** | **Provider: OpenAI**

Demonstrates minimal integration using the async context manager pattern. Shows how to add guardrails with minimal code changes to existing LLM applications.

**Key Features:**
- Async context manager usage with `async with HaliosGuard()`
- Minimal code changes required for integration
- Manual request and response evaluation
- Resource management and cleanup handled automatically

**Run it:**
```bash
python 04_context_manager_pattern.py
```

### 5. OpenAI Agents Integration (`05_openai_agents_guardrails_integration.py`)
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
python 04_context_manager_pattern.py
python 05_openai_agents_guardrails_integration.py

```


## Troubleshooting

- **Import errors**: Ensure `haliosai` is installed and virtual environment is activated
- **API errors**: Check your environment variables and API key validity
- **Network issues**: Verify `HALIOS_BASE_URL` is correct (if set)
- **Provider errors**: Ensure you have the correct API keys for the LLM provider

## Need Help?

- Check the main [HaliosAI SDK README](../README.md) for detailed documentation
- Visit [docs.halioslabs.com](https://docs.halioslabs.com) for comprehensive guides
- Contact support at [support@halioslabs.com](mailto:support@halioslabs.com)
