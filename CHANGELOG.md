# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-16

### Added
- Initial release of HaliosAI SDK
- Core `HaliosGuard` class for basic guardrail integration
- `ParallelGuardedChat` class for advanced parallel processing
- Support for parallel execution of guardrails and LLM calls
- Real-time streaming support with configurable buffer sizes
- Multi-agent support with per-agent guardrail profiles
- OpenAI Agents framework integration
- Decorator patterns for easy function wrapping
- Automatic client patching for OpenAI and Anthropic
- Comprehensive error handling and violation reporting
- Detailed performance timing and metrics
- Context-aware agent detection for multi-agent systems
- Configurable logging with multiple levels
- Environment variable configuration support
- Type hints and comprehensive documentation

### Features
- 🛡️ Simple decorators and patchers for existing LLM code
- ⚡ Parallel processing for optimal performance
- 🌊 Real-time guardrail evaluation for streaming
- 🤖 Multi-agent configurations with context awareness
- 🔧 Built-in support for OpenAI, Anthropic, and OpenAI Agents
- 📊 Performance metrics and execution insights
- 🚨 Automatic violation handling and detailed error reporting

### Technical Details
- Python 3.8+ support
- Async/await throughout for optimal performance
- HTTP connection pooling with httpx
- Comprehensive type annotations
- Modular architecture for extensibility
- Context manager support for automatic cleanup
- Environment-based configuration
- Structured logging with configurable levels
