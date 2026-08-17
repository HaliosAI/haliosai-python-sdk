# Halios Python SDK

[![PyPI version](https://img.shields.io/pypi/v/haliosai.svg)](https://pypi.org/project/haliosai/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

The official Python client library for [Halios](https://halios.ai). The `haliosai` SDK provides lightweight, explicit async APIs for:

- **Runtime Guardrails**: Synchronously validate user inputs and agent responses at application boundaries with policy checks and LLM judges.
- **Programmatic Evaluations**: Trigger, score, and wait on evaluation runs over specific OpenTelemetry trace IDs.
- **Trace Inspection**: Retrieve trace evidence and check execution results programmatically.

---

## Tracing Philosophy: Standard OpenTelemetry

Unlike traditional AI observability tools, the Halios SDK **does not bundle a proprietary tracing framework** or require intrusive decorators in your codebase.

Instead, Halios expects applications to use standard, vendor-neutral **OpenTelemetry**. You instrument your agent using stock OpenTelemetry SDKs and GenAI semantic conventions, forwarding traces to Halios. This keeps your production runtime vendor-neutral, portable, and free of vendor lock-in.

---

## 💡 Recommended: Use the Halios Agent Skill & CLI

For setting up OpenTelemetry telemetry, authoring test suites, simulating multi-turn scenarios, and gating CI pull requests, we recommend using the **[Halios Agent Skill & CLI](https://github.com/HaliosAI/halios)**.

The Halios skill pairs with your AI coding agent (**Codex**, **Claude Code**, **Cursor**, etc.) to inspect your codebase, configure telemetry, and build complete evaluation suites in minutes:

```bash
# Install the Halios Agent Skill
npx skills add HaliosAI/halios --skill halios
```

---

## Installation

```bash
pip install haliosai
```

*Requires Python 3.10 or newer.*

Configure your environment variables:
```bash
export HALIOS_API_KEY="your-api-key"
export HALIOS_AGENT_ID="your-agent-id"
# Optional for self-hosted instances (defaults to https://app.halios.ai)
export HALIOS_BASE_URL="https://app.halios.ai"
```

---

## Usage

### 1. Inline Request & Response Guardrails

Synchronously enforce guardrail policies before calling your model and before returning responses to callers:

```python
import haliosai


async def guarded_agent_turn(messages: list[dict[str, str]]) -> str:
    async with haliosai.Client(agent_id="support-agent") as client:
        # 1. Guardrail input before model call
        request_check = await client.evaluate_request(messages)
        if request_check.blocked:
            raise PermissionError(request_check.violations[0].message)

        # 2. Call your agent / LLM
        response = await call_model(messages)

        # 3. Guardrail output before returning to user
        output_messages = [*messages, {"role": "assistant", "content": response}]
        response_check = await client.evaluate_response(output_messages)
        if response_check.blocked:
            raise PermissionError(response_check.violations[0].message)

        return response
```

### 2. Join an Existing OpenTelemetry Trace

If your application already has an active OpenTelemetry span, pass standard W3C trace and span IDs so Halios guardrail spans link directly into your trace graph:

```python
from opentelemetry import trace
import haliosai


span_context = trace.get_current_span().get_span_context()
trace_id = trace.format_trace_id(span_context.trace_id)
parent_span_id = trace.format_span_id(span_context.span_id)

async with haliosai.Client(agent_id="support-agent") as client:
    result = await client.evaluate_request(
        [{"role": "user", "content": "Transfer funds to account #9821"}],
        trace_id=trace_id,
        parent_span_id=parent_span_id,
    )
```

### 3. Trigger Programmatic Trace Evaluations

Trigger post-hoc evaluation runs over specific W3C trace IDs (e.g. as part of an automated release check):

```python
import haliosai


async def run_release_gate(trace_ids: list[str]) -> None:
    async with haliosai.Client(agent_id="support-agent") as client:
        run = await client.evaluate_traces(
            trace_ids,
            run_name="release-gate-v2.0",
            fail_below=0.95,
            labels=["ci", "service:support"],
        )
        report = await client.wait_for_evaluation_run(run.run_id)
        if not report.gate_passed:
            raise RuntimeError(f"Halios gate failed: pass@k={report.pass_at_k:.3f}")
```

---

## Public API Reference

| Method | Description |
|---|---|
| `Client.evaluate_request(messages, ...)` | Synchronous input check (requires last message role to be `user`). |
| `Client.evaluate_response(messages, ...)` | Synchronous output check (requires last message role to be `assistant`). |
| `Client.evaluate(messages, ...)` | Synchronous check for arbitrary message sequences. |
| `Client.evaluate_traces(trace_ids, ...)` | Create an immutable evaluation run over explicit trace IDs. |
| `Client.get_evaluation_run(run_id)` | Fetch evaluation run status, pass@k score, and trial summaries. |
| `Client.wait_for_evaluation_run(run_id, ...)` | Poll until evaluation run completes or reaches timeout. |
| `Client.get_trace(trace_id)` | Retrieve stored spans and metadata for a trace. |

---

## Deprecation Notice

> **Version 1.x is deprecated.** Version 2.x is an explicit, lightweight client library. Tracing is handled via stock OpenTelemetry, and repository test suites, multi-turn simulations, and prompt optimization are managed via the [Halios CLI & Skill](https://github.com/HaliosAI/halios).

---

## License

[Apache 2.0](LICENSE) © Anomalytica Inc. 2026
