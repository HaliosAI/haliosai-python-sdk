# Halios Python SDK

The Python client for using [Halios](https://halios.ai) from application code.

Use it to evaluate requests and responses at runtime, run evaluations over existing traces, and retrieve trace and evaluation results.

> **Building an evaluation suite?** Start with the [Halios skill](https://github.com/HaliosAI/halios). It lets Codex, Claude Code, Cursor, and other coding agents create scenarios, run fresh multi-turn trials, and investigate failures directly from your repository.

```bash id="4r49yf"
npx skills add HaliosAI/halios --skill halios
```

### When to use the SDK

Use the Python SDK when you need to evaluate a request or response **inline at runtime** — for example, to enforce a Halios check as a guardrail before accepting a request or returning an agent response.

Checks used this way must be configured as **guardrails** in Halios.

For normal post-hoc evaluation, debugging, regression testing, and production analysis, we recommend sending your agent traces to Halios using a standard OpenTelemetry exporter. Halios can then evaluate those traces without adding evaluation calls to your application code.

## Installation

```bash id="ls7fn3"
pip install haliosai
```

Requires Python 3.10+.

Set your Halios API key:

```bash id="dukp4u"
export HALIOS_API_KEY="your-api-key"
```

For self-hosted deployments, set `HALIOS_BASE_URL`.

## Evaluate requests and responses

```python id="alv9d7"
import haliosai

async with haliosai.Client(agent_id="support-agent") as client:
    result = await client.evaluate_request([
        {"role": "user", "content": "Transfer funds to account #9821"}
    ])

    if result.blocked:
        raise PermissionError(result.violations[0].message)
```

Use `evaluate_response()` to evaluate an agent response before returning it to the caller.

### Attach to an existing OpenTelemetry trace

If your application already has an active OpenTelemetry span, pass its W3C trace and span IDs so the Halios evaluation is connected to the same trace:

```python id="g5wzak"
from opentelemetry import trace
import haliosai

ctx = trace.get_current_span().get_span_context()

async with haliosai.Client(agent_id="support-agent") as client:
    result = await client.evaluate_request(
        [{"role": "user", "content": "Transfer funds to account #9821"}],
        trace_id=trace.format_trace_id(ctx.trace_id),
        parent_span_id=trace.format_span_id(ctx.span_id),
    )
```

The SDK does not provide a proprietary tracing layer. Halios accepts standard OpenTelemetry traces and GenAI semantic conventions.

## Evaluate existing traces

Run evaluation checks against one or more traces:

```python id="da9oyt"
import haliosai

async with haliosai.Client(agent_id="support-agent") as client:
    run = await client.evaluate_traces(
        trace_ids,
        run_name="release-gate",
        fail_below=0.95,
    )

    report = await client.wait_for_evaluation_run(run.run_id)

    if not report.gate_passed:
        raise RuntimeError("Halios evaluation failed")
```

## Client API

| Method | Description |
| --- | --- |
| `evaluate_request(messages, ...)` | Evaluate an incoming user request. |
| `evaluate_response(messages, ...)` | Evaluate an agent response. |
| `evaluate(messages, ...)` | Evaluate an arbitrary message sequence. |
| `evaluate_traces(trace_ids, ...)` | Run evaluations against existing traces. |
| `get_evaluation_run(run_id)` | Retrieve an evaluation run and its results. |
| `wait_for_evaluation_run(run_id, ...)` | Wait for an evaluation run to complete. |
| `get_trace(trace_id)` | Retrieve a trace and its spans. |

## Upgrading from 1.x

Version 1.x is deprecated. See [MIGRATING.md](./MIGRATING.md) for migration instructions.

## Resources

- [Halios documentation](https://docs.halios.ai)
- [Halios skill and CLI](https://github.com/HaliosAI/halios)
- [Changelog](./CHANGELOG.md)

## License

[Apache 2.0](./LICENSE) © Anomalytica Inc. 2026
