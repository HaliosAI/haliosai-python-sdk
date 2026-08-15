# Halios Python SDK

The `haliosai` package is a small async API for the cases stock OpenTelemetry and the
[`halios`](https://github.com/HaliosAI/halios) CLI cannot cover:

- synchronously check a request or response at an application boundary;
- start and inspect an immutable evaluation run over known trace IDs; and
- retrieve a trace for programmatic inspection.

It does not contain a CLI, decorators, tracing framework, provider integrations, scenario runner,
or optimizer.

## Install

```bash
python -m pip install haliosai
```

Python 3.10 or newer is required. Configure `HALIOS_API_KEY`, `HALIOS_AGENT_ID`, and, for a
self-hosted deployment, `HALIOS_BASE_URL`.

## Inline request and response checks

```python
import haliosai


async def answer(messages: list[dict[str, str]]) -> str:
    async with haliosai.Client(agent_id="support-agent") as client:
        request_check = await client.evaluate_request(messages)
        if request_check.blocked:
            raise PermissionError(request_check.violations[0].message)

        response = await call_model(messages)  # your application code
        output_messages = [*messages, {"role": "assistant", "content": response}]
        response_check = await client.evaluate_response(output_messages)
        if response_check.blocked:
            raise PermissionError(response_check.violations[0].message)
        return response
```

These methods call Halios synchronously. Your application decides whether `block`, `flag`, or
`allow` is appropriate at the boundary. There is deliberately no hidden fail-open decorator.

## Join an existing OpenTelemetry trace

The SDK does not install or configure OpenTelemetry. If your application already has an active
span, pass its standard W3C IDs:

```python
from opentelemetry import trace
import haliosai


span_context = trace.get_current_span().get_span_context()
trace_id = trace.format_trace_id(span_context.trace_id)
parent_span_id = trace.format_span_id(span_context.span_id)

async with haliosai.Client(agent_id="support-agent") as client:
    result = await client.evaluate_request(
        [{"role": "user", "content": "Transfer the account balance"}],
        trace_id=trace_id,
        parent_span_id=parent_span_id,
    )
```

OpenTelemetry remains an application dependency only when the application uses it. See
[`OTEL_SCHEMA.md`](OTEL_SCHEMA.md)
for the OTLP and semantic evidence Halios consumes.

## Evaluate existing traces

```python
import haliosai


async with haliosai.Client(agent_id="support-agent") as client:
    run = await client.evaluate_traces(
        ["0af7651916cd43dd8448eb211c80319c"],
        run_name="release-candidate",
        fail_below=0.95,
        labels=["ci", "sha:abc123"],
    )
    report = await client.wait_for_evaluation_run(run.run_id)
    if not report.gate_passed:
        raise RuntimeError(f"Halios gate failed: pass@k={report.pass_at_k:.3f}")
```

For repository configuration, scenarios, simulations, CI gates, production-failure analysis, and
coding-agent optimization, install `haliosai-cli` or the
[Halios Agent Skill](https://github.com/HaliosAI/halios/tree/main/skills/halios) instead.

## Public API

- `Client.evaluate_request(...)`
- `Client.evaluate_response(...)`
- `Client.evaluate(...)` for custom message sequences
- `Client.evaluate_traces(...)`
- `Client.get_evaluation_run(...)`
- `Client.wait_for_evaluation_run(...)`
- `Client.get_trace(...)`

All I/O is async. API failures raise `HaliosAPIError`; invalid local configuration raises
`ConfigError`; polling timeouts raise `HaliosTimeoutError`.

## Upgrading from 1.x

Version 2 is an intentionally small, explicit client. The 1.x decorators, provider integrations,
tracing framework, scenario runner, and optimizer are not part of the 2.x SDK. Applications should
use stock OpenTelemetry for tracing, call `Client.evaluate_request(...)` or
`Client.evaluate_response(...)` at explicit inline boundaries, and use the separate Halios CLI for
repository evaluation workflows. See [MIGRATING.md](MIGRATING.md) for the complete migration guide.
