# OpenTelemetry schema consumed by Halios

Halios accepts standard OTLP/HTTP trace export requests at the complete signal endpoint printed by
`halios project instrumentation`:

```text
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://app.halios.ai/v1/traces
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer%20<agent-ingest-token>
```

The endpoint is an OTLP `ExportTraceServiceRequest` receiver. A successful export confirms receipt;
evaluation readiness additionally requires meaningful, ended spans and captured semantic content.

## Identity and topology

- Trace IDs are 16-byte, non-zero W3C identifiers (32 lowercase hexadecimal characters in JSON).
- Span IDs are 8-byte, non-zero W3C identifiers (16 lowercase hexadecimal characters in JSON).
- `parentSpanId` describes the real runtime relationship; Halios does not infer a synthetic chain.
- One request, workflow invocation, or agent turn should normally remain one trace across model,
  tool, retrieval, and delegated-agent operations.
- W3C `traceparent` propagation is preferred across HTTP, queues, subprocesses, and the Halios eval
  adapter.

## Resource attributes

Halios consumes standard resource attributes and expects at least:

| Attribute | Purpose |
| --- | --- |
| `service.name` | Stable application or agent service identity. |
| `service.version` | Immutable deployed Git SHA, image digest, or release identity. |
| `deployment.environment.name` | Exact environment identity; deployed services use `staging` or `production`. |
| `service.instance.id` | Recommended replica, task, or function invocation identity. |

Use `OTEL_SERVICE_NAME` for the stable service name and include the other values in
`OTEL_RESOURCE_ATTRIBUTES`. Keep W3C `tracecontext,baggage` propagation enabled. The project jsonl
adapter is a local/CI simulation bridge: `halios eval run` injects its `ad_hoc` or `ci` identity.
Real staging and production entrypoints must initialize and export their own OpenTelemetry and do
not run through the adapter.

Use the OpenTelemetry GenAI schema URL emitted by the instrumentation in use. The current dedicated
GenAI conventions repository publishes `https://opentelemetry.io/schemas/gen-ai/1.42.0`; do not
substitute a Halios schema URL.

## GenAI span evidence

Halios consumes the current structured OpenTelemetry GenAI attributes below. Instrumentations may
encode structured values in OTLP `AnyValue` form or as JSON strings where the convention permits.

| Operation | Standard attributes Halios consumes |
| --- | --- |
| Any GenAI operation | `gen_ai.operation.name` |
| Agent/workflow | `gen_ai.agent.name`, structured input/output messages |
| Model inference | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions` |
| Tool call | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |
| Retrieval | `gen_ai.retrieval.query.text`, `gen_ai.retrieval.documents` |

Halios also preserves ordinary span name, kind, status, timestamps, events, links, and unrecognized
attributes. OpenTelemetry `SpanKind` is transport topology, not an agent/model/tool taxonomy; emit
the semantic operation attributes instead of choosing a misleading span kind.

`gen_ai.input.messages` and `gen_ai.output.messages` must follow the OpenTelemetry structured
message schemas and preserve message order. Content capture is commonly opt-in and can include PII.
Enable it deliberately for evaluation, then apply redaction, allow-listing, and sampling appropriate
to the environment. Never export credentials, authorization headers, secrets, embedding vectors,
or hidden reasoning.

## Instrumentation ownership

Use maintained OpenTelemetry instrumentation for the provider or framework. Initialize it before
constructing instrumented clients and preserve an existing global `TracerProvider`; add an exporter
or processor rather than replacing the provider.

The `haliosai` Python package does not configure OpenTelemetry. Its explicit inline-evaluation calls
may receive `trace_id` and `parent_span_id` from the application so the server-created guardrail span
joins the same trace.

## Compatibility policy

The public launch contract is OTLP plus the OpenTelemetry GenAI semantic conventions above. New
instrumentation should not emit proprietary `halios.*` span attributes. Older aliases may still be
observable in internal ingestion code during development, but they are not a documented public
compatibility promise.

## Upstream specifications

- [OTLP specification](https://opentelemetry.io/docs/specs/otlp/)
- [OTLP exporter configuration](https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/)
- [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
