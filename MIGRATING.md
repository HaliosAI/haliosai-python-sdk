# Migrating from Halios SDK 1.x to 2.x

Halios SDK 2.x intentionally narrows the Python package to explicit API operations.

## Removed from the SDK

- Guardrail decorators and context managers
- Provider-specific integrations
- SDK-managed tracing and span creation
- Scenario simulation and prompt optimization

## Replacements

| 1.x behavior | 2.x replacement |
| --- | --- |
| Guardrail decorator | `Client.evaluate_request(...)` and `Client.evaluate_response(...)` |
| SDK tracing | Stock OpenTelemetry SDK and ecosystem instrumentation |
| Scenario runner | `halios eval run` from `haliosai-cli` |
| Optimizer | `halios optimize ...` from `haliosai-cli` and the Halios Agent Skill |

Install the new major version explicitly, update one application boundary at a time, and verify
stored OpenTelemetry evidence before removing 1.x integration code.
