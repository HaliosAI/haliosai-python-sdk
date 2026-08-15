"""Stock OpenTelemetry export; use provider/framework instrumentation in real apps."""

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider(
    resource=Resource.create(
        {
            "service.name": "support-agent",
            "service.version": os.getenv("GIT_SHA", "local"),
            "deployment.environment.name": os.getenv("APP_ENV", "development"),
        }
    )
)
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
