#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
OpenTelemetry setup for personal-morning-briefing.

Initializes the OTel SDK when OTEL_EXPORTER_OTLP_ENDPOINT is set.
If the env var is absent (or otel.enabled is false), the global NoOpTracerProvider
is left in place — all tracing calls become zero-cost no-ops.
"""

import contextvars
import logging
import os
from typing import Any, Callable, Dict, TypeVar

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.instrument import (  # noqa: PLC2701
    Counter,
    Gauge,
    Histogram,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    UpDownCounter,
)
from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_initialized = False


def setup_tracing(config: Dict[str, Any]) -> None:
    """
    Initialize OTel SDK if an OTLP endpoint is configured.

    Must be called once at startup (BriefingRunner.__init__).
    Subsequent calls are no-ops.

    Args:
        config: Full config dict (reads config["otel"]).
    """
    global _initialized
    if _initialized:
        return

    otel_cfg = config.get("otel", {})
    if not otel_cfg.get("enabled", True):
        logger.debug("OTel tracing disabled via config")
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.debug("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled")
        return

    service_name = os.getenv(
        "OTEL_SERVICE_NAME",
        otel_cfg.get("service_name", "personal-morning-briefing"),
    )
    resource = Resource.create({
        SERVICE_NAME: service_name,
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "production"),
    })

    # ── Tracing ────────────────────────────────────────────────
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    # ── Metrics ────────────────────────────────────────────────
    # Grafana (Prometheus-based) requires cumulative temporality.
    _cumulative = AggregationTemporality.CUMULATIVE
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            preferred_temporality={
                Counter: _cumulative,
                Gauge: _cumulative,
                Histogram: _cumulative,
                ObservableCounter: _cumulative,
                ObservableGauge: _cumulative,
                ObservableUpDownCounter: _cumulative,
                UpDownCounter: _cumulative,
            }
        )
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # ── Logs ───────────────────────────────────────────────────
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)

    # Auto-instrument stdlib logging so all log records are forwarded via OTel
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        LoggingInstrumentor().instrument()
    except ImportError:
        logger.debug("opentelemetry-instrumentation-logging not installed; skipping log auto-instrumentation")

    # Auto-instrument all outbound HTTP calls made via `requests`
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
    except ImportError:
        logger.debug("opentelemetry-instrumentation-requests not installed; skipping HTTP auto-instrumentation")

    _initialized = True
    logger.info(f"OTel tracing enabled → {endpoint} (service={service_name})")


def get_tracer(name: str = "personal.briefing") -> trace.Tracer:
    """Return a tracer from the current provider (NoOp if not initialised)."""
    return trace.get_tracer(name)


_F = TypeVar("_F")


def with_otel_context(fn: Callable[[], _F]) -> Callable[[], _F]:
    """Wrap a zero-argument callable so it runs with the caller's OTel context.

    Use this when submitting work to a ThreadPoolExecutor so that child spans
    appear under the correct parent trace rather than as disconnected roots.

    Uses contextvars.copy_context() which copies ALL context variables (including
    OTel's internal _CURRENT_CONTEXT) — more reliable than otel_context.attach().

    Example::

        ctx_fn = with_otel_context(worker.run)
        future = executor.submit(ctx_fn)
    """
    ctx = contextvars.copy_context()

    def _wrapper() -> _F:  # type: ignore[type-var]
        return ctx.run(fn)

    return _wrapper
