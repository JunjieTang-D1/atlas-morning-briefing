#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
OpenTelemetry setup for personal-morning-briefing.

Initializes the OTel SDK when OTEL_EXPORTER_OTLP_ENDPOINT is set.
If the env var is absent (or otel.enabled is false), the global NoOpTracerProvider
is left in place — all tracing calls become zero-cost no-ops.
"""

import logging
import os
from typing import Any, Callable, Dict, TypeVar

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

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
    resource = Resource({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

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

    Example::

        ctx_fn = with_otel_context(worker.run)
        future = executor.submit(ctx_fn)
    """
    ctx = otel_context.get_current()

    def _wrapper() -> _F:  # type: ignore[type-var]
        token = otel_context.attach(ctx)
        try:
            return fn()
        finally:
            otel_context.detach(token)

    return _wrapper
