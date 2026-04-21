#!/usr/bin/env python3
"""
Simple thread-safe circuit breaker for external HTTP/service calls.

States:
  CLOSED    — normal operation; failures are counted
  OPEN      — failing fast; all calls raise CircuitOpenError immediately
  HALF_OPEN — one probe call allowed; success → CLOSED, failure → OPEN

Usage::

    from scripts.circuit_breaker import CircuitBreakerRegistry

    cb = CircuitBreakerRegistry.get("brave-search")
    try:
        result = cb.call(requests.get, url, ...)
    except CircuitOpenError:
        return []  # fast-fail, skip this source
"""

import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is open."""


class CircuitBreaker:
    """Per-service circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = self.CLOSED
        self._failure_count = 0
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._check_and_transition()

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* through the circuit breaker.

        Raises CircuitOpenError if the circuit is open and the recovery
        timeout has not yet elapsed.  On success, resets the failure counter.
        On failure, increments the counter and may trip the circuit.
        """
        with self._lock:
            current_state = self._check_and_transition()

            if current_state == self.OPEN:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN — skipping call to protect the pipeline"
                )

            # Allow the call (CLOSED or HALF_OPEN probe)
            if current_state == self.HALF_OPEN:
                logger.debug(f"Circuit '{self.name}' HALF_OPEN — probing")

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            with self._lock:
                self._on_failure(exc)
            raise

        with self._lock:
            self._on_success()
        return result

    # ------------------------------------------------------------------
    # Internal helpers (must be called with _lock held)

    def _check_and_transition(self) -> str:
        if self._state == self.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout:
                self._state = self.HALF_OPEN
                logger.info(
                    f"Circuit '{self.name}': OPEN → HALF_OPEN after {elapsed:.0f}s"
                )
        return self._state

    def _on_success(self) -> None:
        if self._state in (self.HALF_OPEN, self.CLOSED):
            if self._failure_count > 0:
                logger.info(
                    f"Circuit '{self.name}': recovered — resetting failure count"
                )
            self._failure_count = 0
            self._state = self.CLOSED

    def _on_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        logger.warning(
            f"Circuit '{self.name}': failure #{self._failure_count} — {exc}"
        )

        if self._state == self.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            self._opened_at = time.monotonic()
            logger.error(
                f"Circuit '{self.name}': tripped OPEN after {self._failure_count} failure(s) — "
                f"will retry after {self.recovery_timeout:.0f}s"
            )


class CircuitBreakerRegistry:
    """Global registry of named circuit breakers (one per service)."""

    _breakers: dict = {}
    _lock = threading.Lock()

    @classmethod
    def get(
        cls,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ) -> CircuitBreaker:
        """Return the named breaker, creating it if it doesn't exist yet."""
        with cls._lock:
            if name not in cls._breakers:
                cls._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )
            return cls._breakers[name]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all breakers to CLOSED state (useful in tests)."""
        with cls._lock:
            cls._breakers.clear()
