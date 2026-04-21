"""Re-export shim — CircuitBreaker now lives in n8t_scraper."""

from n8t_scraper.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitOpenError

__all__ = ["CircuitBreaker", "CircuitBreakerRegistry", "CircuitOpenError"]
