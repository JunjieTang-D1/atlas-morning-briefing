#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
LLM client with tiered model support.

Primary: MiniMax 2.7 via Anthropic-compatible API.
Fallback: OpenRouter free models via OpenAI-compatible API.
"""

import logging
import os
import re
from typing import Any, Dict, Optional

import requests
from opentelemetry import trace
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from scripts.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError

_llm_tracer = trace.get_tracer("personal.llm")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM inference with tiered model support and fallback."""

    DEFAULT_PRIMARY = {
        "provider": "minimax",
        "base_url": "https://api.minimax.io",
        "model": "MiniMax-M2.7",
    }

    DEFAULT_FALLBACK_MODELS = {
        "heavy": "nvidia/nemotron-3-super-120b-a12b:free",
        "medium": "google/gemma-4-31b-it:free",
        "light": "google/gemma-4-26b-a4b-it:free",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLMClient.

        Args:
            config: LLM configuration from config.yaml.
                    Keys: enabled, temperature, max_tokens, max_calls_per_run,
                    primary (provider, base_url, model),
                    fallback (provider, base_url, models).
        """
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.3)
        self.max_calls = config.get("max_calls_per_run", 20)
        self._call_count = 0
        self._available = None

        # Primary provider (MiniMax via Anthropic-compatible API)
        primary = config.get("primary", {})
        self._primary_base_url = primary.get(
            "base_url", self.DEFAULT_PRIMARY["base_url"]
        )
        self._primary_model = primary.get("model", self.DEFAULT_PRIMARY["model"])
        self._primary_api_key = os.environ.get("MINIMAX_API_KEY", "")

        # Fallback provider (OpenRouter via OpenAI-compatible API)
        fallback = config.get("fallback", {})
        self._fallback_base_url = fallback.get(
            "base_url", "https://openrouter.ai/api/v1/chat/completions"
        )
        fallback_models = fallback.get("models", {})
        self._fallback_models = {
            "heavy": fallback_models.get(
                "heavy", self.DEFAULT_FALLBACK_MODELS["heavy"]
            ),
            "medium": fallback_models.get(
                "medium", self.DEFAULT_FALLBACK_MODELS["medium"]
            ),
            "light": fallback_models.get(
                "light", self.DEFAULT_FALLBACK_MODELS["light"]
            ),
        }
        self._fallback_api_key = os.environ.get("OPENROUTER_API_KEY", "")

    @property
    def available(self) -> bool:
        """Check if LLM features are available and enabled."""
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False
        if not self._primary_api_key and not self._fallback_api_key:
            logger.warning(
                "No API keys found (MINIMAX_API_KEY or OPENROUTER_API_KEY). "
                "LLM features disabled."
            )
            self._available = False
            return False
        self._available = True
        return True

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        name: str = "llm_call",
    ) -> Optional[str]:
        """
        Invoke an LLM model. Tries primary (MiniMax), falls back to OpenRouter.

        Args:
            prompt: User prompt text.
            tier: Model tier - "heavy", "medium", or "light".
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            system_prompt: Optional system prompt.
            name: Span name used for OTel tracing (e.g. "synthesize_briefing").

        Returns:
            Model response text, or None if invocation fails.
        """
        if not self.available:
            logger.debug("LLM not available, skipping invocation")
            return None

        if self._call_count >= self.max_calls:
            logger.warning(
                f"LLM call budget exhausted ({self.max_calls} calls). "
                "Skipping invocation. Increase llm.max_calls_per_run to allow more."
            )
            return None
        self._call_count += 1

        tokens = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        with _llm_tracer.start_as_current_span(name) as span:
            span.set_attribute("langfuse.observation.type", "generation")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.tier", tier)
            span.set_attribute("gen_ai.request.max_tokens", tokens)
            span.set_attribute("gen_ai.request.temperature", temp)

            result = None
            used_fallback = False

            # Try primary (MiniMax)
            if self._primary_api_key:
                span.set_attribute("gen_ai.request.model", self._primary_model)
                span.set_attribute("gen_ai.system", "minimax")
                result, usage = self._invoke_minimax(prompt, tokens, temp, system_prompt)
                if result is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
                    span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
                else:
                    logger.warning("MiniMax invocation failed, trying fallback")

            # Try fallback (OpenRouter)
            if result is None and self._fallback_api_key:
                fallback_model = self._fallback_models.get(tier, self._fallback_models["medium"])
                span.set_attribute("gen_ai.request.model", fallback_model)
                span.set_attribute("gen_ai.system", "openrouter")
                result, usage = self._invoke_openrouter(
                    prompt, fallback_model, tokens, temp, system_prompt
                )
                if result is not None:
                    used_fallback = True
                    span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
                    span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))

            span.set_attribute("gen_ai.fallback_used", used_fallback)

            # Langfuse input/output attributes (truncated to avoid oversized spans)
            span.set_attribute("langfuse.observation.input", prompt[:2000])
            if result is not None:
                span.set_attribute("langfuse.observation.output", result[:2000])

            if result is None:
                logger.error("All LLM providers failed")
                span.set_status(trace.StatusCode.ERROR, "All LLM providers failed")

            return result

    # Minimum token budget for MiniMax M2.7 (reasoning model needs room to think + respond)
    _MINIMAX_MIN_TOKENS = 8192

    def _invoke_minimax(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ):
        """Invoke MiniMax via native OpenAI-compatible API.

        Returns:
            Tuple of (response_text, usage_dict). Both are None/empty on failure.
        """
        # Reasoning model exhausts small budgets on <think> blocks; enforce a floor
        max_tokens = max(max_tokens, self._MINIMAX_MIN_TOKENS)
        url = f"{self._primary_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._primary_api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": self._primary_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        def _is_5xx(exc: BaseException) -> bool:
            return (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code >= 500
            )

        try:
            logger.info(f"Invoking MiniMax: {self._primary_model}")
            cb = CircuitBreakerRegistry.get("minimax-api", failure_threshold=3, recovery_timeout=120.0)

            @retry(
                retry=retry_if_exception(_is_5xx),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=8),
                reraise=True,
                before_sleep=lambda rs: logger.warning(
                    f"MiniMax 5xx on attempt {rs.attempt_number}, retrying in {rs.next_action.sleep:.0f}s"
                ),
            )
            def _post() -> requests.Response:
                r = cb.call(requests.post, url, headers=headers, json=body, timeout=120)
                r.raise_for_status()
                return r

            resp = _post()
            data = resp.json()

            if "error" in data:
                logger.warning(f"MiniMax API returned error: {data['error']}")
                return None, {}

            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            text = choice["message"]["content"]

            if text is None:
                logger.warning(f"MiniMax returned None content (finish_reason={finish_reason})")
                return None, {}

            # Strip <think>...</think> reasoning blocks
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            if not text:
                logger.warning(f"MiniMax returned empty content after stripping think blocks (finish_reason={finish_reason})")
                return None, {}

            usage = data.get("usage", {})
            token_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
            logger.info(f"MiniMax response received ({len(text)} chars, finish_reason={finish_reason})")
            return text, token_usage
        except CircuitOpenError as e:
            logger.warning(f"MiniMax skipped: {e}")
            return None, {}
        except (requests.RequestException, KeyError, IndexError) as e:
            logger.error(f"MiniMax API error: {e}")
            return None, {}

    def _invoke_openrouter(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ):
        """Invoke OpenRouter via OpenAI-compatible API.

        Returns:
            Tuple of (response_text, usage_dict). Both are None/empty on failure.
        """
        headers = {
            "Authorization": f"Bearer {self._fallback_api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        try:
            logger.info(f"Invoking OpenRouter: {model}")
            cb = CircuitBreakerRegistry.get("openrouter-api", failure_threshold=3, recovery_timeout=120.0)
            resp = cb.call(requests.post, self._fallback_base_url, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if text is None:
                logger.warning(f"OpenRouter returned None content for {model} (likely reasoning-only response)")
                return None, {}
            usage = data.get("usage", {})
            token_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
            logger.info(f"OpenRouter response received ({len(text)} chars)")
            return text, token_usage
        except CircuitOpenError as e:
            logger.warning(f"OpenRouter skipped: {e}")
            return None, {}
        except (requests.RequestException, KeyError, IndexError) as e:
            logger.error(f"OpenRouter API error: {e}")
            return None, {}

    @staticmethod
    def _extract_anthropic_text(data: Dict[str, Any]) -> str:
        """Extract text from Anthropic-format response."""
        content = data.get("content", [])
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(texts)
