#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
LLM client with tiered model support.

Primary: MiniMax 2.7 via Anthropic-compatible API.
Fallback: OpenRouter free models via OpenAI-compatible API.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional

import requests

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
        "heavy": "qwen/qwen3.6-plus-preview",
        "medium": "stepfun/step-3.5-flash",
        "light": "stepfun/step-3.5-flash",
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
    ) -> Optional[str]:
        """
        Invoke an LLM model. Tries primary (MiniMax), falls back to OpenRouter.

        Args:
            prompt: User prompt text.
            tier: Model tier - "heavy", "medium", or "light".
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            system_prompt: Optional system prompt.

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

        # Try primary (MiniMax)
        if self._primary_api_key:
            result = self._invoke_minimax(prompt, tokens, temp, system_prompt)
            if result is not None:
                return result
            logger.warning("MiniMax invocation failed, trying fallback")

        # Try fallback (OpenRouter)
        if self._fallback_api_key:
            model = self._fallback_models.get(tier, self._fallback_models["medium"])
            result = self._invoke_openrouter(
                prompt, model, tokens, temp, system_prompt
            )
            if result is not None:
                return result

        logger.error("All LLM providers failed")
        return None

    # Minimum token budget for MiniMax M2.7 (reasoning model needs room to think + respond)
    _MINIMAX_MIN_TOKENS = 4096

    def _invoke_minimax(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> Optional[str]:
        """Invoke MiniMax via native OpenAI-compatible API."""
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

        try:
            logger.info(f"Invoking MiniMax: {self._primary_model}")
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                logger.warning(f"MiniMax API returned error: {data['error']}")
                return None

            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            text = choice["message"]["content"]

            if text is None:
                logger.warning(f"MiniMax returned None content (finish_reason={finish_reason})")
                return None

            # Strip <think>...</think> reasoning blocks
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            if not text:
                logger.warning(f"MiniMax returned empty content after stripping think blocks (finish_reason={finish_reason})")
                return None

            logger.info(f"MiniMax response received ({len(text)} chars, finish_reason={finish_reason})")
            return text
        except (requests.RequestException, KeyError, IndexError) as e:
            logger.error(f"MiniMax API error: {e}")
            return None

    def _invoke_openrouter(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> Optional[str]:
        """Invoke OpenRouter via OpenAI-compatible API."""
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
            resp = requests.post(
                self._fallback_base_url, headers=headers, json=body, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if text is None:
                logger.warning(f"OpenRouter returned None content for {model} (likely reasoning-only response)")
                return None
            logger.info(f"OpenRouter response received ({len(text)} chars)")
            return text
        except (requests.RequestException, KeyError, IndexError) as e:
            logger.error(f"OpenRouter API error: {e}")
            return None

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
