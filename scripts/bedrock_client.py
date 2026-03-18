#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Amazon Bedrock model client.

Provides tiered model access to Amazon Bedrock for intelligence features.
Supports multiple models with automatic fallback.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class BedrockClient:
    """Client for Amazon Bedrock model inference with tiered model support.

    Supports model IDs, cross-region inference profile IDs (e.g. us.anthropic.*),
    and full inference profile ARNs (arn:aws:bedrock:...:inference-profile/...).
    """

    # Default model IDs for each tier.
    # Use cross-region inference profile IDs (us.*) for broad region support.
    # On-demand IDs (e.g. amazon.nova-lite-v1:0) are not available in all regions.
    DEFAULT_MODELS = {
        "heavy": "us.anthropic.claude-opus-4-6-v1",
        "medium": "us.amazon.nova-pro-v1:0",
        "light": "us.amazon.nova-lite-v1:0",
    }

    # Known provider keywords for request/response format routing.
    _PROVIDER_KEYWORDS = ["anthropic", "amazon.nova", "meta", "mistral", "cohere"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BedrockClient.

        Args:
            config: Optional Bedrock configuration from config.yaml.
                    Keys: region, models (dict of tier->model_id or inference profile ARN),
                    max_tokens, temperature, inference_profile_arn (optional override).
        """
        config = config or {}
        self.region = config.get("region", os.environ.get("AWS_REGION", "us-east-1"))
        self.enabled = config.get("enabled", True)
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.3)

        # Model IDs per tier — values can be model IDs, inference profile IDs,
        # or full inference profile ARNs.
        models_config = config.get("models", {})
        self.models = {
            "heavy": models_config.get("heavy", self.DEFAULT_MODELS["heavy"]),
            "medium": models_config.get("medium", self.DEFAULT_MODELS["medium"]),
            "light": models_config.get("light", self.DEFAULT_MODELS["light"]),
        }

        # Allow env var override for the heavy tier inference profile ARN.
        # This lets operators swap models without touching config.yaml.
        arn_override = os.environ.get("BEDROCK_INFERENCE_PROFILE_ARN")
        if arn_override:
            self.models["heavy"] = arn_override

        self.max_calls = config.get("max_calls_per_run", 20)
        self._call_count = 0
        self._client = None
        self._available = None

    @staticmethod
    def detect_provider(model_id: str) -> str:
        """Detect the model provider from a model ID, inference profile ID, or ARN.

        Works with all supported formats:
          - On-demand:  anthropic.claude-opus-4-6-v1
          - Cross-region: eu.anthropic.claude-opus-4-6-v1
          - ARN: arn:aws:bedrock:us-east-1:123456:inference-profile/us.anthropic.claude-opus-4-6-v1

        Returns:
            Provider string ("anthropic", "amazon.nova", etc.) or "generic".
        """
        lowered = model_id.lower()
        if "anthropic" in lowered:
            return "anthropic"
        if "amazon.nova" in lowered:
            return "amazon.nova"
        return "generic"

    @property
    def client(self):
        """Lazy-initialize Bedrock runtime client."""
        if self._client is None:
            if not HAS_BOTO3:
                logger.warning("boto3 not installed. Bedrock features disabled.")
                self._available = False
                return None
            try:
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region,
                )
            except NoCredentialsError:
                logger.warning("AWS credentials not found. Bedrock features disabled.")
                self._available = False
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize Bedrock client: {e}")
                self._available = False
                return None
        return self._client

    @property
    def available(self) -> bool:
        """Check if Bedrock is available and enabled."""
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False
        if not HAS_BOTO3:
            logger.warning("boto3 not installed. Bedrock features disabled.")
            self._available = False
            return False
        # Try to initialize client
        if self.client is None:
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
        Invoke a Bedrock model.

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
            logger.debug("Bedrock not available, skipping invocation")
            return None

        if self._call_count >= self.max_calls:
            logger.warning(
                f"LLM call budget exhausted ({self.max_calls} calls). "
                "Skipping invocation. Increase bedrock.max_calls_per_run to allow more."
            )
            return None
        self._call_count += 1

        model_id = self.models.get(tier, self.models["medium"])
        tokens = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        try:
            logger.info(f"Invoking Bedrock model: {model_id} (tier: {tier})")

            # Build the request based on model provider
            body = self._build_request_body(
                model_id=model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=tokens,
                temperature=temp,
            )

            response = self.client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            raw_body = response["body"].read()
            try:
                response_body = json.loads(raw_body)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Invalid JSON in Bedrock response: {e}")
                return None
            if not isinstance(response_body, dict):
                logger.error("Bedrock response body is not a JSON object")
                return None
            result = self._extract_response_text(model_id, response_body)

            logger.info(f"Bedrock response received ({len(result)} chars)")
            return result

        except Exception as e:
            # Handle AWS ClientError specifically if available
            if HAS_BOTO3 and isinstance(e, ClientError):
                error_code = e.response["Error"]["Code"]
                logger.error(f"Bedrock API error ({error_code}): {e}")
                return None
            logger.error(f"Bedrock invocation failed: {e}")
            return None

    def _build_request_body(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """
        Build request body based on model provider format.

        Args:
            model_id: The Bedrock model ID.
            prompt: User prompt.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Request body dictionary.
        """
        provider = self.detect_provider(model_id)

        if provider == "anthropic":
            # Anthropic models (Claude) use their own message format with "type" key.
            # Works for on-demand IDs, cross-region profiles, and inference profile ARNs.
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            if system_prompt:
                body["system"] = system_prompt
        elif provider == "amazon.nova":
            # Nova models do NOT accept "type" key in content blocks
            # and use "max_new_tokens" (snake_case) not "maxNewTokens"
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            body = {
                "messages": messages,
                "inferenceConfig": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                },
            }
            if system_prompt:
                body["system"] = [{"text": system_prompt}]
        else:
            # Generic format (no "type" key, snake_case params)
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            body = {
                "messages": messages,
                "inferenceConfig": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                },
            }
            if system_prompt:
                body["system"] = [{"text": system_prompt}]

        return body

    def _extract_response_text(
        self, model_id: str, response_body: Dict[str, Any]
    ) -> str:
        """
        Extract text from model response based on provider format.

        Args:
            model_id: The Bedrock model ID.
            response_body: Parsed JSON response body.

        Returns:
            Extracted text string.
        """
        provider = self.detect_provider(model_id)

        if provider == "anthropic":
            content = response_body.get("content", [])
            texts = [block.get("text", "") for block in content if block.get("type") == "text"]
            return "\n".join(texts)
        elif provider == "amazon.nova":
            output = response_body.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])
            texts = [block.get("text", "") for block in content]
            return "\n".join(texts)
        else:
            # Try common response formats
            output = response_body.get("output", {})
            if isinstance(output, dict):
                message = output.get("message", {})
                content = message.get("content", [])
                if content:
                    return "\n".join(
                        block.get("text", "") for block in content
                    )
            # Fallback
            return str(response_body)
