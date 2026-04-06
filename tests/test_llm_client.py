# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for LLM client module."""

from unittest.mock import patch, MagicMock

import requests as req_lib
from scripts.llm_client import LLMClient


class TestLLMClientInit:
    def test_default_config(self):
        client = LLMClient()
        assert client.enabled is True
        assert client.max_tokens == 2048
        assert client.temperature == 0.3
        assert client.max_calls == 20
        assert client._primary_model == "MiniMax-M2.7"

    def test_custom_config(self):
        config = {
            "enabled": False,
            "max_tokens": 1024,
            "temperature": 0.5,
            "max_calls_per_run": 10,
            "primary": {
                "base_url": "https://custom.api/anthropic",
                "model": "custom-model",
            },
            "fallback": {
                "models": {"heavy": "test/heavy", "medium": "test/med", "light": "test/light"},
            },
        }
        client = LLMClient(config)
        assert client.enabled is False
        assert client.max_tokens == 1024
        assert client._primary_model == "custom-model"
        assert client._fallback_models["heavy"] == "test/heavy"

    def test_disabled_not_available(self):
        client = LLMClient({"enabled": False})
        assert client.available is False


class TestLLMClientAvailability:
    def test_no_keys_not_available(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient()
            # Clear any cached keys
            client._primary_api_key = ""
            client._fallback_api_key = ""
            client._available = None
            assert client.available is False

    def test_minimax_key_available(self):
        client = LLMClient()
        client._primary_api_key = "test-key"
        client._fallback_api_key = ""
        client._available = None
        assert client.available is True

    def test_openrouter_key_only_available(self):
        client = LLMClient()
        client._primary_api_key = ""
        client._fallback_api_key = "test-key"
        client._available = None
        assert client.available is True


class TestLLMClientBudget:
    def test_budget_enforcement(self):
        client = LLMClient({"max_calls_per_run": 2})
        client._primary_api_key = "test"
        client._available = True
        client._call_count = 2
        result = client.invoke("test prompt")
        assert result is None


class TestLLMClientInvoke:
    @patch("scripts.llm_client.requests.post")
    def test_minimax_invoke_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello from MiniMax"}, "finish_reason": "stop"}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = LLMClient()
        client._primary_api_key = "test-key"
        client._available = True

        result = client.invoke("test prompt", tier="heavy")
        assert result == "Hello from MiniMax"
        mock_post.assert_called_once()

        # Verify request format (native OpenAI-compatible endpoint)
        call_args = mock_post.call_args
        assert "api.minimax.io" in call_args[0][0]
        assert "/v1/chat/completions" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["model"] == "MiniMax-M2.7"
        assert body["messages"][-1]["content"] == "test prompt"

    @patch("scripts.llm_client.requests.post")
    def test_minimax_failure_falls_back_to_openrouter(self, mock_post):
        # First call (MiniMax) fails, second call (OpenRouter) succeeds
        minimax_resp = MagicMock()
        minimax_resp.raise_for_status.side_effect = req_lib.HTTPError("MiniMax down")

        openrouter_resp = MagicMock()
        openrouter_resp.status_code = 200
        openrouter_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenRouter"}}],
        }
        openrouter_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [minimax_resp, openrouter_resp]

        client = LLMClient()
        client._primary_api_key = "minimax-key"
        client._fallback_api_key = "openrouter-key"
        client._available = True

        result = client.invoke("test prompt", tier="light")
        assert result == "Hello from OpenRouter"
        assert mock_post.call_count == 2

    @patch("scripts.llm_client.requests.post")
    def test_system_prompt_passed(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = LLMClient()
        client._primary_api_key = "test-key"
        client._available = True

        client.invoke("prompt", system_prompt="You are helpful")
        body = mock_post.call_args[1]["json"]
        # System prompt is passed as first message in OpenAI format
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are helpful"


class TestExtractAnthropicText:
    def test_single_text_block(self):
        data = {"content": [{"type": "text", "text": "Hello"}]}
        assert LLMClient._extract_anthropic_text(data) == "Hello"

    def test_multiple_text_blocks(self):
        data = {"content": [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
        ]}
        assert LLMClient._extract_anthropic_text(data) == "Line 1\nLine 2"

    def test_empty_content(self):
        data = {"content": []}
        assert LLMClient._extract_anthropic_text(data) == ""

    def test_missing_content(self):
        data = {}
        assert LLMClient._extract_anthropic_text(data) == ""
