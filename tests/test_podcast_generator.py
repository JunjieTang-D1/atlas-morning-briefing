# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for podcast_generator module."""

import base64
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.podcast_generator import PodcastGenerator, _RETRY_BACKOFFS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_disabled():
    return {"enabled": False}


@pytest.fixture
def config_enabled():
    return {
        "enabled": True,
        "audio_format": "brief",
        "audio_length": "short",
        "max_wait_seconds": 600,
        "include_paper_urls": True,
        "instructions": "Test instructions",
    }


@pytest.fixture
def mock_storage_state(tmp_path):
    """Write a minimal storage_state.json to a temp file."""
    storage = tmp_path / "storage_state.json"
    storage.write_text('{"cookies": [], "origins": []}')
    return str(storage)


@pytest.fixture
def generator_with_storage(config_enabled, mock_storage_state):
    cfg = {**config_enabled, "storage_state_path": mock_storage_state}
    return PodcastGenerator(cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_SHARE_URL = "https://notebooklm.google.com/notebook/test-nb-123"


def _make_mock_client():
    """Build a fully mocked NotebookLMClient."""
    mock_notebook = MagicMock()
    mock_notebook.id = "nb-test-123"

    mock_source = MagicMock()
    mock_source.id = "src-001"

    mock_share_status = MagicMock()
    mock_share_status.share_url = FAKE_SHARE_URL

    mock_notebooks = MagicMock()
    mock_notebooks.create = AsyncMock(return_value=mock_notebook)
    mock_notebooks.delete = AsyncMock(return_value=True)

    mock_sources = MagicMock()
    mock_sources.add_text = AsyncMock(return_value=mock_source)
    mock_sources.add_url = AsyncMock(return_value=mock_source)

    mock_sharing = MagicMock()
    mock_sharing.set_public = AsyncMock(return_value=mock_share_status)

    mock_artifacts = MagicMock()
    mock_artifacts.generate_audio = AsyncMock(return_value=MagicMock(task_id="task-abc"))
    mock_artifacts.wait_for_completion = AsyncMock()  # should NOT be called

    client = MagicMock()
    client.notebooks = mock_notebooks
    client.sources = mock_sources
    client.sharing = mock_sharing
    client.artifacts = mock_artifacts
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ---------------------------------------------------------------------------
# TestPodcastGeneratorInit
# ---------------------------------------------------------------------------

class TestPodcastGeneratorInit:
    def test_disabled_by_default(self):
        gen = PodcastGenerator({})
        assert gen.enabled is False

    def test_enabled_flag(self, config_enabled):
        gen = PodcastGenerator(config_enabled)
        assert gen.enabled is True

    def test_defaults(self):
        gen = PodcastGenerator({"enabled": True})
        assert gen._audio_format_name == "BRIEF"
        assert gen._audio_length_name == "SHORT"
        assert gen.max_wait_seconds == 1200.0
        assert gen.include_paper_urls is True

    def test_custom_config(self):
        gen = PodcastGenerator({
            "enabled": True,
            "audio_format": "deep-dive",
            "audio_length": "long",
            "max_wait_seconds": 300,
            "include_paper_urls": False,
        })
        assert gen._audio_format_name == "DEEP_DIVE"
        assert gen._audio_length_name == "LONG"
        assert gen.max_wait_seconds == 300.0
        assert gen.include_paper_urls is False


# ---------------------------------------------------------------------------
# TestStoragePathResolution
# ---------------------------------------------------------------------------

class TestStoragePathResolution:
    def test_explicit_path_found(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text("{}")
        gen = PodcastGenerator({"enabled": True, "storage_state_path": str(p)})
        assert gen._resolve_storage_path() == str(p)

    def test_explicit_path_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("NOTEBOOKLM_STORAGE_STATE_PATH", raising=False)
        monkeypatch.delenv("NOTEBOOKLM_STORAGE_STATE_B64", raising=False)
        gen = PodcastGenerator({
            "enabled": True,
            "storage_state_path": str(tmp_path / "nonexistent.json"),
        })
        assert gen._resolve_storage_path() is None

    def test_env_path(self, tmp_path, monkeypatch):
        p = tmp_path / "state.json"
        p.write_text("{}")
        monkeypatch.setenv("NOTEBOOKLM_STORAGE_STATE_PATH", str(p))
        gen = PodcastGenerator({"enabled": True})
        assert gen._resolve_storage_path() == str(p)

    def test_b64_env_returns_none_from_resolve(self, monkeypatch):
        monkeypatch.setenv("NOTEBOOKLM_STORAGE_STATE_B64", "e30=")  # base64 of '{}'
        monkeypatch.delenv("NOTEBOOKLM_STORAGE_STATE_PATH", raising=False)
        gen = PodcastGenerator({"enabled": True})
        # _resolve_storage_path() returns None when only B64 is set (decoded per-run)
        assert gen._resolve_storage_path() is None

    def test_decode_b64_to_temp(self, monkeypatch):
        content = b'{"cookies": []}'
        monkeypatch.setenv("NOTEBOOKLM_STORAGE_STATE_B64", base64.b64encode(content).decode())
        gen = PodcastGenerator({"enabled": True})
        tmp = gen._decode_b64_to_temp()
        assert tmp is not None
        assert os.path.exists(tmp)
        with open(tmp, "rb") as f:
            assert f.read() == content
        os.unlink(tmp)

    def test_decode_b64_invalid(self, monkeypatch):
        monkeypatch.setenv("NOTEBOOKLM_STORAGE_STATE_B64", "!!!invalid!!!")
        gen = PodcastGenerator({"enabled": True})
        assert gen._decode_b64_to_temp() is None


# ---------------------------------------------------------------------------
# TestGenerateDisabled
# ---------------------------------------------------------------------------

class TestGenerateDisabled:
    def test_returns_none_when_disabled(self, config_disabled):
        gen = PodcastGenerator(config_disabled)
        result = gen.generate("some content", datetime.now())
        assert result is None

    def test_returns_none_without_notebooklm(self, config_enabled, mock_storage_state, monkeypatch):
        monkeypatch.setattr("scripts.podcast_generator.HAS_NOTEBOOKLM", False)
        gen = PodcastGenerator({**config_enabled, "storage_state_path": mock_storage_state})
        result = gen.generate("content", datetime.now())
        assert result is None

    def test_returns_none_no_auth_configured(self, config_enabled, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))  # prevent ~/.notebooklm/storage_state.json fallback
        monkeypatch.delenv("NOTEBOOKLM_STORAGE_STATE_PATH", raising=False)
        monkeypatch.delenv("NOTEBOOKLM_STORAGE_STATE_B64", raising=False)
        gen = PodcastGenerator(config_enabled)  # no storage_state_path in config
        result = gen.generate("content", datetime.now())
        assert result is None


# ---------------------------------------------------------------------------
# TestGenerateHappyPath
# ---------------------------------------------------------------------------

class TestGenerateHappyPath:
    def test_returns_share_url(self, generator_with_storage):
        mock_client = _make_mock_client()
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            result = generator_with_storage.generate("# Briefing", datetime(2026, 4, 6))
        assert result == FAKE_SHARE_URL

    def test_sharing_set_public_called(self, generator_with_storage):
        mock_client = _make_mock_client()
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            generator_with_storage.generate("content", datetime(2026, 4, 6))
        mock_client.sharing.set_public.assert_awaited_once_with("nb-test-123", True)

    def test_does_not_delete_notebook(self, generator_with_storage):
        mock_client = _make_mock_client()
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            generator_with_storage.generate("content", datetime(2026, 4, 6))
        mock_client.notebooks.delete.assert_not_awaited()

    def test_generate_audio_called_not_waited(self, generator_with_storage):
        mock_client = _make_mock_client()
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            generator_with_storage.generate("content", datetime(2026, 4, 6))
        mock_client.artifacts.generate_audio.assert_awaited_once()
        mock_client.artifacts.wait_for_completion.assert_not_awaited()

    def test_adds_text_source(self, generator_with_storage):
        mock_client = _make_mock_client()
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            generator_with_storage.generate("# My briefing", datetime(2026, 4, 6))
        mock_client.sources.add_text.assert_awaited_once()
        call_kwargs = mock_client.sources.add_text.call_args
        content_val = call_kwargs.kwargs.get("content", "") or (
            call_kwargs.args[2] if len(call_kwargs.args) > 2 else ""
        )
        assert "# My briefing" in content_val

    def test_adds_paper_urls(self, generator_with_storage):
        mock_client = _make_mock_client()
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            generator_with_storage.generate(
                "content",
                datetime(2026, 4, 6),
                source_urls=["https://arxiv.org/abs/1234.5678"],
            )
        mock_client.sources.add_url.assert_awaited()

    def test_skips_paper_urls_when_disabled(self, mock_storage_state):
        config = {
            "enabled": True,
            "include_paper_urls": False,
            "storage_state_path": mock_storage_state,
        }
        gen = PodcastGenerator(config)
        mock_client = _make_mock_client()
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            gen.generate("content", datetime(2026, 4, 6), source_urls=["https://example.com"])
        mock_client.sources.add_url.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestGenerateFailures
# ---------------------------------------------------------------------------

class TestGenerateFailures:
    def test_notebook_creation_failure_returns_none(self, generator_with_storage):
        mock_client = _make_mock_client()
        mock_client.notebooks.create = AsyncMock(side_effect=Exception("create failed"))
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient, \
             patch("scripts.podcast_generator.time.sleep"):
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            result = generator_with_storage.generate("content", datetime(2026, 4, 6))
        assert result is None

    def test_set_public_failure_returns_none_and_deletes_notebook(self, generator_with_storage):
        mock_client = _make_mock_client()
        mock_client.sharing.set_public = AsyncMock(side_effect=Exception("share failed"))
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient, \
             patch("scripts.podcast_generator.time.sleep"):
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            result = generator_with_storage.generate("content", datetime(2026, 4, 6))
        assert result is None
        # Retry loop makes 3 attempts — delete called once per attempt
        assert mock_client.notebooks.delete.await_count == len(_RETRY_BACKOFFS) + 1
        mock_client.notebooks.delete.assert_awaited_with("nb-test-123")

    def test_audio_start_failure_still_returns_url(self, generator_with_storage):
        mock_client = _make_mock_client()
        mock_client.artifacts.generate_audio = AsyncMock(side_effect=Exception("API error"))
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            result = generator_with_storage.generate("content", datetime(2026, 4, 6))
        # URL is still returned even when audio generation start fails
        assert result == FAKE_SHARE_URL
        # Notebook is NOT deleted
        mock_client.notebooks.delete.assert_not_awaited()

    def test_url_source_failure_does_not_abort(self, generator_with_storage):
        mock_client = _make_mock_client()
        mock_client.sources.add_url = AsyncMock(side_effect=Exception("URL error"))
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            result = generator_with_storage.generate(
                "content", datetime(2026, 4, 6), source_urls=["https://bad-url.com"]
            )
        assert result == FAKE_SHARE_URL

    def test_client_init_error_returns_none(self, generator_with_storage):
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient, \
             patch("scripts.podcast_generator.time.sleep"):
            MockClient.from_storage = AsyncMock(side_effect=Exception("network error"))
            result = generator_with_storage.generate("content", datetime(2026, 4, 6))
        assert result is None
