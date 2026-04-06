# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for podcast_generator module."""

import asyncio
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.podcast_generator import PodcastGenerator


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
        assert gen._audio_format_name == "DEEP-DIVE"
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

    def test_explicit_path_missing_returns_none(self, tmp_path):
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
        import base64
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

    def test_returns_none_no_auth_configured(self, config_enabled, monkeypatch):
        monkeypatch.delenv("NOTEBOOKLM_STORAGE_STATE_PATH", raising=False)
        monkeypatch.delenv("NOTEBOOKLM_STORAGE_STATE_B64", raising=False)
        gen = PodcastGenerator(config_enabled)  # no storage_state_path in config
        result = gen.generate("content", datetime.now())
        assert result is None


# ---------------------------------------------------------------------------
# TestGenerateAsync (mocked NotebookLMClient)
# ---------------------------------------------------------------------------

def _make_mock_client():
    """Build a fully mocked NotebookLMClient."""
    mock_notebook = MagicMock()
    mock_notebook.id = "nb-test-123"

    mock_source = MagicMock()
    mock_source.id = "src-001"

    mock_status = MagicMock()
    mock_status.task_id = "task-abc"

    mock_notebooks = MagicMock()
    mock_notebooks.create = AsyncMock(return_value=mock_notebook)
    mock_notebooks.delete = AsyncMock(return_value=True)

    mock_sources = MagicMock()
    mock_sources.add_text = AsyncMock(return_value=mock_source)
    mock_sources.add_url = AsyncMock(return_value=mock_source)

    mock_artifacts = MagicMock()
    mock_artifacts.generate_audio = AsyncMock(return_value=mock_status)
    mock_artifacts.wait_for_completion = AsyncMock(return_value=mock_status)
    mock_artifacts.download_audio = AsyncMock(return_value="/tmp/fake.mp3")

    client = MagicMock()
    client.notebooks = mock_notebooks
    client.sources = mock_sources
    client.artifacts = mock_artifacts
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestGenerateAsyncHappyPath:
    def test_returns_mp3_bytes(self, generator_with_storage, tmp_path):
        fake_mp3 = b"ID3FAKEMP3DATA"
        mock_client = _make_mock_client()

        # Patch download_audio to write to the expected temp file path
        async def fake_download(notebook_id, path):
            with open(path, "wb") as f:
                f.write(fake_mp3)
            return path

        mock_client.artifacts.download_audio = fake_download

        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            result = generator_with_storage.generate(
                "# Briefing content", datetime(2026, 4, 6)
            )

        assert result == fake_mp3

    def test_creates_and_deletes_notebook(self, generator_with_storage):
        fake_mp3 = b"MP3DATA"
        mock_client = _make_mock_client()

        async def fake_download(notebook_id, path):
            with open(path, "wb") as f:
                f.write(fake_mp3)
            return path

        mock_client.artifacts.download_audio = fake_download

        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            generator_with_storage.generate("content", datetime(2026, 4, 6))

        mock_client.notebooks.create.assert_awaited_once()
        mock_client.notebooks.delete.assert_awaited_once_with("nb-test-123")

    def test_adds_text_source(self, generator_with_storage):
        fake_mp3 = b"MP3"
        mock_client = _make_mock_client()

        async def fake_download(notebook_id, path):
            with open(path, "wb") as f:
                f.write(fake_mp3)
            return path

        mock_client.artifacts.download_audio = fake_download

        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            generator_with_storage.generate("# My briefing", datetime(2026, 4, 6))

        mock_client.sources.add_text.assert_awaited_once()
        call_kwargs = mock_client.sources.add_text.call_args
        assert "# My briefing" in call_kwargs.kwargs.get("content", "") or \
               "# My briefing" in (call_kwargs.args[2] if len(call_kwargs.args) > 2 else "")

    def test_adds_paper_urls(self, generator_with_storage):
        fake_mp3 = b"MP3"
        mock_client = _make_mock_client()

        async def fake_download(notebook_id, path):
            with open(path, "wb") as f:
                f.write(fake_mp3)
            return path

        mock_client.artifacts.download_audio = fake_download

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
        fake_mp3 = b"MP3"
        mock_client = _make_mock_client()

        async def fake_download(notebook_id, path):
            with open(path, "wb") as f:
                f.write(fake_mp3)
            return path

        mock_client.artifacts.download_audio = fake_download

        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            gen.generate("content", datetime(2026, 4, 6), source_urls=["https://example.com"])

        mock_client.sources.add_url.assert_not_awaited()


class TestGenerateAsyncFailures:
    def test_timeout_returns_none(self, generator_with_storage):
        mock_client = _make_mock_client()
        mock_client.artifacts.wait_for_completion = AsyncMock(side_effect=TimeoutError("timed out"))

        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            result = generator_with_storage.generate("content", datetime(2026, 4, 6))

        assert result is None

    def test_generation_error_returns_none(self, generator_with_storage):
        mock_client = _make_mock_client()
        mock_client.artifacts.generate_audio = AsyncMock(
            side_effect=Exception("API error")
        )

        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            result = generator_with_storage.generate("content", datetime(2026, 4, 6))

        assert result is None

    def test_notebook_deleted_on_failure(self, generator_with_storage):
        mock_client = _make_mock_client()
        mock_client.artifacts.generate_audio = AsyncMock(
            side_effect=Exception("API error")
        )

        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            generator_with_storage.generate("content", datetime(2026, 4, 6))

        # Notebook should be cleaned up even when generation fails
        mock_client.notebooks.delete.assert_awaited_once_with("nb-test-123")

    def test_client_init_error_returns_none(self, generator_with_storage):
        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(side_effect=Exception("auth failed"))
            result = generator_with_storage.generate("content", datetime(2026, 4, 6))

        assert result is None

    def test_url_source_failure_does_not_abort(self, generator_with_storage):
        """A failing URL source should not abort audio generation."""
        fake_mp3 = b"MP3DATA"
        mock_client = _make_mock_client()
        mock_client.sources.add_url = AsyncMock(side_effect=Exception("URL error"))

        async def fake_download(notebook_id, path):
            with open(path, "wb") as f:
                f.write(fake_mp3)
            return path

        mock_client.artifacts.download_audio = fake_download

        with patch("scripts.podcast_generator.NotebookLMClient") as MockClient:
            MockClient.from_storage = AsyncMock(return_value=mock_client)
            result = generator_with_storage.generate(
                "content", datetime(2026, 4, 6), source_urls=["https://bad-url.com"]
            )

        # Audio generation should still succeed despite URL source failure
        assert result == fake_mp3
