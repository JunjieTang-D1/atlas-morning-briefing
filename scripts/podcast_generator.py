#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
NotebookLM podcast generator.

Generates a NotebookLM Audio Overview (podcast) from the daily briefing
and returns MP3 bytes for email attachment delivery.

Auth uses a Playwright storage_state.json file containing Google session cookies.
- Local dev: set NOTEBOOKLM_STORAGE_STATE_PATH to the file path
- Production (k8s): set NOTEBOOKLM_STORAGE_STATE_B64 to base64-encoded JSON,
  the generator decodes it to a temp file automatically.
"""

import asyncio
import base64
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from notebooklm import NotebookLMClient
    from notebooklm.rpc import AudioFormat, AudioLength
    HAS_NOTEBOOKLM = True
except ImportError:
    HAS_NOTEBOOKLM = False
    NotebookLMClient = None  # type: ignore[assignment,misc]
    AudioFormat = None  # type: ignore[assignment]
    AudioLength = None  # type: ignore[assignment]


class PodcastGenerator:
    """Generates NotebookLM Audio Overview podcasts from daily briefings."""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self.instructions = config.get(
            "instructions",
            (
                "Create a concise executive briefing podcast covering today's most important "
                "AI research papers, news, and emerging trends. Focus on practical implications "
                "and highlight the top 2-3 most significant developments."
            ),
        )
        self._audio_format_name = config.get("audio_format", "brief").upper()
        self._audio_length_name = config.get("audio_length", "short").upper()
        self.max_wait_seconds = float(config.get("max_wait_seconds", 1200))
        self.include_paper_urls = config.get("include_paper_urls", True)
        self._storage_state_path: Optional[str] = config.get("storage_state_path")

    @property
    def available(self) -> bool:
        """True if notebooklm-py is installed and auth is configured."""
        if not HAS_NOTEBOOKLM:
            return False
        return bool(self._resolve_storage_path())

    def _resolve_storage_path(self) -> Optional[str]:
        """Return the storage_state.json path, decoding from B64 env var if needed."""
        # Explicit config path (local dev)
        if self._storage_state_path:
            p = Path(self._storage_state_path).expanduser()
            if p.exists():
                return str(p)
            logger.warning(f"Configured storage_state_path not found: {p}")

        # Path from env var
        env_path = os.environ.get("NOTEBOOKLM_STORAGE_STATE_PATH")
        if env_path:
            p = Path(env_path).expanduser()
            if p.exists():
                return str(p)
            logger.warning(f"NOTEBOOKLM_STORAGE_STATE_PATH not found: {p}")

        # Base64-encoded JSON (k8s secret) — written to a temp file
        b64 = os.environ.get("NOTEBOOKLM_STORAGE_STATE_B64")
        if b64:
            return None  # Decoded per-run in _resolve_storage_path_or_temp

        # Default location created by `notebooklm login`
        default = Path("~/.notebooklm/storage_state.json").expanduser()
        if default.exists():
            return str(default)

        return None

    def _decode_b64_to_temp(self) -> Optional[str]:
        """Decode NOTEBOOKLM_STORAGE_STATE_B64 to a temp file. Caller must delete it."""
        b64 = os.environ.get("NOTEBOOKLM_STORAGE_STATE_B64")
        if not b64:
            return None
        try:
            json_bytes = base64.b64decode(b64)
            with tempfile.NamedTemporaryFile(
                suffix=".json", prefix="notebooklm_", delete=False
            ) as f:
                f.write(json_bytes)
                return f.name
        except Exception as e:
            logger.warning(f"Failed to decode NOTEBOOKLM_STORAGE_STATE_B64: {e}")
            return None

    def generate(
        self,
        briefing_markdown: str,
        date: datetime,
        source_urls: Optional[List[str]] = None,
    ) -> Optional[bytes]:
        """Generate Audio Overview podcast. Returns MP3 bytes or None on failure.

        This is a synchronous wrapper around the async notebooklm-py client.
        Audio generation typically takes 5–15 minutes; max_wait_seconds controls
        the timeout (default 1200s = 20 min).
        """
        if not self.enabled:
            return None
        if not HAS_NOTEBOOKLM:
            logger.warning("notebooklm-py not installed; skipping podcast generation")
            return None
        return asyncio.run(self._generate_async(briefing_markdown, date, source_urls or []))

    async def _generate_async(
        self,
        briefing_markdown: str,
        date: datetime,
        source_urls: List[str],
    ) -> Optional[bytes]:
        """Async implementation: create notebook, add sources, generate audio, download."""
        tmp_audio_path: Optional[str] = None
        tmp_storage_path: Optional[str] = None

        # Resolve storage state file
        storage_path = self._resolve_storage_path()
        if not storage_path:
            tmp_storage_path = self._decode_b64_to_temp()
            storage_path = tmp_storage_path
        if not storage_path:
            logger.warning(
                "No NotebookLM storage state found. "
                "Run `notebooklm login` or set NOTEBOOKLM_STORAGE_STATE_B64."
            )
            return None

        try:
            client = await NotebookLMClient.from_storage(storage_path)
            async with client:
                notebook_id: Optional[str] = None
                try:
                    # 1. Create daily notebook
                    nb = await client.notebooks.create(
                        f"Atlas Daily Brief {date.strftime('%Y-%m-%d')}"
                    )
                    notebook_id = nb.id
                    logger.info(f"Created NotebookLM notebook: {notebook_id}")

                    # 2. Add briefing markdown as primary text source
                    await client.sources.add_text(
                        notebook_id,
                        title=f"Atlas Briefing {date.strftime('%Y-%m-%d')}",
                        content=briefing_markdown,
                        wait=True,
                    )
                    logger.debug("Added briefing text source")

                    # 3. Add top paper URLs for richer grounding (best-effort)
                    if self.include_paper_urls:
                        for url in source_urls[:10]:
                            try:
                                await client.sources.add_url(notebook_id, url, wait=False)
                            except Exception as e:
                                logger.debug(f"Skipping URL source {url[:60]}: {e}")

                    # 4. Trigger audio generation
                    audio_format = AudioFormat[self._audio_format_name]
                    audio_length = AudioLength[self._audio_length_name]
                    status = await client.artifacts.generate_audio(
                        notebook_id,
                        instructions=self.instructions,
                        audio_format=audio_format,
                        audio_length=audio_length,
                    )
                    logger.info(
                        f"Audio generation started (task_id={status.task_id}); "
                        f"waiting up to {self.max_wait_seconds:.0f}s"
                    )

                    # 5. Poll until complete (built-in exponential backoff)
                    await client.artifacts.wait_for_completion(
                        notebook_id,
                        status.task_id,
                        timeout=self.max_wait_seconds,
                    )
                    logger.info("Audio generation complete")

                    # 6. Download to temp file, read bytes
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp3", prefix="atlas_podcast_", delete=False
                    ) as f:
                        tmp_audio_path = f.name
                    await client.artifacts.download_audio(notebook_id, tmp_audio_path)
                    with open(tmp_audio_path, "rb") as f:
                        audio_bytes = f.read()
                    logger.info(
                        f"Downloaded podcast audio: {len(audio_bytes) / 1024:.0f} KB"
                    )
                    return audio_bytes

                except TimeoutError:
                    logger.warning(
                        f"NotebookLM audio generation timed out after {self.max_wait_seconds:.0f}s"
                    )
                    return None
                except Exception as e:
                    logger.warning(f"NotebookLM podcast generation failed: {e}")
                    return None
                finally:
                    # Always clean up the notebook to avoid accumulation
                    if notebook_id:
                        try:
                            await client.notebooks.delete(notebook_id)
                            logger.debug(f"Deleted notebook: {notebook_id}")
                        except Exception:
                            pass

        except Exception as e:
            logger.warning(f"NotebookLM client init failed: {e}")
            return None
        finally:
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)
            if tmp_storage_path and os.path.exists(tmp_storage_path):
                os.unlink(tmp_storage_path)
