#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
NotebookLM podcast generator.

Creates a shared NotebookLM notebook from the daily briefing and returns a
public share URL for embedding in the briefing markdown. Audio generation is
kicked off asynchronously — the link is valid immediately; audio is ready
~10 minutes later on Google's side.

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
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RETRY_BACKOFFS = [15, 45]  # seconds between attempts 1→2 and 2→3


class _RetryableError(Exception):
    """Transient NotebookLM failure — safe to retry after backoff."""


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
    """Generates a shared NotebookLM notebook from the daily briefing.

    Returns a public share URL that is embedded in the briefing markdown.
    Audio generation is fire-and-forget — the notebook link is valid immediately;
    the audio overview becomes playable ~10 minutes after the briefing runs.
    Notebooks are kept alive (not deleted) so the link remains accessible.
    """

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
        return bool(self._resolve_storage_path()) or bool(
            os.environ.get("NOTEBOOKLM_STORAGE_STATE_B64")
        )

    def _resolve_storage_path(self) -> Optional[str]:
        """Return the storage_state.json path, or None if only B64/missing."""
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

        # Base64-encoded JSON (k8s secret) — decoded per-run in generate()
        if os.environ.get("NOTEBOOKLM_STORAGE_STATE_B64"):
            return None

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
    ) -> Optional[str]:
        """Create a shared NotebookLM notebook and return its public URL.

        Returns the share URL (str) on success, or None if disabled/auth missing/error.
        Audio generation is kicked off asynchronously — the link is valid immediately;
        the audio overview is ready ~10 minutes later on Google's side.
        """
        if not self.enabled:
            return None
        if not HAS_NOTEBOOKLM:
            logger.warning("notebooklm-py not installed; skipping podcast generation")
            return None

        max_attempts = len(_RETRY_BACKOFFS) + 1
        for attempt in range(max_attempts):
            try:
                result = asyncio.run(
                    self._generate_async(briefing_markdown, date, source_urls or [])
                )
                if result is not None:
                    return result
                # _generate_async returned None for a permanent/known failure;
                # no point retrying.
                return None
            except _RetryableError as exc:
                if attempt < max_attempts - 1:
                    wait = _RETRY_BACKOFFS[attempt]
                    logger.warning(
                        "NotebookLM transient error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, max_attempts, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        "NotebookLM failed after %d attempts: %s — skipping podcast",
                        max_attempts, exc,
                    )
        return None

    @staticmethod
    def _is_geo_blocked(exc: BaseException) -> bool:
        """Return True if the error is a permanent geo-restriction (not retryable)."""
        msg = str(exc).lower()
        return "location=unsupported" in msg or "location unsupported" in msg

    @staticmethod
    def _is_auth_error(exc: BaseException) -> bool:
        """Return True if the exception looks like an expired/invalid session."""
        msg = str(exc).lower()
        auth_signals = (
            "401", "403", "unauthorized", "forbidden",
            "unauthenticated", "not authenticated",
            "session", "login", "expired", "invalid credentials",
            "access denied", "sign in", "google account",
        )
        # Also check HTTP status codes surfaced by httpx / aiohttp / requests
        for attr in ("status_code", "status", "response"):
            val = getattr(exc, attr, None)
            if val is not None:
                code = getattr(val, "status_code", None) or (val if isinstance(val, int) else None)
                if code in (401, 403):
                    return True
        return any(sig in msg for sig in auth_signals)

    async def _generate_async(
        self,
        briefing_markdown: str,
        date: datetime,
        source_urls: List[str],
    ) -> Optional[str]:
        """Async core: create notebook, share it, add sources, kick off audio."""
        tmp_storage_path: Optional[str] = None

        # Resolve auth
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
                        f"Personal Daily Brief {date.strftime('%Y-%m-%d')}"
                    )
                    notebook_id = nb.id
                    logger.info(f"Created NotebookLM notebook: {notebook_id}")

                    # 2. Make public — share_url is now fixed and embeddable
                    share_status = await client.sharing.set_public(notebook_id, True)
                    share_url = getattr(share_status, "share_url", None)
                    if not share_url:
                        # Fallback: construct URL from notebook ID
                        share_url = f"https://notebooklm.google.com/notebook/{notebook_id}"
                    logger.info(f"NotebookLM share URL: {share_url}")

                    # 3. Add briefing markdown as primary text source
                    await client.sources.add_text(
                        notebook_id,
                        title=f"Personal Briefing {date.strftime('%Y-%m-%d')}",
                        content=briefing_markdown,
                        wait=True,
                    )
                    logger.debug("Added briefing text source")

                    # 4. Add top paper URLs for richer grounding (best-effort)
                    if self.include_paper_urls:
                        for url in source_urls[:10]:
                            try:
                                await client.sources.add_url(notebook_id, url, wait=False)
                            except Exception as e:
                                logger.debug(f"Skipping URL source {url[:60]}: {e}")

                    # 5. Kick off audio generation — fire-and-forget
                    #    Audio is ready ~10 min later; we return the link now.
                    try:
                        audio_format = AudioFormat[self._audio_format_name]
                        audio_length = AudioLength[self._audio_length_name]
                        await client.artifacts.generate_audio(
                            notebook_id,
                            instructions=self.instructions,
                            audio_format=audio_format,
                            audio_length=audio_length,
                        )
                        logger.info("Audio generation started (ready in ~10 min)")
                    except Exception as e:
                        logger.warning(f"Audio generation start failed (link still valid): {e}")

                    return share_url

                except Exception as e:
                    is_permanent = self._is_auth_error(e) or self._is_geo_blocked(e)
                    if self._is_auth_error(e):
                        logger.error(
                            "NotebookLM auth error — Google session has expired or been revoked. "
                            "HTTP %s. "
                            "Re-run `notebooklm login`, base64-encode the new storage_state.json, "
                            "and update NOTEBOOKLM_STORAGE_STATE_B64 in Infisical "
                            "(/providers/ai/google). Podcast skipped for today.",
                            getattr(getattr(e, "response", None), "status_code", "4xx"),
                        )
                    elif self._is_geo_blocked(e):
                        logger.warning(
                            "NotebookLM geo-blocked inside notebook op — skipping podcast."
                        )
                    else:
                        logger.warning(f"NotebookLM podcast setup failed: {e}")
                    # Clean up partial notebook
                    if notebook_id:
                        try:
                            await client.notebooks.delete(notebook_id)
                            logger.debug(f"Deleted notebook after failed setup: {notebook_id}")
                        except Exception:
                            pass
                    if is_permanent:
                        return None
                    raise _RetryableError(e) from e

        except Exception as e:
            if self._is_auth_error(e):
                logger.error(
                    "NotebookLM auth error — Google session has expired or been revoked. "
                    "HTTP %s. "
                    "Re-run `notebooklm login`, base64-encode the new storage_state.json, "
                    "and update NOTEBOOKLM_STORAGE_STATE_B64 in Infisical "
                    "(/providers/ai/google). Podcast skipped for today.",
                    getattr(getattr(e, "response", None), "status_code", "4xx"),
                )
                return None  # permanent — don't retry
            elif self._is_geo_blocked(e):
                logger.warning(
                    "NotebookLM geo-blocked (location=unsupported) — "
                    "the cluster region is not supported. Podcast skipped."
                )
                return None  # permanent — don't retry
            else:
                logger.warning(f"NotebookLM client init failed: {e}")
                raise _RetryableError(e) from e  # transient — let caller retry
        finally:
            if tmp_storage_path and os.path.exists(tmp_storage_path):
                os.unlink(tmp_storage_path)
