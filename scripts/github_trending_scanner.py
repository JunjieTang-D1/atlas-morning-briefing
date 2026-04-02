#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
GitHub trending scanner.

Fetches trending GitHub repos from Supabase (populated by n8n
github-trending-digest workflow) and returns them as briefing source items.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class GitHubTrendingScanner:
    """Fetches trending GitHub repos from Supabase memory_entries table."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        max_items: int = 20,
    ):
        """
        Initialize GitHubTrendingScanner.

        Args:
            supabase_url: Supabase project URL.
            supabase_key: Supabase scoped API key.
            max_items: Maximum items to fetch.
        """
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL", "")
        self.supabase_key = supabase_key or os.environ.get(
            "SUPABASE_API_KEY_N8N_ATLAS_MORNING_BRIEF", ""
        )
        self.max_items = max_items

    def scan(self) -> List[Dict[str, Any]]:
        """
        Fetch recent trending repos from Supabase memory_entries.

        Returns:
            List of article-like dicts compatible with the blog/news format.
        """
        if not self.supabase_url or not self.supabase_key:
            logger.warning(
                "Supabase credentials not configured, skipping GitHub trending scan"
            )
            return []

        url = (
            f"{self.supabase_url.rstrip('/')}/rest/v1/memory_entries"
            f"?source=eq.github_trending"
            f"&select=title,summary,url,metadata,created_at"
            f"&order=created_at.desc"
            f"&limit={self.max_items}"
        )
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Accept": "application/json",
        }

        try:
            logger.info("Scanning GitHub trending from Supabase")
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            rows = resp.json()

            if not isinstance(rows, list):
                logger.error("Unexpected response format from Supabase")
                return []

            items = []
            for row in rows:
                metadata = row.get("metadata") or {}
                stars = metadata.get("stars", "")
                language = metadata.get("language", "")
                star_tag = f" ({stars} stars)" if stars else ""
                lang_tag = f" [{language}]" if language else ""

                items.append({
                    "source": "GitHub Trending",
                    "title": row.get("title", ""),
                    "link": row.get("url", ""),
                    "summary": row.get("summary", ""),
                    "published": row.get("created_at", ""),
                    "stars": stars,
                    "language": language,
                })

            logger.info(f"Found {len(items)} trending repos")
            return items

        except requests.RequestException as e:
            logger.error(f"Failed to fetch GitHub trending from Supabase: {e}")
            return []
