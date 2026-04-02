#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Newsletter scanner.

Fetches classified newsletter emails from Supabase (populated by n8n
email-ingestion workflow) and returns them as briefing source items.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class NewsletterScanner:
    """Fetches undigested newsletter items from Supabase."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        max_items: int = 20,
    ):
        """
        Initialize NewsletterScanner.

        Args:
            supabase_url: Supabase project URL (e.g. https://xxx.supabase.co).
            supabase_key: Supabase scoped API key.
            max_items: Maximum items to fetch.
        """
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL", "")
        self.supabase_key = supabase_key or os.environ.get(
            "SUPABASE_API_KEY_N8N_ATLAS_MORNING_BRIEF", ""
        )
        self.max_items = max_items
        self._fetched_ids: List[str] = []

    def scan(self) -> List[Dict[str, Any]]:
        """
        Fetch undigested newsletter emails from Supabase.

        Returns:
            List of article-like dicts compatible with the blog/news format.
        """
        if not self.supabase_url or not self.supabase_key:
            logger.warning(
                "Supabase credentials not configured, skipping newsletter scan"
            )
            return []

        url = (
            f"{self.supabase_url.rstrip('/')}/rest/v1/email_classifications"
            f"?should_include_in_digest=eq.true"
            f"&digest_included_at=is.null"
            f"&select=id,subject,from_name,from_address,summary,category,created_at"
            f"&order=created_at.desc"
            f"&limit={self.max_items}"
        )
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Accept": "application/json",
        }

        try:
            logger.info("Scanning newsletters from Supabase")
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            rows = resp.json()

            if not isinstance(rows, list):
                logger.error("Unexpected response format from Supabase")
                return []

            items = []
            for row in rows:
                self._fetched_ids.append(row["id"])
                items.append({
                    "source": row.get("from_name") or row.get("from_address", "Newsletter"),
                    "title": row.get("subject", ""),
                    "link": "",
                    "summary": row.get("summary", ""),
                    "published": row.get("created_at", ""),
                    "category": row.get("category", ""),
                })

            logger.info(f"Found {len(items)} undigested newsletters")
            return items

        except requests.RequestException as e:
            logger.error(f"Failed to fetch newsletters from Supabase: {e}")
            return []

    def mark_digested(self) -> bool:
        """
        Mark fetched newsletters as digested in Supabase.

        Should be called after the briefing has been successfully generated
        and distributed.

        Returns:
            True if successful, False otherwise.
        """
        if not self._fetched_ids:
            return True

        if not self.supabase_url or not self.supabase_key:
            return False

        ids_filter = ",".join(self._fetched_ids)
        url = (
            f"{self.supabase_url.rstrip('/')}/rest/v1/email_classifications"
            f"?id=in.({ids_filter})"
        )
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        body = {
            "digest_included_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            resp = requests.patch(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            logger.info(
                f"Marked {len(self._fetched_ids)} newsletters as digested"
            )
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to mark newsletters as digested: {e}")
            return False
