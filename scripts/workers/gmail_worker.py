#!/usr/bin/env python3
"""
Gmail newsletter worker for v0.2 multi-agent architecture.

Fetches unread newsletter emails via IMAP and returns them as findings.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from scripts.gmail_scanner import GmailScanner
from scripts.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class GmailWorker(BaseWorker):
    """Fetches newsletter emails from Gmail via IMAP."""

    def __init__(self, config: Dict[str, Any], ref_date: Optional[datetime] = None):
        super().__init__(config, "gmail_worker", ref_date=ref_date)
        ns_config = config.get("newsletter_source", {})
        self.enabled = ns_config.get("enabled", False)
        self.source_label = ns_config.get("source_label", "INBOX")
        self.max_items = ns_config.get("max_items", 20)
        self.mark_read = ns_config.get("mark_read", False)
        self._scanner: Optional[GmailScanner] = None

    @property
    def scanner(self) -> Optional[GmailScanner]:
        """Expose scanner so coordinator can call mark_digested() after distribution."""
        return self._scanner

    def execute(self) -> Dict[str, Any]:
        self._start_timing()

        if not self.enabled:
            return self._create_finding(
                status="success",
                items=[],
                synthesis="Gmail newsletter scanning disabled",
            )

        try:
            self._scanner = GmailScanner(
                source_label=self.source_label,
                max_items=self.max_items,
            )
            items = self._scanner.scan()

            return self._create_finding(
                status="success",
                items=items,
                items_found=len(items),
                synthesis=f"Found {len(items)} newsletter emails",
            )

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error: {e}")
            return self._create_finding(
                status="error",
                items=[],
                error=str(e),
            )
