#!/usr/bin/env python3
"""
GitHub trending worker for v0.2 multi-agent architecture.

Scrapes github.com/trending and returns daily trending repos as findings.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from scripts.github_trending_scanner import GitHubTrendingScanner
from scripts.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class GitHubTrendingWorker(BaseWorker):
    """Scrapes GitHub trending repos."""

    def __init__(self, config: Dict[str, Any], ref_date: Optional[datetime] = None):
        super().__init__(config, "github_trending_worker", ref_date=ref_date)
        gt_config = config.get("github_trending", {})
        self.enabled = gt_config.get("enabled", False)
        self.max_items = gt_config.get("max_items", 20)

    def execute(self) -> Dict[str, Any]:
        self._start_timing()

        if not self.enabled:
            return self._create_finding(
                status="success",
                items=[],
                synthesis="GitHub trending scanning disabled",
            )

        try:
            scanner = GitHubTrendingScanner(max_items=self.max_items)
            items = scanner.scan()

            return self._create_finding(
                status="success",
                items=items,
                items_found=len(items),
                synthesis=f"Found {len(items)} trending repos",
            )

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error: {e}")
            return self._create_finding(
                status="error",
                items=[],
                error=str(e),
            )
