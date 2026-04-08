#!/usr/bin/env python3
"""
Blogs worker for v0.2 multi-agent architecture.

PURPOSE: Fetch blog posts from RSS feeds, enrich with LLM ranking and summarization,
and return structured findings.

Self-contained worker that does NOT delegate back to coordinator.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.blog_scanner import BlogScanner
from scripts.llm_client import LLMClient
from scripts.intelligence import BriefingIntelligence
from scripts.workers.base_worker import BaseWorker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s)")
logger = logging.getLogger(__name__)


class BlogsWorker(BaseWorker):
    """Fetches and enriches blog posts independently."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BlogsWorker.

        Args:
            config: Full configuration dictionary
        """
        super().__init__(config, "blogs_worker")
        self.feeds = config.get("blog_feeds", [])
        self.days_back = 7
        self.max_blogs = config.get("max_blogs", 12)

    def execute(self) -> Dict[str, Any]:
        """
        Execute blogs workflow: fetch + enrich + rank + filter.

        Returns:
            Finding dict with enriched blogs
        """
        self._start_timing()
        token_count = 0

        try:
            logger.info(f"[{self.worker_name}] Starting blog scan")

            # Step 1: Fetch blogs from RSS feeds
            scanner = BlogScanner(
                feeds=self.feeds,
                days_back=self.days_back,
                max_items=10
            )
            blogs = scanner.scan_all_feeds()
            items_found = len(blogs)
            logger.info(f"[{self.worker_name}] Fetched {items_found} blog posts")

            if not blogs:
                return self._create_finding(
                    status="success",
                    items=[],
                    synthesis="No blog posts found in configured feeds",
                    items_found=0
                )

            # Step 2: Initialize intelligence layer for enrichment
            llm = LLMClient(self.config.get("llm", {}))
            intelligence = BriefingIntelligence(llm, self.config)

            if not intelligence.available:
                logger.warning(f"[{self.worker_name}] Intelligence layer unavailable, returning raw blogs")
                return self._create_finding(
                    status="success",
                    items=blogs[:self.max_blogs],
                    synthesis=f"Found {len(blogs)} blog posts. LLM enrichment unavailable.",
                    items_found=items_found
                )

            # Step 3: Rank and summarize blogs with LLM
            logger.info(f"[{self.worker_name}] Enriching blogs with LLM")
            topics = self.config.get("arxiv_topics", [])
            blogs = intelligence.rank_and_summarize_blogs(blogs, topics)

            # Estimate token usage: ~400 tokens per blog for ranking/summarization
            token_count = len(blogs) * 400

            # Step 4: Filter to top blogs
            blogs = [b for b in blogs if b.get("llm_score", 0) >= 3]
            blogs = sorted(blogs, key=lambda b: b.get("llm_score", 0), reverse=True)[:self.max_blogs]

            # Step 5: Generate synthesis
            synthesis = self._generate_synthesis(blogs)

            logger.info(f"[{self.worker_name}] Completed. {len(blogs)} blogs enriched.")

            return self._create_finding(
                status="success",
                items=blogs,
                synthesis=synthesis,
                token_count=token_count,
                items_found=items_found
            )

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error: {e}")
            return self._create_finding(
                status="error",
                items=[],
                error=str(e)
            )

    def _generate_synthesis(self, blogs: list) -> str:
        """
        Generate synthesis summary from blogs.

        Args:
            blogs: Enriched blog posts

        Returns:
            Summary string
        """
        if not blogs:
            return "No high-scoring blogs found"

        avg_score = sum(b.get("llm_score", 0) for b in blogs) / len(blogs)
        sources = set(b.get("source", "Unknown") for b in blogs)

        synthesis = f"Found {len(blogs)} high-quality blog posts (avg score: {avg_score:.1f}/5). "
        synthesis += f"Sources: {', '.join(list(sources)[:3])}. "
        if blogs:
            synthesis += f"Top post: '{blogs[0].get('title', 'Unknown')}' from {blogs[0].get('source', 'Unknown')}"

        return synthesis
