#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Blog feed scanner.

Scans RSS feeds for new blog articles.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser

from scripts.utils import load_config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class BlogScanner:
    """Scans RSS feeds for new blog posts."""

    def __init__(
        self,
        feeds: List[Dict[str, str]],
        days_back: int = 7,
        max_items: int = 10,
        ref_date: Optional[datetime] = None,
    ):
        """
        Initialize BlogScanner.

        Args:
            feeds: List of feed dictionaries with 'name' and 'url'
            days_back: Number of days to look back
            max_items: Maximum number of items per feed
            ref_date: Reference point for the look-back window (defaults to
                now). Pass a past datetime to re-generate a historical briefing.
        """
        self.feeds = feeds
        self.days_back = days_back
        self.max_items = max_items
        self.ref_date = ref_date

    def scan_feed(self, feed_name: str, feed_url: str) -> List[Dict[str, Any]]:
        """
        Scan a single RSS feed.

        Args:
            feed_name: Name of the blog/feed
            feed_url: URL of the RSS feed

        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            logger.info(f"Scanning feed: {feed_name}")
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning(f"Feed parsing issue for {feed_name}: {feed.bozo_exception}")

            # Calculate cutoff date
            ref = (
                self.ref_date.replace(tzinfo=timezone.utc)
                if self.ref_date and self.ref_date.tzinfo is None
                else self.ref_date or datetime.now(timezone.utc)
            )
            cutoff_date = ref - timedelta(days=self.days_back)

            for entry in feed.entries[:self.max_items]:
                # Parse published date
                published_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

                # Filter by date if available
                if published_date and published_date < cutoff_date:
                    continue

                article = {
                    "source": feed_name,
                    "title": entry.get("title", "").strip(),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", entry.get("description", "")).strip(),
                    "published": (
                        published_date.isoformat() if published_date else ""
                    ),
                    "author": entry.get("author", ""),
                }

                articles.append(article)

            logger.info(f"Found {len(articles)} articles from {feed_name}")
            return articles

        except Exception as e:
            logger.error(f"Failed to scan feed '{feed_name}': {e}")
            return []

    def scan_all_feeds(self) -> List[Dict[str, Any]]:
        """
        Scan all configured feeds in parallel.

        Returns:
            List of all articles found across feeds
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_articles = []
        valid_feeds = [
            f for f in self.feeds
            if f.get("name") and f.get("url")
        ]

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(self.scan_feed, f["name"], f["url"]): f["name"]
                for f in valid_feeds
            }
            for future in as_completed(futures):
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    feed_name = futures[future]
                    logger.warning(f"Feed scan failed for {feed_name}: {e}")

        logger.info(f"Total articles found: {len(all_articles)}")
        return all_articles



def main() -> int:
    """
    Main entry point for blog_scanner.

    Returns:
        Exit code (0 for success, 1 for partial failure, 2 for total failure)
    """
    parser = argparse.ArgumentParser(description="Scan RSS feeds for new articles")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="blogs.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(getattr(logging, args.log_level))

    # Load config
    config = load_config(args.config)

    # Extract settings
    feeds = config.get("blog_feeds", [])
    days_back = config.get("arxiv_days_back", 7)  # Reuse same setting
    max_blogs = config.get("max_blogs", 10)

    if not feeds:
        logger.error("No blog_feeds configured")
        return 2

    # Scan feeds
    scanner = BlogScanner(feeds=feeds, days_back=days_back, max_items=max_blogs)
    articles = scanner.scan_all_feeds()

    if not articles:
        logger.warning("No articles found")
        return 1

    # Save results
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(articles, f, indent=2)
        logger.info(f"Saved {len(articles)} articles to {args.output}")
        return 0
    except IOError as e:
        logger.error(f"Failed to write output file: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
