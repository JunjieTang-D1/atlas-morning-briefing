#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
News aggregator.

Primary source: RSS feeds from news_feeds config.
Fallback: Brave Search API (per-query, only when RSS yields < 5 results).
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import feedparser
import requests

from scripts.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError
from scripts.utils import load_config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregates news: RSS feeds first, Brave Search as per-query fallback."""

    BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/news/search"
    # Minimum RSS hits per query before we fall back to Brave
    RSS_FALLBACK_THRESHOLD = 5

    def __init__(
        self,
        api_key: str,
        queries: List[str],
        news_feeds: List[Dict[str, str]] = None,
        max_results: int = 15,
        request_delay: float = 1.0,
    ):
        """
        Initialize NewsAggregator.

        Args:
            api_key: Brave Search API key (used as fallback only)
            queries: List of search queries
            news_feeds: List of RSS feed dicts with 'name' and 'url'
            max_results: Maximum number of results per Brave query
            request_delay: Seconds to wait between Brave API calls
        """
        self.api_key = api_key
        self.queries = queries
        self.news_feeds = news_feeds or []
        self.max_results = max_results
        self.request_delay = request_delay

    def fetch_rss_news(self) -> List[Dict[str, Any]]:
        """
        Fetch recent news articles from all configured RSS feeds.

        Filters to articles published in the last 24 hours.

        Returns:
            List of news article dicts (title, url, description, source, query)
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        articles: List[Dict[str, Any]] = []

        for feed_cfg in self.news_feeds:
            name = feed_cfg.get("name", "")
            url = feed_cfg.get("url", "")
            if not url:
                continue
            try:
                logger.info(f"Fetching RSS news feed: {name}")
                parsed = feedparser.parse(url)
                for entry in parsed.entries:
                    pub = entry.get("published_parsed") or entry.get("updated_parsed")
                    if pub:
                        pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
                        if pub_dt < cutoff:
                            continue
                    articles.append({
                        "query": "",  # filled in when matched against a query
                        "title": entry.get("title", "").strip(),
                        "url": entry.get("link", ""),
                        "description": entry.get("summary", entry.get("description", "")).strip(),
                        "age": "",
                        "source": name,
                        "thumbnail": "",
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed '{name}': {e}")

        logger.info(f"RSS feeds: {len(articles)} articles from last 24h")
        return articles

    def _filter_rss_by_query(
        self, rss_items: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Return RSS items where query terms appear in title or description."""
        terms = [t.lower() for t in query.split() if len(t) > 2]
        matched = []
        for item in rss_items:
            haystack = (item["title"] + " " + item["description"]).lower()
            if any(term in haystack for term in terms):
                matched.append({**item, "query": query})
        return matched

    def search_news(self, query: str) -> List[Dict[str, Any]]:
        """
        Search news via Brave Search API (fallback path).

        Args:
            query: Search query

        Returns:
            List of news article dictionaries
        """
        try:
            logger.info(f"Brave fallback search for: {query}")
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            }
            params = {
                "q": query,
                "count": self.max_results,
                "freshness": "pd",  # Past day
            }

            cb = CircuitBreakerRegistry.get("brave-search-api", failure_threshold=3, recovery_timeout=60.0)
            response = cb.call(
                requests.get,
                self.BRAVE_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            results = data.get("results", [])

            for result in results:
                article = {
                    "query": query,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "description": result.get("description", ""),
                    "age": result.get("age", ""),
                    "source": result.get("meta_url", {}).get("hostname", ""),
                    "thumbnail": result.get("thumbnail", {}).get("src", ""),
                }
                articles.append(article)

            logger.info(f"Brave: found {len(articles)} articles for: {query}")
            return articles

        except CircuitOpenError as e:
            logger.warning(f"Brave Search skipped (circuit open): {e}")
            return []
        except requests.RequestException as e:
            logger.error(f"Failed Brave search for '{query}': {e}")
            return []

    def aggregate_all_queries(self) -> List[Dict[str, Any]]:
        """
        Aggregate news: RSS-first per query, Brave as fallback.

        For each query: filter RSS items by keyword relevance.
        If fewer than RSS_FALLBACK_THRESHOLD results, call Brave Search.
        Deduplicates by URL across all results.

        Returns:
            List of all unique news articles found
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step 1: fetch all RSS feeds up front
        rss_items = self.fetch_rss_news() if self.news_feeds else []

        all_articles: List[Dict[str, Any]] = []
        seen_urls: set = set()

        # Step 2: for each query, match RSS items; fall back to Brave if needed
        queries_needing_brave: List[str] = []
        for query in self.queries:
            matched = self._filter_rss_by_query(rss_items, query)
            for item in matched:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    all_articles.append(item)
                    seen_urls.add(url)
            if len(matched) < self.RSS_FALLBACK_THRESHOLD:
                logger.info(
                    f"RSS matched only {len(matched)} items for '{query}' "
                    f"(threshold {self.RSS_FALLBACK_THRESHOLD}) — queuing Brave fallback"
                )
                queries_needing_brave.append(query)

        # Step 3: run Brave for queries that didn't get enough RSS coverage
        if queries_needing_brave and self.api_key:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    pool.submit(self.search_news, q): q
                    for q in queries_needing_brave
                }
                for future in as_completed(futures):
                    try:
                        articles = future.result()
                        for article in articles:
                            url = article.get("url", "")
                            if url and url not in seen_urls:
                                all_articles.append(article)
                                seen_urls.add(url)
                    except Exception as e:
                        q = futures[future]
                        logger.warning(f"Brave search failed for '{q}': {e}")

        logger.info(f"Total unique articles found: {len(all_articles)}")
        return all_articles



def main() -> int:
    """
    Main entry point for news_aggregator.

    Returns:
        Exit code (0 for success, 1 for partial failure, 2 for total failure)
    """
    parser = argparse.ArgumentParser(description="Aggregate news headlines")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="news.json",
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

    # Get API key from environment (optional when news_feeds are configured)
    api_key = os.environ.get("BRAVE_API_KEY", "")

    # Load config
    config = load_config(args.config)

    # Extract settings
    queries = config.get("news_queries", [])
    news_feeds = config.get("news_feeds", [])
    max_news = config.get("max_news", 15)

    if not queries:
        logger.error("No news_queries configured")
        return 2

    if not api_key and not news_feeds:
        logger.error("BRAVE_API_KEY not set and no news_feeds configured — nothing to fetch")
        return 2

    if not api_key:
        logger.warning("BRAVE_API_KEY not set — Brave Search fallback disabled; using RSS only")

    # Aggregate news (RSS-first; Brave as fallback)
    aggregator = NewsAggregator(
        api_key=api_key,
        queries=queries,
        news_feeds=news_feeds,
        max_results=max_news,
    )
    articles = aggregator.aggregate_all_queries()

    if not articles:
        logger.warning("No news articles found")
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
