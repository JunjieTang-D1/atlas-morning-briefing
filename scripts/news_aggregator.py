#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
News aggregator.

Aggregates news headlines using Brave Search API.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

from scripts.utils import load_config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregates news using Brave Search API."""

    BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/news/search"

    def __init__(self, api_key: str, queries: List[str], max_results: int = 15, request_delay: float = 1.0):
        """
        Initialize NewsAggregator.

        Args:
            api_key: Brave Search API key
            queries: List of search queries
            max_results: Maximum number of results per query
            request_delay: Seconds to wait between API calls
        """
        self.api_key = api_key
        self.queries = queries
        self.max_results = max_results
        self.request_delay = request_delay

    def search_news(self, query: str) -> List[Dict[str, Any]]:
        """
        Search news for a specific query.

        Args:
            query: Search query

        Returns:
            List of news article dictionaries
        """
        try:
            logger.info(f"Searching news for: {query}")
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            }
            params = {
                "q": query,
                "count": self.max_results,
                "freshness": "pd",  # Past day
            }

            response = requests.get(
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

            logger.info(f"Found {len(articles)} articles for query: {query}")
            return articles

        except requests.RequestException as e:
            logger.error(f"Failed to search news for '{query}': {e}")
            return []

    def aggregate_all_queries(self) -> List[Dict[str, Any]]:
        """
        Aggregate news for all queries in parallel.

        Uses controlled concurrency to respect API rate limits.

        Returns:
            List of all news articles found
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_articles = []
        seen_urls = set()

        # Use 4 workers (Brave free tier is generous but don't blast it)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(self.search_news, query): query
                for query in self.queries
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
                    query = futures[future]
                    logger.warning(f"News search failed for '{query}': {e}")

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

    # Get API key from environment
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        logger.error("BRAVE_API_KEY environment variable not set")
        return 2

    # Load config
    config = load_config(args.config)

    # Extract settings
    queries = config.get("news_queries", [])
    max_news = config.get("max_news", 15)

    if not queries:
        logger.error("No news_queries configured")
        return 2

    # Aggregate news
    aggregator = NewsAggregator(api_key=api_key, queries=queries, max_results=max_news)
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
