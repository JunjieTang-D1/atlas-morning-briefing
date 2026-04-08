#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
ArXiv paper scanner.

Scans arxiv.org for papers matching configured topics.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from scripts.utils import load_config

try:
    from defusedxml.ElementTree import fromstring as _xml_fromstring
except ImportError:
    raise ImportError(
        "defusedxml is required for safe XML parsing. "
        "Install it with: pip install defusedxml"
    )


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ArxivScanner:
    """Scans ArXiv for papers matching configured topics."""

    ARXIV_API_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        topics: List[str],
        days_back: int = 7,
        max_results: int = 20,
        end_date: Optional[datetime] = None,
    ):
        """
        Initialize ArxivScanner.

        Args:
            topics: List of topics to search for
            days_back: Number of days to look back
            max_results: Maximum number of results per topic
            end_date: End of the search window (defaults to now). Pass a past
                datetime to re-generate a briefing for a historical date.
        """
        self.topics = topics
        self.days_back = days_back
        self.max_results = max_results
        self.end_date = end_date

    def search_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Search ArXiv for papers on a specific topic.

        Args:
            topic: Topic to search for

        Returns:
            List of paper dictionaries
        """
        try:
            # Calculate date range
            end_date = (
                self.end_date.replace(tzinfo=timezone.utc)
                if self.end_date and self.end_date.tzinfo is None
                else self.end_date or datetime.now(timezone.utc)
            )
            start_date = end_date - timedelta(days=self.days_back)

            # Build query
            query = f"all:{topic}"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            logger.info(f"Searching ArXiv for topic: {topic}")
            response = requests.get(self.ARXIV_API_URL, params=params, timeout=30)
            response.raise_for_status()

            papers = self._parse_arxiv_response(response.text, start_date)
            logger.info(f"Found {len(papers)} papers for topic: {topic}")
            return papers

        except requests.RequestException as e:
            logger.error(f"Failed to search ArXiv for topic '{topic}': {e}")
            return []

    def _parse_arxiv_response(
        self, xml_content: str, start_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Parse ArXiv API XML response.

        Args:
            xml_content: XML response from ArXiv API
            start_date: Filter papers published after this date

        Returns:
            List of paper dictionaries
        """
        papers = []
        try:
            root = _xml_fromstring(xml_content)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", namespace):
                # Extract paper details
                paper_id = entry.find("atom:id", namespace)
                title = entry.find("atom:title", namespace)
                summary = entry.find("atom:summary", namespace)
                published = entry.find("atom:published", namespace)
                updated = entry.find("atom:updated", namespace)

                # Extract authors
                authors = []
                for author in entry.findall("atom:author", namespace):
                    name = author.find("atom:name", namespace)
                    if name is not None and name.text:
                        authors.append(name.text.strip())

                # Extract categories
                categories = []
                for category in entry.findall("atom:category", namespace):
                    term = category.get("term")
                    if term:
                        categories.append(term)

                # Extract links
                pdf_link = None
                for link in entry.findall("atom:link", namespace):
                    if link.get("title") == "pdf":
                        pdf_link = link.get("href")
                        break

                if not pdf_link:
                    # Fallback to constructing PDF link from ID
                    paper_url = paper_id.text if paper_id is not None else ""
                    if paper_url:
                        pdf_link = paper_url.replace("/abs/", "/pdf/") + ".pdf"

                # Parse published date
                if published is not None and published.text:
                    pub_date = datetime.fromisoformat(
                        published.text.replace("Z", "+00:00")
                    )
                    # Filter by date range
                    if pub_date < start_date:
                        continue
                else:
                    continue

                paper = {
                    "id": paper_id.text.strip() if paper_id is not None else "",
                    "title": title.text.strip() if title is not None else "",
                    "summary": summary.text.strip() if summary is not None else "",
                    "authors": authors,
                    "published": published.text if published is not None else "",
                    "updated": updated.text if updated is not None else "",
                    "categories": categories,
                    "pdf_link": pdf_link,
                    "arxiv_url": paper_id.text if paper_id is not None else "",
                }

                papers.append(paper)

        except Exception as e:
            logger.error(f"Failed to parse ArXiv XML response: {e}")

        return papers

    def scan_all_topics(self) -> List[Dict[str, Any]]:
        """
        Scan all configured topics in parallel.

        Returns:
            List of all papers found across topics
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_papers = []
        seen_ids = set()

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(self.search_topic, topic): topic
                for topic in self.topics
            }
            for future in as_completed(futures):
                try:
                    papers = future.result()
                    for paper in papers:
                        paper_id = paper.get("id", "")
                        if paper_id and paper_id not in seen_ids:
                            all_papers.append(paper)
                            seen_ids.add(paper_id)
                except Exception as e:
                    topic = futures[future]
                    logger.warning(f"ArXiv scan failed for topic '{topic}': {e}")

        logger.info(f"Total unique papers found: {len(all_papers)}")
        return all_papers



def main() -> int:
    """
    Main entry point for arxiv_scanner.

    Returns:
        Exit code (0 for success, 1 for partial failure, 2 for total failure)
    """
    parser = argparse.ArgumentParser(description="Scan ArXiv for papers on topics")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="papers.json",
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
    topics = config.get("arxiv_topics", [])
    days_back = config.get("arxiv_days_back", 7)
    max_papers = config.get("max_papers", 20)

    if not topics:
        logger.error("No arxiv_topics configured")
        return 2

    # Scan papers
    scanner = ArxivScanner(topics=topics, days_back=days_back, max_results=max_papers)
    papers = scanner.scan_all_topics()

    if not papers:
        logger.warning("No papers found")
        return 1

    # Save results
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(papers, f, indent=2)
        logger.info(f"Saved {len(papers)} papers to {args.output}")
        return 0
    except IOError as e:
        logger.error(f"Failed to write output file: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
