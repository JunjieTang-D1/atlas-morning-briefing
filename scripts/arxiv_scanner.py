#!/usr/bin/env python3
"""
ArXiv paper scanner using DeepXiv SDK.

Replaces the legacy ArXiv API scanner with DeepXiv's agent-first
search and progressive reading interface.

DeepXiv advantages over raw ArXiv API:
- Semantic search (hybrid mode) vs keyword-only
- TLDR / brief summaries without loading full paper
- Citation counts and GitHub URLs
- 200M+ papers indexed
- T+1 daily sync
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Try DeepXiv SDK first, fall back to legacy ArXiv API
try:
    from deepxiv_sdk import Reader as DeepXivReader
    HAS_DEEPXIV = True
    logger.info("Using DeepXiv SDK for paper search")
except ImportError:
    HAS_DEEPXIV = False
    logger.warning("DeepXiv SDK not installed, falling back to legacy ArXiv API")


def _load_deepxiv_token() -> str | None:
    """Load DeepXiv token from ~/.env or environment."""
    import os
    token = os.environ.get("DEEPXIV_TOKEN")
    if token:
        return token
    # Try ~/.env (where deepxiv CLI saves auto-registered token)
    env_path = Path.home() / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DEEPXIV_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class DeepXivScanner:
    """Scans ArXiv via DeepXiv SDK with semantic search and progressive reading."""

    def __init__(self, topics: list[str], days_back: int = 7, max_results: int = 20):
        self.topics = topics
        self.days_back = days_back
        self.max_results = max_results
        token = _load_deepxiv_token()
        self.reader = DeepXivReader(token=token)

    def search_topic(self, topic: str) -> list[dict[str, Any]]:
        """Search DeepXiv for papers on a topic."""
        try:
            logger.info(f"DeepXiv search: {topic}")

            start_date = datetime.now(timezone.utc) - timedelta(days=self.days_back)
            date_from = start_date.strftime("%Y-%m-%d")

            response = self.reader.search(
                topic,
                size=self.max_results,
                search_mode="hybrid",
                date_from=date_from,
            )

            # DeepXiv returns {"total": N, "results": [...], "took": ms}
            if isinstance(response, dict) and "results" in response:
                raw_papers = response["results"]
            elif isinstance(response, list):
                raw_papers = response
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                raw_papers = []

            papers = []
            for r in raw_papers:
                paper = self._normalize_result(r)
                if paper:
                    papers.append(paper)

            logger.info(f"Found {len(papers)} papers for: {topic}")
            return papers

        except Exception as e:
            logger.error(f"DeepXiv search failed for '{topic}': {e}")
            return []

    def _normalize_result(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize a DeepXiv search result to our standard format."""
        try:
            arxiv_id = data.get("arxiv_id", data.get("id", ""))
            if isinstance(arxiv_id, str) and "arxiv.org" in arxiv_id:
                arxiv_id = arxiv_id.split("/")[-1]

            title = data.get("title", "").strip()
            summary = data.get("abstract", data.get("summary", "")).strip()

            # Authors: DeepXiv 'authors' = list[str], 'author_names' = space-joined string (unreliable)
            raw_authors = data.get("authors", [])
            if isinstance(raw_authors, list) and raw_authors:
                authors = [
                    a.get("name", str(a)) if isinstance(a, dict) else str(a)
                    for a in raw_authors
                ]
            elif isinstance(raw_authors, str) and "," in raw_authors:
                authors = [a.strip() for a in raw_authors.split(",")]
            else:
                # Last resort: author_names might be comma-separated
                an = data.get("author_names", "")
                authors = [a.strip() for a in an.split(",")] if "," in an else []

            published = data.get("publish_at", data.get("published", data.get("created_at", "")))
            categories = data.get("categories", [])
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(",") if c.strip()]

            citations = data.get("citation", data.get("citations", 0)) or 0
            score = data.get("score", 0)

            return {
                "id": f"http://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "arxiv_id": str(arxiv_id),
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": str(published) if published else "",
                "updated": str(data.get("updated_at", data.get("modified_at", published) or "")),
                "categories": categories,
                "pdf_link": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "citations": citations,
                "deepxiv_score": score,
                "source": "deepxiv",
            }
        except Exception as e:
            logger.warning(f"Failed to normalize result: {e}")
            return None

    def enrich_paper(self, arxiv_id: str) -> dict[str, Any]:
        """Get brief summary for a paper (saves tokens vs full read)."""
        try:
            clean_id = arxiv_id.split("/")[-1] if "/" in arxiv_id else arxiv_id
            brief = self.reader.brief(clean_id)
            if isinstance(brief, dict):
                return brief
            elif isinstance(brief, str):
                return {"brief": brief}
            elif hasattr(brief, "__dict__"):
                return brief.__dict__
            return {}
        except Exception as e:
            logger.debug(f"Brief failed for {arxiv_id}: {e}")
            return {}

    def scan_all_topics(self) -> list[dict[str, Any]]:
        """Scan all configured topics, deduplicate by arxiv ID."""
        all_papers = []
        seen_ids: set[str] = set()

        for topic in self.topics:
            papers = self.search_topic(topic)
            for paper in papers:
                pid = paper.get("arxiv_id", paper.get("id", ""))
                if pid and pid not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(pid)

        logger.info(f"Total unique papers: {len(all_papers)}")

        # Enrich top 10 papers with briefs (save API budget)
        top_papers = sorted(all_papers, key=lambda p: p.get("deepxiv_score", 0), reverse=True)[:10]
        for paper in top_papers:
            aid = paper.get("arxiv_id", "")
            if aid:
                brief_data = self.enrich_paper(aid)
                if brief_data:
                    if "tldr" in brief_data:
                        paper["tldr"] = brief_data["tldr"]
                    if brief_data.get("github_url"):
                        paper["github_url"] = brief_data["github_url"]
                    if "keywords" in brief_data:
                        paper["keywords"] = brief_data["keywords"]

        return all_papers


# ── Legacy fallback (original ArXiv API scanner) ──


class ArxivScanner:
    """Legacy ArXiv API scanner (fallback when DeepXiv is not installed)."""

    ARXIV_API_URL = "http://export.arxiv.org/api/query"

    def __init__(self, topics: list[str], days_back: int = 7, max_results: int = 20):
        self.topics = topics
        self.days_back = days_back
        self.max_results = max_results

    def search_topic(self, topic: str) -> list[dict[str, Any]]:
        """Search ArXiv API for papers on a topic."""
        try:
            import requests
            from xml.etree import ElementTree as ET

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.days_back)

            query = f"all:{topic}"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            logger.info(f"Legacy ArXiv search: {topic}")
            response = requests.get(self.ARXIV_API_URL, params=params, timeout=30)
            response.raise_for_status()

            papers = []
            root = ET.fromstring(response.text)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", namespace):
                paper_id = entry.find("atom:id", namespace)
                title = entry.find("atom:title", namespace)
                summary = entry.find("atom:summary", namespace)
                published = entry.find("atom:published", namespace)
                updated = entry.find("atom:updated", namespace)

                authors = []
                for author in entry.findall("atom:author", namespace):
                    name = author.find("atom:name", namespace)
                    if name is not None and name.text:
                        authors.append(name.text.strip())

                categories = []
                for category in entry.findall("atom:category", namespace):
                    term = category.get("term")
                    if term:
                        categories.append(term)

                pdf_link = None
                for link in entry.findall("atom:link", namespace):
                    if link.get("title") == "pdf":
                        pdf_link = link.get("href")
                        break
                if not pdf_link and paper_id is not None and paper_id.text:
                    pdf_link = paper_id.text.replace("/abs/", "/pdf/") + ".pdf"

                if published is not None and published.text:
                    pub_date = datetime.fromisoformat(published.text.replace("Z", "+00:00"))
                    if pub_date < start_date:
                        continue
                else:
                    continue

                papers.append({
                    "id": paper_id.text.strip() if paper_id is not None else "",
                    "title": title.text.strip() if title is not None else "",
                    "summary": summary.text.strip() if summary is not None else "",
                    "authors": authors,
                    "published": published.text if published is not None else "",
                    "updated": updated.text if updated is not None else "",
                    "categories": categories,
                    "pdf_link": pdf_link,
                    "arxiv_url": paper_id.text if paper_id is not None else "",
                    "source": "arxiv_api",
                })

            logger.info(f"Found {len(papers)} papers for: {topic}")
            return papers

        except Exception as e:
            logger.error(f"Legacy ArXiv search failed for '{topic}': {e}")
            return []

    def scan_all_topics(self) -> list[dict[str, Any]]:
        """Scan all topics, deduplicate."""
        all_papers = []
        seen_ids: set[str] = set()

        for topic in self.topics:
            papers = self.search_topic(topic)
            for paper in papers:
                paper_id = paper.get("id", "")
                if paper_id and paper_id not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(paper_id)

        logger.info(f"Total unique papers: {len(all_papers)}")
        return all_papers


# ── Factory function ──


def create_scanner(topics: list[str], days_back: int = 7, max_results: int = 20):
    """Create the best available scanner (DeepXiv > legacy ArXiv API)."""
    if HAS_DEEPXIV:
        return DeepXivScanner(topics=topics, days_back=days_back, max_results=max_results)
    return ArxivScanner(topics=topics, days_back=days_back, max_results=max_results)


# Keep old class name as alias for backwards compatibility
# (briefing_runner.py imports ArxivScanner directly)
if HAS_DEEPXIV:
    # When DeepXiv is available, ArxivScanner points to DeepXiv
    _OriginalArxivScanner = ArxivScanner
    ArxivScanner = DeepXivScanner  # type: ignore[misc]


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(2)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config: {e}")
        sys.exit(2)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scan ArXiv for papers (DeepXiv or legacy)")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--output", type=str, default="papers.json", help="Output JSON path")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    config = load_config(args.config)
    topics = config.get("arxiv_topics", [])
    days_back = config.get("arxiv_days_back", 7)
    max_papers = config.get("max_papers", 20)

    if not topics:
        logger.error("No arxiv_topics configured")
        return 2

    scanner = create_scanner(topics=topics, days_back=days_back, max_results=max_papers)
    papers = scanner.scan_all_topics()

    if not papers:
        logger.warning("No papers found")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(papers, f, indent=2)
    logger.info(f"Saved {len(papers)} papers to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
