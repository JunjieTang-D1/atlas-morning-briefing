#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Morning briefing runner.

Main orchestrator that runs all scanners, applies intelligence layer,
and generates the briefing. Supports Amazon Bedrock for LLM-powered
synthesis and summarization.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml

# Ensure scripts directory is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.arxiv_scanner import ArxivScanner
from scripts.blog_scanner import BlogScanner
from scripts.stock_fetcher import StockFetcher
from scripts.news_aggregator import NewsAggregator
from scripts.paper_scorer import PaperScorer
from scripts.pdf_generator import PDFGenerator
from scripts.email_distributor import EmailDistributor
from scripts.config_validator import validate_config, check_environment
from scripts.llm_client import LLMClient
from scripts.intelligence import BriefingIntelligence
from scripts.newsletter_scanner import NewsletterScanner
from scripts.github_trending_scanner import GitHubTrendingScanner
from opentelemetry import trace
from scripts.tracing import setup_tracing, get_tracer
from scripts.obsidian_writer import ObsidianWriter
from scripts.podcast_generator import PodcastGenerator


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STATE_FILENAME = ".personal-state.json"


class BriefingRunner:
    """Main orchestrator for morning briefing generation."""

    # Default section order
    DEFAULT_SECTION_ORDER = ["stocks", "news", "community_picks", "newsletters", "top_papers", "blogs"]

    def __init__(self, config: Dict[str, Any], dry_run: bool = False):
        """
        Initialize BriefingRunner.

        Args:
            config: Configuration dictionary.
            dry_run: If True, don't send email.
        """
        self.config = config
        self.dry_run = dry_run
        self.errors = []
        self._briefing_title = self._format_filename(datetime.now())
        self.status = {
            "timestamp": datetime.now().isoformat(),
            "papers_found": 0,
            "blogs_found": 0,
            "stocks_fetched": 0,
            "news_found": 0,
            "newsletters_found": 0,
            "github_trending_found": 0,
            "intelligence_enabled": False,
            "errors": [],
            "pdf_generated": False,
            "email_sent": False,
            "elapsed_seconds": 0,
        }

        # Initialize LLM client and intelligence layer
        llm_config = config.get("llm", config.get("bedrock", {}))
        self.llm = LLMClient(llm_config)
        self.intelligence = BriefingIntelligence(self.llm, config)
        self.newsletter_scanner = None  # Initialized in run_newsletter_scan
        self.status["intelligence_enabled"] = self.intelligence.available

        # Initialize podcast generator (requires notebooklm-py + auth)
        podcast_config = config.get("podcast", {})
        self.podcast_generator = PodcastGenerator(podcast_config)

        # Initialize OTel tracing (no-op if OTEL_EXPORTER_OTLP_ENDPOINT not set)
        setup_tracing(config)
        self._tracer = get_tracer()

    def run_arxiv_scan(self, topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run arxiv paper scan."""
        try:
            logger.info("=== Scanning ArXiv Papers ===")
            if topics is None:
                topics = self.config.get("arxiv_topics", [])
            days_back = self.config.get("arxiv_days_back", 7)
            max_papers = self.config.get("max_papers", 20)

            if not topics:
                logger.warning("No arxiv_topics configured, skipping")
                return []

            scanner = ArxivScanner(
                topics=topics,
                days_back=days_back,
                max_results=max_papers,
            )
            papers = scanner.scan_all_topics()
            self.status["papers_found"] = len(papers)
            logger.info(f"Found {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"ArXiv scan failed: {e}")
            self.errors.append(f"ArXiv scan: {e}")
            return []

    def run_blog_scan(self) -> List[Dict[str, Any]]:
        """Run blog feed scan."""
        try:
            logger.info("=== Scanning Blog Feeds ===")
            feeds = self.config.get("blog_feeds", [])
            days_back = self.config.get("arxiv_days_back", 7)
            max_blogs = self.config.get("max_blogs", 10)

            if not feeds:
                logger.warning("No blog_feeds configured, skipping")
                return []

            scanner = BlogScanner(
                feeds=feeds,
                days_back=days_back,
                max_items=max_blogs,
            )
            articles = scanner.scan_all_feeds()
            self.status["blogs_found"] = len(articles)
            logger.info(f"Found {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"Blog scan failed: {e}")
            self.errors.append(f"Blog scan: {e}")
            return []

    def run_stock_fetch(self) -> List[Dict[str, Any]]:
        """Run stock data fetch."""
        try:
            logger.info("=== Fetching Stock Data ===")
            api_key = os.environ.get("FINNHUB_API_KEY")
            symbols = self.config.get("stocks", [])

            if not api_key:
                logger.warning("FINNHUB_API_KEY not set, skipping stocks")
                return []

            if not symbols:
                logger.warning("No stocks configured, skipping")
                return []

            fetcher = StockFetcher(api_key=api_key, symbols=symbols)
            stocks = fetcher.fetch_all_stocks()
            self.status["stocks_fetched"] = len(stocks)
            logger.info(f"Fetched data for {len(stocks)} stocks")
            return stocks

        except Exception as e:
            logger.error(f"Stock fetch failed: {e}")
            self.errors.append(f"Stock fetch: {e}")
            return []

    def run_news_aggregation(
        self, queries: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run news aggregation.

        Args:
            queries: Optional list of queries. If None, uses config.

        Returns:
            List of news articles.
        """
        try:
            logger.info("=== Aggregating News ===")
            api_key = os.environ.get("BRAVE_API_KEY")
            if queries is None:
                queries = self.config.get("news_queries", [])
            max_news = self.config.get("max_news", 15)

            if not api_key:
                logger.warning("BRAVE_API_KEY not set, skipping news")
                return []

            if not queries:
                logger.warning("No news_queries configured, skipping")
                return []

            aggregator = NewsAggregator(
                api_key=api_key,
                queries=queries,
                max_results=max_news,
            )
            articles = aggregator.aggregate_all_queries()
            self.status["news_found"] = len(articles)
            logger.info(f"Found {len(articles)} news articles")
            return articles

        except Exception as e:
            logger.error(f"News aggregation failed: {e}")
            self.errors.append(f"News aggregation: {e}")
            return []

    def run_newsletter_scan(self) -> List[Dict[str, Any]]:
        """Run newsletter scan from Supabase."""
        try:
            ns_config = self.config.get("newsletter_source", {})
            if not ns_config.get("enabled", False):
                return []

            logger.info("=== Scanning Newsletters ===")
            self.newsletter_scanner = NewsletterScanner(
                supabase_url=ns_config.get("supabase_url"),
                supabase_key=ns_config.get("supabase_key"),
                max_items=ns_config.get("max_items", 20),
            )
            items = self.newsletter_scanner.scan()
            self.status["newsletters_found"] = len(items)
            logger.info(f"Found {len(items)} newsletter items")
            return items

        except Exception as e:
            logger.error(f"Newsletter scan failed: {e}")
            self.errors.append(f"Newsletter scan: {e}")
            return []

    def run_github_trending_scan(self) -> List[Dict[str, Any]]:
        """Run GitHub trending scan from Supabase."""
        try:
            gt_config = self.config.get("github_trending", {})
            if not gt_config.get("enabled", False):
                return []

            logger.info("=== Scanning GitHub Trending ===")
            scanner = GitHubTrendingScanner(
                max_items=gt_config.get("max_items", 20),
            )
            items = scanner.scan()
            self.status["github_trending_found"] = len(items)
            logger.info(f"Found {len(items)} trending repos")
            return items

        except Exception as e:
            logger.error(f"GitHub trending scan failed: {e}")
            self.errors.append(f"GitHub trending scan: {e}")
            return []

    def score_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank papers."""
        try:
            if not papers:
                return []

            logger.info("=== Scoring Papers ===")
            topics = self.config.get("arxiv_topics", [])
            weights = self.config.get("paper_scoring", {})
            num_picks = self.config.get("num_paper_picks", 3)

            scorer = PaperScorer(topics=topics, weights=weights, num_picks=num_picks)
            top_papers = scorer.get_top_picks(papers)
            logger.info(f"Selected top {len(top_papers)} papers")
            return top_papers

        except Exception as e:
            logger.error(f"Paper scoring failed: {e}")
            self.errors.append(f"Paper scoring: {e}")
            return []

    def deduplicate_news_and_blogs(
        self,
        news: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
    ) -> tuple:
        """
        Remove duplicate content between news and blog sections.

        Args:
            news: News articles.
            blogs: Blog articles.

        Returns:
            Tuple of (deduplicated_news, deduplicated_blogs).
        """
        blog_domains = set()
        blog_titles_lower = set()

        for blog in blogs:
            link = blog.get("link", "")
            if link:
                try:
                    domain = urlparse(link).netloc.lower()
                    blog_domains.add(domain)
                except Exception:
                    pass
            title = blog.get("title", "").lower().strip()
            if title:
                blog_titles_lower.add(title)

        deduped_news = []
        for article in news:
            url = article.get("url", "")
            title = article.get("title", "").lower().strip()

            # Skip if same title appears in blogs
            if title and title in blog_titles_lower:
                logger.debug(f"Dedup: removing news '{title}' (duplicate of blog)")
                continue

            # Skip if URL points to same domain as a blog feed
            if url:
                try:
                    domain = urlparse(url).netloc.lower()
                    if domain in blog_domains:
                        logger.debug(f"Dedup: removing news from {domain} (covered by blog feed)")
                        continue
                except Exception:
                    pass

            deduped_news.append(article)

        removed = len(news) - len(deduped_news)
        if removed > 0:
            logger.info(f"Dedup: removed {removed} news articles duplicated in blogs")

        return deduped_news, blogs

    @staticmethod
    def _dedup_against_previous(
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        previous_state: Dict[str, Any],
    ) -> tuple:
        """Remove papers, blogs, and news that appeared in yesterday's briefing."""
        if not previous_state:
            return papers, blogs, news

        prev_papers = set(t.lower() for t in previous_state.get("top_paper_titles", []))
        prev_blogs = set(t.lower() for t in previous_state.get("top_blog_titles", []))
        prev_news = set(t.lower() for t in previous_state.get("top_news_titles", []))

        def _filter(items, prev_titles):
            before = len(items)
            filtered = [i for i in items if i.get("title", "").lower() not in prev_titles]
            removed = before - len(filtered)
            if removed:
                logger.info(f"Cross-day dedup: removed {removed} items seen yesterday")
            return filtered

        return _filter(papers, prev_papers), _filter(blogs, prev_blogs), _filter(news, prev_news)

    def deduplicate_similar_papers(
        self, papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove papers with very similar titles (>85% match).

        Catches near-duplicates found via different topic queries.

        Args:
            papers: List of paper dictionaries.

        Returns:
            Deduplicated paper list.
        """
        if len(papers) <= 1:
            return papers

        deduped = []
        for paper in papers:
            title = paper.get("title", "").lower()
            is_dup = False
            for kept in deduped:
                kept_title = kept.get("title", "").lower()
                if SequenceMatcher(None, title, kept_title).ratio() > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(paper)

        removed = len(papers) - len(deduped)
        if removed:
            logger.info(f"Dedup: removed {removed} near-duplicate papers by title similarity")
        return deduped

    def generate_markdown_briefing(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        stocks: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        top_papers: List[Dict[str, Any]],
        synthesis: Optional[Dict[str, str]] = None,
        market_trend: str = "",
        newsletters: Optional[List[Dict[str, Any]]] = None,
        community_picks: Optional[List[Dict[str, Any]]] = None,
        weekly_deep_dive: str = "",
    ) -> str:
        """
        Generate markdown briefing from all data.

        Args:
            papers: ArXiv papers.
            blogs: Blog articles.
            stocks: Stock data.
            news: News articles.
            top_papers: Top-scored papers.
            synthesis: Optional intelligence synthesis output.
            market_trend: Pre-generated market trend summary.
            community_picks: Scored newsletter links + GitHub trending repos.
            weekly_deep_dive: Optional weekly deep dive section (Saturday only).

        Returns:
            Markdown string.
        """
        logger.info("=== Generating Briefing ===")

        md = []

        # Editorial intro (from synthesis)
        if synthesis and synthesis.get("editorial_intro"):
            intro = synthesis["editorial_intro"].strip()
            # Aggressively strip LLM preamble: headings, titles, dates
            lines = intro.split("\n")
            cleaned = []
            for line in lines:
                stripped = line.strip()
                # Skip markdown headings
                if stripped.startswith("#"):
                    continue
                # Skip lines containing "Executive Summary" (LLM echo)
                if "executive summary" in stripped.lower():
                    continue
                # Skip lines containing "Morning Briefing" or "AI Briefing" (LLM title echo)
                if "morning briefing" in stripped.lower() or "ai briefing" in stripped.lower():
                    continue
                # Skip stray date lines (e.g. "– 2026-03-08", "2026-03-07")
                date_stripped = stripped.lstrip("–—-*# ").strip()
                if re.match(r"^\d{4}-\d{2}-\d{2}$", date_stripped):
                    continue
                cleaned.append(line)
            intro = "\n".join(cleaned).strip()
            md.append("## Executive Summary\n\n")
            md.append(f"{intro}\n\n")

            # Feature 3: Entity Watch — DISABLED per user request (2026-03-08)
            # Only show if an entity has a spike (e.g., 5+ mentions).
            # entity_mentions = synthesis.get("entity_mentions", [])
            # if entity_mentions: ...

        # Fixed section order -- no LLM override
        section_order = self.DEFAULT_SECTION_ORDER

        # Section renderers
        section_data = {
            "stocks": stocks,
            "news": news,
            "newsletters": newsletters or [],
            "community_picks": community_picks or [],
            "blogs": blogs,
            "top_papers": top_papers,
            "papers": papers,
        }

        for section in section_order:
            data = section_data.get(section, [])
            if not data:
                continue

            if section == "stocks":
                md.append(self._render_stocks(data, market_trend=market_trend))
            elif section == "news":
                md.append(self._render_news(data))
            elif section == "community_picks":
                md.append(self._render_community_picks(data))
            elif section == "newsletters":
                md.append(self._render_newsletters(data))
            elif section == "blogs":
                md.append(self._render_blogs(data))
            elif section == "top_papers":
                md.append(self._render_top_papers(data))
            elif section == "papers":
                md.append(self._render_papers(data))

        # Feature 2: Weekly Deep Dive section (Saturday only)
        if weekly_deep_dive:
            md.append("## This Week in AI\n\n")
            md.append(f"{weekly_deep_dive}\n\n")

        # Errors section
        if self.errors:
            md.append("## Errors\n\n")
            for error in self.errors:
                md.append(f"- {error}\n")
            md.append("\n")

        return "".join(md)

    def _format_filename(self, now: datetime) -> str:
        """Format the output filename from config pattern, ignoring unknown keys."""
        file_naming = self.config.get("file_naming", "Personal-Briefing-{yyyy}.{mm}.{dd}")
        known_vars = {
            "yyyy": now.strftime("%Y"),
            "mm": now.strftime("%m"),
            "dd": now.strftime("%d"),
            "type": "Daily",
        }
        return file_naming.format_map(known_vars)

    def _enrich_papers(self, papers: list, topics: list) -> list:
        """Run paper summarization + semantic scoring sequentially (used in parallel batch)."""
        papers = self.intelligence.summarize_papers(papers)
        papers = self.intelligence.score_papers_semantically(papers, topics)
        return papers

    def _analyze_market_trend(self, stocks: List[Dict[str, Any]]) -> str:
        """Generate a 2-line market trend summary from stock data."""
        if not stocks or not self.intelligence.available:
            return ""
        stock_lines = []
        for s in stocks:
            if "error" not in s:
                pct = s.get("percent_change", 0)
                sign = "+" if pct >= 0 else ""
                corr = s.get("news_correlation", "")
                line = f"{s.get('symbol', '')}: {sign}{pct:.2f}%"
                if corr:
                    line += f" ({corr})"
                stock_lines.append(line)
        if not stock_lines:
            return ""
        data_block = "\n".join(stock_lines)
        prompt = (
            "You are a financial analyst. Given today's stock movements, "
            "write exactly 2 sentences summarizing the market trend and key drivers. "
            "Be specific about which sectors/stocks moved and why.\n\n"
            f"<stock_data>\n{data_block}\n</stock_data>"
        )
        result = self.intelligence.bedrock.invoke(
            prompt, tier="light", max_tokens=150
        )
        return result.strip() if result else ""

    def _render_stocks(self, stocks: List[Dict[str, Any]], market_trend: str = "") -> str:
        """Render stock watchlist as compact overview table with trend analysis."""
        md = ["## Financial Market Overview\n\n"]

        if market_trend:
            md.append(f"{market_trend}\n\n")

        md.append("| Ticker | Price | Change | Driver |\n")
        md.append("|--------|-------|--------|--------|\n")
        for stock in stocks:
            if "error" in stock:
                md.append(f"| {stock['symbol']} | — | Error | — |\n")
                continue

            symbol = stock.get("symbol", "")
            price = stock.get("current_price", 0)
            pct = stock.get("percent_change", 0)
            sign = "+" if pct >= 0 else ""
            driver = stock.get("news_correlation", "")
            if len(driver) > 30:
                driver = driver[:27] + "..."

            md.append(f"| **{symbol}** | ${price:.2f} | {sign}{pct:.2f}% | {driver} |\n")
        md.append("\n")
        return "".join(md)

    @staticmethod
    def _render_stars(score: int) -> str:
        """Render score as Amazon-style stars. 5 filled = best, 5 empty = worst."""
        if score is None:
            return ""
        score = max(0, min(score, 5))
        return "★" * score + "☆" * (5 - score)

    @staticmethod
    def _clean_summary(summary: str, title: str, source: str = "") -> str:
        """Remove title/source echo from LLM-generated summary."""
        if not summary:
            return summary
        # Strip leading * / ** markdown bold and "Summary:" prefix
        s = summary.lstrip("* ").strip()
        if s.lower().startswith("summary:"):
            s = s[8:].lstrip("* ").strip()
        if not title:
            return s
        # Check if summary starts with title text
        title_lower = title.lower()[:40]
        if s.lower().startswith(title_lower):
            rest = s[len(title):].strip()
            if rest.startswith("(") and ")" in rest:
                rest = rest[rest.index(")") + 1:].strip()
            if rest.startswith(("-", ":", "\u2013")):
                rest = rest[1:].strip()
            return rest if rest else summary
        return s

    def _render_news(self, news: List[Dict[str, Any]]) -> str:
        """Render news section with summaries."""
        max_news = self._get_limit("max_news_render", 5)
        md = ["## AI & Tech News\n\n"]
        for article in news[:max_news]:
            article_title = article.get("title", "")
            url = article.get("url", "")
            summary = self._clean_summary(
                article.get("brief_summary", ""), article_title
            )

            if url:
                md.append(f"**[{article_title}]({url})**\n")
            else:
                md.append(f"**{article_title}**\n")
            if summary:
                md.append(f"{summary}\n")
            md.append("\n")
        return "".join(md)

    def _render_blogs(self, blogs: List[Dict[str, Any]]) -> str:
        """Render blog updates section with summaries, sorted by score."""
        max_blogs = self._get_limit("max_blogs_render", 5)
        md = ["## Blog Updates\n\n"]
        sorted_blogs = sorted(blogs[:max_blogs], key=lambda x: x.get("score_combined", 0), reverse=True)
        # Only filter by score when scores are present (Bedrock enabled)
        if any(b.get("score_combined") for b in sorted_blogs):
            sorted_blogs = [b for b in sorted_blogs if b.get("score_combined", 0) >= 3]
        for article in sorted_blogs:
            article_title = article.get("title", "")
            source = article.get("source", "")
            link = article.get("link", "")
            score = article.get("score_combined")
            summary = self._clean_summary(
                article.get("brief_summary", ""), article_title, source
            )

            score_tag = f" {self._render_stars(score)}" if score else ""
            if link:
                md.append(f"**[{article_title}]({link})** *({source})*{score_tag}\n")
            else:
                md.append(f"**{article_title}** *({source})*{score_tag}\n")
            if summary:
                md.append(f"{summary}\n")
            md.append("\n")
        return "".join(md)

    def _render_github_trending(self, repos: List[Dict[str, Any]]) -> str:
        """Render GitHub trending repos section."""
        md = ["## GitHub Trending\n\n"]
        for repo in repos:
            title = repo.get("title", "")
            link = repo.get("link", "")
            summary = repo.get("summary", "")
            stars = repo.get("stars", "")
            language = repo.get("language", "")
            tags = []
            if stars:
                tags.append(f"{stars} stars")
            if language:
                tags.append(language)
            tag_str = f" ({', '.join(tags)})" if tags else ""
            if link:
                md.append(f"**[{title}]({link})**{tag_str}\n")
            else:
                md.append(f"**{title}**{tag_str}\n")
            if summary:
                md.append(f"{summary}\n")
            md.append("\n")
        return "".join(md)

    def _render_community_picks(self, items: List[Dict[str, Any]]) -> str:
        """Render scored community & newsletter picks section."""
        max_items = self._get_limit("max_community_picks", 8)
        sorted_items = sorted(
            items, key=lambda x: x.get("score_combined", 0), reverse=True
        )
        # Score gate: only filter when scores are present (intelligence enabled)
        if any(i.get("score_combined") for i in sorted_items):
            sorted_items = [
                i for i in sorted_items if i.get("score_combined", 0) >= 3
            ]
        sorted_items = sorted_items[:max_items]

        if not sorted_items:
            return ""

        md = ["## Community & Newsletter Picks\n\n"]
        for item in sorted_items:
            title = item.get("title") or item.get("name") or "Link"
            url = item.get("url") or item.get("link") or ""
            source = item.get("source", "")
            score = item.get("score_combined")
            summary = self._clean_summary(
                item.get("brief_summary") or item.get("summary") or "",
                title, source,
            )

            source_type = item.get("source_type", "")
            if source_type == "github":
                stars = item.get("stars", "")
                lang = item.get("language", "")
                parts = ["GitHub"]
                if stars:
                    parts.append(f"{stars} stars")
                if lang:
                    parts.append(lang)
                tag = f" ({', '.join(parts)})"
            else:
                tag = f" *(via {source})*" if source else ""

            score_tag = f" {self._render_stars(score)}" if score else ""
            if url:
                md.append(f"**[{title}]({url})**{tag}{score_tag}\n")
            else:
                md.append(f"**{title}**{tag}{score_tag}\n")
            if summary:
                md.append(f"{summary}\n")
            md.append("\n")
        return "".join(md)

    def _render_newsletters(self, newsletters: List[Dict[str, Any]]) -> str:
        """Render newsletter highlights section."""
        max_newsletters = self._get_limit("max_newsletters_render", 10)
        md = ["## Newsletter Highlights\n\n"]
        for item in newsletters[:max_newsletters]:
            title = item.get("title", "Untitled")
            source = item.get("source", "")
            summary = item.get("summary", "")
            category = item.get("category", "")
            category_tag = f" [{category}]" if category else ""
            md.append(f"**{title}** *({source})*{category_tag}\n")
            if summary:
                md.append(f"{summary}\n")
            md.append("\n")
        return "".join(md)

    def _ensure_paper_summaries(
        self, papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensure each paper has a brief_summary and score. Batch-generate missing ones."""
        missing = [
            (i, p) for i, p in enumerate(papers)
            if not (p.get("brief_summary") and p.get("score_combined"))
        ]
        if not missing or not self.intelligence.available:
            return papers

        paper_texts = []
        indices = []
        for i, paper in missing:
            title = paper.get("title", "")
            abstract = paper.get("summary", "")[:600]
            if not abstract:
                continue
            paper_texts.append(f"[{len(paper_texts)+1}] {title}\n{abstract}")
            indices.append(i)

        if not paper_texts:
            return papers

        papers_block = "\n\n".join(paper_texts)
        prompt = (
            "For each paper, write a 2-3 sentence summary of its key contribution "
            "and rate it.\n\n"
            f"<papers>\n{papers_block}\n</papers>\n\n"
            "Respond in this exact format for each paper:\n"
            "[number] SCORE:X/5 Your 2-3 sentence summary here.\n\n"
            "SCORE is a combined rating (1-5) of impact, complexity, and innovation. "
            "5 = groundbreaking, 1 = routine.\n"
            "Be factual. Do not add information not in the abstract."
        )
        result = self.intelligence.bedrock.invoke(
            prompt, tier="light", max_tokens=500
        )
        if not result:
            return papers

        parsed = self.intelligence._parse_ranked_response(result)
        for rank_idx, text in parsed:
            if 0 <= rank_idx < len(indices):
                paper_idx = indices[rank_idx]
                score, summary = self.intelligence.extract_score(text)
                papers[paper_idx]["brief_summary"] = summary
                if score:
                    papers[paper_idx]["score_combined"] = score
                logger.info(
                    f"Generated summary+score for: {papers[paper_idx].get('title', '')[:50]}"
                )
        return papers

    def _render_top_papers(self, top_papers: List[Dict[str, Any]]) -> str:
        """Render top papers section with summaries, scores, and repro assessment."""
        max_top = self._get_limit("max_top_papers_render", 3)
        md = ["## Top Papers\n\n"]
        sorted_papers = sorted(top_papers[:max_top], key=lambda x: x.get("score_combined", 0), reverse=True)
        # Only filter by score when scores are present (Bedrock enabled)
        if any(p.get("score_combined") for p in sorted_papers):
            sorted_papers = [p for p in sorted_papers if p.get("score_combined", 0) >= 3]
        for i, paper in enumerate(sorted_papers, 1):
            paper_title = paper.get("title", "")
            authors = paper.get("authors", [])
            arxiv_url = paper.get("arxiv_url", "")
            brief_summary = paper.get("brief_summary", "")
            relevance_reason = paper.get("relevance_reason", "")
            score = paper.get("score_combined")
            repro_total = paper.get("repro_total")
            repro_verdict = paper.get("repro_verdict", "")
            difficulty = paper.get("reproduction_difficulty", "")

            score_tag = f" {self._render_stars(score)}" if score else ""
            if arxiv_url:
                md.append(f"### {i}. [{paper_title}]({arxiv_url}){score_tag}\n")
            else:
                md.append(f"### {i}. {paper_title}{score_tag}\n")
            if authors:
                md.append(f"*{', '.join(authors[:3])}*\n\n")

            if brief_summary:
                md.append(f"{brief_summary}\n\n")
            elif relevance_reason:
                md.append(f"{relevance_reason}\n\n")

            # Show reproduction feasibility badge
            if repro_total is not None:
                badge = "✅" if repro_total >= 18 else "🟡" if repro_total >= 12 else "🔴"
                md.append(f"**Repro: {badge} {repro_total}/25** ({difficulty})")
                if repro_verdict:
                    md.append(f" — {repro_verdict}")
                md.append("\n")

            md.append("\n\n")
        return "".join(md)

    def _render_papers(self, papers: List[Dict[str, Any]]) -> str:
        """Render recent papers section (compact)."""
        max_papers = self._get_limit("max_papers_render", 5)
        md = ["## Recent Papers\n\n"]
        for paper in papers[:max_papers]:
            paper_title = paper.get("title", "")
            authors = paper.get("authors", [])
            arxiv_url = paper.get("arxiv_url", "")
            brief_summary = paper.get("brief_summary", "")

            md.append(f"**{paper_title}**")
            if authors:
                md.append(f" *{', '.join(authors[:2])}*")
            if arxiv_url:
                md.append(f" [arxiv]({arxiv_url})")
            md.append("\n")
            if brief_summary:
                md.append(f"{brief_summary}\n")
            md.append("\n")
        return "".join(md)

    def generate_pdf(self, markdown_content: str, output_path: str) -> bool:
        """Generate PDF from markdown."""
        try:
            logger.info("=== Generating PDF ===")
            pdf_config = self.config.get("pdf", {})
            page_format = self.config.get("output_format", "kindle")
            font_size = pdf_config.get("font_size", 10)
            line_spacing = pdf_config.get("line_spacing", 1.5)

            generator = PDFGenerator(
                page_format=page_format,
                font_size=font_size,
                line_spacing=line_spacing,
            )
            generator.generate_pdf(markdown_content, output_path)
            self.status["pdf_generated"] = True
            return True

        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            self.errors.append(f"PDF generation: {e}")
            return False

    def distribute_briefing(
        self, markdown_content: str, pdf_path: Optional[str], subject: str
    ) -> Dict[str, bool]:
        """
        Distribute briefing to all configured channels.

        Sends PDF to Kindle + rich HTML to email recipients.

        Args:
            markdown_content: Markdown briefing content.
            pdf_path: Path to generated PDF.
            subject: Email subject / filename.

        Returns:
            Dictionary mapping channel -> success boolean.
        """
        if self.dry_run:
            logger.info("Dry run: Skipping all distribution")
            return {}

        sender_email = os.environ.get("GMAIL_USER")
        sender_password = os.environ.get("GMAIL_APP_PASSWORD")

        if not sender_email or not sender_password:
            logger.warning("Gmail credentials not set, skipping distribution")
            return {}

        try:
            distributor = EmailDistributor(
                sender_email=sender_email,
                sender_password=sender_password,
            )

            results = distributor.distribute(
                config=self.config,
                markdown_content=markdown_content,
                pdf_path=pdf_path,
                subject=subject,
                dry_run=self.dry_run,
            )

            # Update status
            sent_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            self.status["email_sent"] = sent_count > 0
            self.status["distribution"] = {
                "sent": sent_count,
                "total": total_count,
                "details": results,
            }

            logger.info(f"Distribution: {sent_count}/{total_count} channels delivered")
            return results

        except Exception as e:
            logger.error(f"Distribution failed: {e}")
            self.errors.append(f"Distribution: {e}")
            return {}

    def publish_to_obsidian(
        self,
        markdown_content: str,
        date: datetime,
        top_papers: List[Dict[str, Any]],
        emerging_themes: List[str],
        entity_mentions: List[Dict[str, Any]],
        trending_topics: Dict[str, Any],
        weekly_deep_dive: str,
        briefing_name: str,
        weekly_items: List[Dict[str, Any]],
        podcast_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish briefing artifacts to Obsidian vault."""
        obsidian_config = self.config.get("obsidian", {})
        if not obsidian_config.get("enabled", False):
            return {}

        api_url = obsidian_config.get("api_url", "http://localhost:27123")
        api_key = os.environ.get("OBSIDIAN_API_KEY", "")
        if not api_key:
            logger.warning("OBSIDIAN_API_KEY not set, skipping Obsidian publish")
            return {}

        if self.dry_run:
            logger.info("Dry run: Skipping Obsidian publish")
            return {}

        try:
            writer = ObsidianWriter(api_url, api_key, obsidian_config)
            weekly_briefing_names = list({
                f"Personal-Briefing-{item['date']}" for item in weekly_items
            }) if weekly_items else []

            results = writer.publish(
                markdown_content=markdown_content,
                date=date,
                status=self.status,
                emerging_themes=emerging_themes,
                top_papers=top_papers,
                entity_mentions=entity_mentions,
                trending_topics=trending_topics,
                weekly_deep_dive=weekly_deep_dive,
                briefing_name=briefing_name,
                weekly_briefing_names=weekly_briefing_names,
                podcast_url=podcast_url,
            )
            self.status["obsidian"] = results
            logger.info(
                f"Obsidian publish: briefing={'ok' if results.get('briefing') else 'fail'}"
            )
            return results

        except Exception as e:
            logger.error(f"Obsidian publish failed: {e}")
            self.errors.append(f"Obsidian publish: {e}")
            return {}

    @staticmethod
    def _inject_podcast_section(markdown_content: str, podcast_url: str) -> str:
        """Append an Audio Overview section with the NotebookLM share link."""
        section = (
            "\n\n## Audio Overview\n\n"
            f"[Listen on NotebookLM]({podcast_url}) "
            "*(audio generating — ready in ~10 min)*\n"
        )
        return markdown_content.rstrip() + section

    _URL_NOISE = re.compile(
        r'(unsubscribe|track|pixel|click\.|\?utm_|mailto:|/login|/signin'
        r'|twitter\.com|x\.com|t\.co|linkedin\.com|facebook\.com|instagram\.com)',
        re.IGNORECASE,
    )

    @staticmethod
    def _extract_urls_from_text(text: str) -> List[str]:
        """Extract article URLs from newsletter text, filtering noise."""
        if not text:
            return []
        raw = re.findall(r'https?://[^\s<>"\'\)\]\}]+', text)
        seen: set = set()
        out: List[str] = []
        for url in raw:
            url = url.rstrip('.,;:')
            if url in seen or len(url) < 20:
                continue
            seen.add(url)
            if BriefingRunner._URL_NOISE.search(url):
                continue
            out.append(url)
        return out[:10]

    def _get_limit(self, key: str, default: int) -> int:
        """Get a render limit from env var BRIEFING_{KEY} or config, with fallback."""
        env_val = os.environ.get(f"BRIEFING_{key.upper()}")
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                pass
        return self.config.get(key, default)

    def save_status(self, output_dir: str = ".") -> None:
        """
        Save run status to JSON file for monitoring.

        Args:
            output_dir: Directory to save status file.
        """
        self.status["errors"] = self.errors
        status_path = Path(output_dir) / "status.json"
        try:
            with open(status_path, "w") as f:
                json.dump(self.status, f, indent=2)
            logger.info(f"Status saved: {status_path}")
        except IOError as e:
            logger.warning(f"Failed to save status: {e}")

    @staticmethod
    def _load_previous_state() -> Dict[str, Any]:
        """Load previous briefing state for cross-day trend tracking."""
        state_path = Path(STATE_FILENAME)
        if state_path.exists():
            try:
                with open(state_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    @staticmethod
    def _save_state(
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        stocks: List[Dict[str, Any]],
        emerging_themes: List[str],
        trending_topics: Optional[Dict[str, Any]] = None,
        weekly_items: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Save current briefing state for next run's trend tracking and dedup."""
        state = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "top_paper_titles": [p.get("title", "") for p in papers[:10]],
            "top_blog_titles": [b.get("title", "") for b in blogs[:10]],
            "top_news_titles": [n.get("title", "") for n in news[:10]],
            "stock_closes": {
                s.get("symbol", ""): s.get("current_price", 0)
                for s in stocks if "error" not in s
            },
            "emerging_themes": emerging_themes,
        }
        # Feature 1: Save trending topics
        if trending_topics is not None:
            state["trending_topics"] = trending_topics
        # Feature 2: Save weekly items for Saturday deep dive
        if weekly_items is not None:
            state["weekly_items"] = weekly_items
        try:
            with open(STATE_FILENAME, "w") as f:
                json.dump(state, f, indent=2)
        except IOError:
            pass

    def run(self) -> int:
        """
        Run the complete briefing pipeline.

        Returns:
            Exit code (0=success, 1=partial failure, 2=total failure).
        """
        start_time = time.time()
        logger.info("=== Starting Morning Briefing ===")
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        with self._tracer.start_as_current_span("personal.briefing.run") as root_span:
            root_span.set_attribute("briefing.date", today_str)
            root_span.set_attribute("briefing.dry_run", self.dry_run)

            # --- Load previous state for cross-day tracking ---
            previous_state = self._load_previous_state()

            # --- Topic expansion (intelligence layer) ---
            topics = self.config.get("arxiv_topics", [])
            if self.intelligence.available:
                logger.info("=== Intelligence Layer: Expanding Topics ===")
                topics = self.intelligence.expand_topics(topics)

            # --- Run scanners in parallel (papers + blogs + stocks + newsletters) ---
            from concurrent.futures import ThreadPoolExecutor
            with self._tracer.start_as_current_span("personal.fetch"):
                logger.info("=== Parallel data fetch (papers/blogs/stocks/newsletters) ===")
                with ThreadPoolExecutor(max_workers=5) as pool:
                    fut_papers = pool.submit(self.run_arxiv_scan, topics)
                    fut_blogs = pool.submit(self.run_blog_scan)
                    fut_stocks = pool.submit(self.run_stock_fetch)
                    fut_newsletters = pool.submit(self.run_newsletter_scan)
                    fut_github = pool.submit(self.run_github_trending_scan)

                    papers = fut_papers.result()
                    blogs = fut_blogs.result()
                    stocks = fut_stocks.result()
                    newsletters = fut_newsletters.result()
                    github_trending = fut_github.result()

            # --- Generate dynamic news queries (intelligence layer) ---
            news_queries = self.config.get("news_queries", [])
            if self.intelligence.available:
                logger.info("=== Intelligence Layer: Generating Dynamic Queries ===")
                news_queries = self.intelligence.generate_dynamic_queries(
                    previous_state, news_queries
                )

            with self._tracer.start_as_current_span("personal.fetch.news"):
                news = self.run_news_aggregation(queries=news_queries)

            # --- Cross-section deduplication ---
            news, blogs = self.deduplicate_news_and_blogs(news, blogs)

            # --- Deduplicate similar papers by title ---
            papers = self.deduplicate_similar_papers(papers)

            # --- Cross-day deduplication (skip items from yesterday) ---
            papers, blogs, news = self._dedup_against_previous(
                papers, blogs, news, previous_state
            )

            # --- Build community picks: newsletter-extracted links + GitHub repos ---
            community_picks: List[Dict[str, Any]] = []
            for nl in newsletters:
                text = (nl.get("snippet") or "") + " " + (nl.get("summary") or "")
                for url in self._extract_urls_from_text(text):
                    community_picks.append({
                        "url": url,
                        "title": nl.get("title", "Newsletter article"),
                        "source": nl.get("source", "Newsletter"),
                        "summary": "",
                        "source_type": "newsletter",
                    })
            for repo in github_trending:
                if repo.get("link"):
                    community_picks.append({
                        "url": repo["link"],
                        "title": repo.get("title", ""),
                        "name": repo.get("name", ""),
                        "description": repo.get("description", ""),
                        "summary": repo.get("summary", ""),
                        "source": "GitHub Trending",
                        "stars": repo.get("stars", ""),
                        "language": repo.get("language", ""),
                        "source_type": "github",
                    })
            if community_picks:
                logger.info(
                    f"Built {len(community_picks)} community picks "
                    f"({len([c for c in community_picks if c['source_type'] == 'newsletter'])} newsletter, "
                    f"{len([c for c in community_picks if c['source_type'] == 'github'])} GitHub)"
                )

            # --- Intelligence layer: enrich data ---
            synthesis = {}
            emerging_themes = []
            if self.intelligence.available:
                with self._tracer.start_as_current_span("personal.enrich"):
                    logger.info("=== Intelligence Layer: Enriching Data ===")
                    from concurrent.futures import ThreadPoolExecutor

                    # Two-stage relevance filtering (NEW) — must run first (reduces paper count)
                    interest_profile = self.config.get("interest_profile")
                    if interest_profile:
                        logger.info("=== Intelligence Layer: Stage 1 Relevance Filtering ===")
                        papers = self.intelligence.filter_papers_by_relevance(papers, interest_profile)

                    # --- Parallel batch 1: papers, news, blogs, community_picks ---
                    logger.info("=== Intelligence Layer: Parallel enrichment (papers/news/blogs/community) ===")
                    interest_topics = self.config.get("interest_profile", [])
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        fut_papers = pool.submit(self._enrich_papers, papers, topics)
                        fut_news = pool.submit(self.intelligence.rank_and_summarize_news, news, topics)
                        fut_blogs = pool.submit(self.intelligence.rank_and_summarize_blogs, blogs, topics)
                        fut_community = pool.submit(
                            self.intelligence.rank_source_links,
                            community_picks, interest_topics,
                        )

                        papers = fut_papers.result()
                        news = fut_news.result()
                        blogs = fut_blogs.result()
                        community_picks = fut_community.result()

                    # --- Parallel batch 2: stocks + themes (both depend on news) ---
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        fut_stocks = pool.submit(self.intelligence.correlate_stocks_and_news, stocks, news)
                        fut_themes = pool.submit(
                            self.intelligence.detect_emerging_themes,
                            papers, blogs, news, newsletters, github_trending,
                        )

                        stocks = fut_stocks.result()
                        emerging_themes = fut_themes.result()

                    # Track trending topics across days (Feature 1)
                    previous_state, papers, blogs, news = self.intelligence.track_trending(
                        papers, blogs, news, previous_state, newsletters, github_trending,
                    )

            # --- Market trend analysis (must happen after correlation) ---
            market_trend = ""
            if self.intelligence.available and stocks:
                market_trend = self._analyze_market_trend(stocks)

            # --- Score papers (combines TF-IDF + semantic if available) ---
            top_papers = self.score_papers(papers)

            # --- Intelligence layer: assess top papers & synthesize ---
            if self.intelligence.available:
                with self._tracer.start_as_current_span("personal.synthesize"):
                    top_papers = self.intelligence.assess_reproduction_feasibility(top_papers)

                    # Ensure top 3 papers all have summaries (batched)
                    top_papers = self._ensure_paper_summaries(top_papers[:3]) + top_papers[3:]

                    synthesis = self.intelligence.synthesize_briefing(
                        papers, blogs[:5], stocks, news[:5], top_papers[:3],
                        emerging_themes=emerging_themes,
                        previous_state=previous_state,
                        newsletters=newsletters,
                        github_trending=github_trending,
                    )

                    # Feature 3: Competitive Intelligence (entity tracking)
                    tracked_entities = self.config.get("tracked_entities", [])
                    entity_mentions = []
                    if tracked_entities:
                        logger.info("=== Intelligence Layer: Entity Tracking ===")
                        entity_mentions = self.intelligence.detect_entity_mentions(
                            papers, blogs, news, tracked_entities,
                            newsletters, github_trending,
                        )
                        # Add to synthesis for rendering in Executive Summary
                        synthesis["entity_mentions"] = entity_mentions

            # Feature 2: Weekly Deep Dive (accumulate items & generate on Saturday)
            is_saturday = now.weekday() == 5
            weekly_deep_dive = ""
            weekly_items = previous_state.get("weekly_items", [])

            # Accumulate today's top items for the week
            for paper in top_papers[:3]:
                weekly_items.append({
                    "date": today_str,
                    "type": "paper",
                    "title": paper.get("title", ""),
                })
            for article in news[:3]:
                weekly_items.append({
                    "date": today_str,
                    "type": "news",
                    "title": article.get("title", ""),
                })

            # On Saturday, generate the deep dive and clear weekly_items
            if is_saturday and self.intelligence.available and weekly_items:
                logger.info("=== Intelligence Layer: Weekly Deep Dive (Saturday) ===")
                weekly_deep_dive = self.intelligence.generate_weekly_deep_dive(weekly_items)
                # Clear weekly items after generation
                weekly_items = []

            # --- Check if we have any data ---
            has_data = any([papers, blogs, stocks, news, newsletters, github_trending])
            if not has_data:
                logger.error("No data collected from any source")
                self.status["elapsed_seconds"] = round(time.time() - start_time, 1)
                self.save_status()
                root_span.set_status(trace.StatusCode.ERROR, "No data collected")
                return 2

            podcast_url: Optional[str] = None
            with self._tracer.start_as_current_span("personal.render"):
                # --- Generate markdown briefing ---
                filename = self._format_filename(now)
                self._briefing_title = filename
                markdown_content = self.generate_markdown_briefing(
                    papers, blogs, stocks, news, top_papers, synthesis,
                    market_trend=market_trend,
                    weekly_deep_dive=weekly_deep_dive,
                    newsletters=newsletters,
                    community_picks=community_picks,
                )

                # --- Generate NotebookLM podcast URL (fire-and-forget, ~10-20s) ---
                if self.podcast_generator.enabled:
                    # Collect source URLs from items that made it into the
                    # rendered briefing (mirror render-method filter logic).
                    _max_top = self._get_limit("max_top_papers_render", 3)
                    _rendered_top = sorted(
                        top_papers[:_max_top],
                        key=lambda x: x.get("score_combined", 0), reverse=True,
                    )
                    if any(p.get("score_combined") for p in _rendered_top):
                        _rendered_top = [p for p in _rendered_top if p.get("score_combined", 0) >= 3]

                    _max_blogs = self._get_limit("max_blogs_render", 5)
                    _rendered_blogs = sorted(
                        blogs[:_max_blogs],
                        key=lambda x: x.get("score_combined", 0), reverse=True,
                    )
                    if any(b.get("score_combined") for b in _rendered_blogs):
                        _rendered_blogs = [b for b in _rendered_blogs if b.get("score_combined", 0) >= 3]

                    _max_news = self._get_limit("max_news_render", 5)
                    _rendered_news = news[:_max_news]

                    _max_community = self._get_limit("max_community_picks", 8)
                    _rendered_community = sorted(
                        community_picks,
                        key=lambda x: x.get("score_combined", 0), reverse=True,
                    )
                    if any(c.get("score_combined") for c in _rendered_community):
                        _rendered_community = [
                            c for c in _rendered_community
                            if c.get("score_combined", 0) >= 3
                        ]
                    _rendered_community = _rendered_community[:_max_community]

                    source_urls = (
                        [p.get("url") or p.get("arxiv_url") for p in _rendered_top
                         if p.get("url") or p.get("arxiv_url")]
                        + [b["link"] for b in _rendered_blogs if b.get("link")]
                        + [n["url"] for n in _rendered_news if n.get("url")]
                        + [c.get("url") or c.get("link") for c in _rendered_community
                           if c.get("url") or c.get("link")]
                    )[:10]

                    podcast_url = self.podcast_generator.generate(
                        markdown_content, now, source_urls
                    )
                    if podcast_url:
                        markdown_content = self._inject_podcast_section(
                            markdown_content, podcast_url
                        )
                        self.status["podcast_url"] = podcast_url
                        logger.info(f"Podcast URL injected into briefing: {podcast_url}")

                # --- Save markdown ---
                md_path = f"{filename}.md"
                try:
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    logger.info(f"Saved markdown: {md_path}")
                except IOError as e:
                    logger.warning(f"Failed to save markdown: {e}")

                # --- Generate PDF (skip for HTML-only mode) ---
                pdf_path = None
                if self.config.get("output_format", "kindle") != "html":
                    pdf_path = f"{filename}.pdf"
                    pdf_success = self.generate_pdf(markdown_content, pdf_path)

                    if not pdf_success:
                        logger.error("Failed to generate PDF")
                        self.status["elapsed_seconds"] = round(time.time() - start_time, 1)
                        self.save_status()
                        root_span.set_status(trace.StatusCode.ERROR, "PDF generation failed")
                        return 2

            # --- Distribute to all channels ---
            self.distribute_briefing(markdown_content, pdf_path, filename)

        # --- Publish to Obsidian vault ---
        self.publish_to_obsidian(
            markdown_content=markdown_content,
            date=now,
            top_papers=top_papers,
            emerging_themes=emerging_themes,
            entity_mentions=synthesis.get("entity_mentions", []),
            trending_topics=previous_state.get("trending_topics", {}),
            weekly_deep_dive=weekly_deep_dive,
            briefing_name=filename,
            weekly_items=weekly_items,
            podcast_url=podcast_url,
        )

        # --- Mark newsletters as digested in Supabase ---
        ns_config = self.config.get("newsletter_source", {})
        if self.newsletter_scanner and ns_config.get("mark_digested", True):
            self.newsletter_scanner.mark_digested()

        # --- Save state for cross-day tracking ---
        # Save updated trending_topics and weekly_items from current run
        self._save_state(
            top_papers, blogs, news, stocks, emerging_themes,
            trending_topics=previous_state.get("trending_topics", {}),
            weekly_items=weekly_items  # Use updated weekly_items from this run
        )

        # --- Finalize ---
        elapsed = time.time() - start_time
        self.status["elapsed_seconds"] = round(elapsed, 1)
        self.save_status()

        logger.info(f"=== Briefing Complete in {elapsed:.1f}s ===")

        if self.errors:
            logger.warning(f"Completed with {len(self.errors)} errors")
            return 1
        else:
            logger.info("Completed successfully")
            return 0


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(2)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        sys.exit(2)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate morning briefing")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate briefing but don't send email",
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

    # Validate config
    is_valid, messages = validate_config(config)
    if not is_valid:
        logger.error("Configuration is invalid. Fix errors above and retry.")
        return 2

    # Check environment
    check_environment(config, dry_run=args.dry_run)

    # Run briefing
    runner = BriefingRunner(config=config, dry_run=args.dry_run)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
