#!/usr/bin/env python3
"""
Morning briefing runner v0.2 - Coordinator + Parallel Workers Architecture.

The coordinator spawns self-contained workers in parallel, reads all findings,
then runs deduplication, intelligence enrichment, synthesis, rendering, and
distribution as coordinator-level stages.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.workers.papers_worker import PapersWorker
from scripts.workers.blogs_worker import BlogsWorker
from scripts.workers.news_market_worker import NewsMarketWorker
from scripts.workers.gmail_worker import GmailWorker
from scripts.workers.github_trending_worker import GitHubTrendingWorker
from scripts.llm_client import LLMClient
from scripts.intelligence import BriefingIntelligence
from scripts.paper_scorer import PaperScorer
from scripts.news_aggregator import NewsAggregator
from scripts.pdf_generator import PDFGenerator
from scripts.email_distributor import EmailDistributor
from scripts.obsidian_writer import ObsidianWriter
from scripts.podcast_generator import PodcastGenerator
from scripts.config_validator import validate_config, check_environment
from scripts.utils import load_config
from opentelemetry import trace
from scripts.tracing import setup_tracing, get_tracer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STATE_FILENAME = ".personal-state.json"


class _SupabaseStateStore:
    """Supabase-backed state store with local-file fallback for dev.

    Uses SUPABASE_URL + SUPABASE_SERVICE_KEY env vars.  When either is absent
    the store transparently falls back to STATE_FILENAME on disk so local
    development works without a database.
    """

    TABLE = "briefing_state"

    def __init__(self) -> None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        self._client = None
        if url and key:
            try:
                from supabase import create_client  # type: ignore

                self._client = create_client(url, key)
            except Exception as exc:  # pragma: no cover
                logger.warning("Supabase init failed (%s) — using file fallback", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """Return the most-recent state row, or {} if none exists."""
        if self._client is not None:
            try:
                resp = (
                    self._client.table(self.TABLE)
                    .select("*")
                    .order("date", desc=True)
                    .limit(1)
                    .execute()
                )
                rows = resp.data
                if rows:
                    return self._from_row(rows[0])
            except Exception as exc:
                logger.warning("Supabase load failed (%s) — using file fallback", exc)

        # File fallback
        state_path = Path(STATE_FILENAME)
        if state_path.exists():
            try:
                with open(state_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def save(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        stocks: List[Dict[str, Any]],
        emerging_themes: List[str],
        trending_topics: Optional[Dict[str, Any]] = None,
        weekly_items: Optional[List[Dict[str, Any]]] = None,
        github_trending: Optional[List[Dict[str, Any]]] = None,
        email_sent_date: Optional[str] = None,
    ) -> None:
        """Upsert state. Any field passed as None is omitted from the update."""
        today = datetime.now().strftime("%Y-%m-%d")
        state: Dict[str, Any] = {
            "date": today,
            "top_paper_titles": [p.get("title", "") for p in papers[:10]],
            "top_blog_titles": [b.get("title", "") for b in blogs[:10]],
            "top_news_titles": [n.get("title", "") for n in news[:10]],
            "top_github_titles": [r.get("title", "") for r in (github_trending or [])[:20]],
            "stock_closes": {
                s.get("symbol", ""): s.get("current_price", 0)
                for s in stocks
                if "error" not in s
            },
            "emerging_themes": emerging_themes,
        }
        if trending_topics is not None:
            state["trending_topics"] = trending_topics
        if weekly_items is not None:
            state["weekly_items"] = weekly_items
        if email_sent_date is not None:
            state["email_sent_date"] = email_sent_date

        if self._client is not None:
            try:
                self._client.table(self.TABLE).upsert(state).execute()
                return
            except Exception as exc:
                logger.warning("Supabase save failed (%s) — using file fallback", exc)

        # File fallback
        try:
            with open(STATE_FILENAME, "w") as f:
                json.dump(state, f, indent=2)
        except IOError:
            pass

    def mark_email_sent(self, date: str) -> None:
        """Record that the email was sent for *date* (YYYY-MM-DD)."""
        if self._client is not None:
            try:
                self._client.table(self.TABLE).upsert(
                    {"date": date, "email_sent_date": date}, on_conflict="date"
                ).execute()
                return
            except Exception as exc:
                logger.warning("Supabase mark_email_sent failed (%s) — using file fallback", exc)

        # File fallback — patch existing file in-place
        try:
            existing: Dict[str, Any] = {}
            state_path = Path(STATE_FILENAME)
            if state_path.exists():
                with open(state_path) as f:
                    existing = json.load(f)
            existing["email_sent_date"] = date
            with open(STATE_FILENAME, "w") as f:
                json.dump(existing, f, indent=2)
        except (json.JSONDecodeError, IOError):
            pass

    def email_sent_today(self, today: str) -> bool:
        """Return True if an email was already sent for *today*."""
        if self._client is not None:
            try:
                resp = (
                    self._client.table(self.TABLE)
                    .select("email_sent_date")
                    .eq("date", today)
                    .limit(1)
                    .execute()
                )
                rows = resp.data
                if rows and rows[0].get("email_sent_date") == today:
                    return True
                return False
            except Exception as exc:
                logger.warning("Supabase email_sent_today failed (%s) — using file fallback", exc)

        # File fallback
        try:
            state_path = Path(STATE_FILENAME)
            if state_path.exists():
                with open(state_path) as f:
                    return json.load(f).get("email_sent_date") == today
        except (json.JSONDecodeError, IOError):
            pass
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a Supabase row back to the legacy state dict shape."""
        out = dict(row)
        # email_sent_date may be a date object from the driver — coerce to str
        if out.get("email_sent_date") and not isinstance(out["email_sent_date"], str):
            out["email_sent_date"] = str(out["email_sent_date"])
        return out


class BriefingCoordinator:
    """
    Coordinator for v0.2 multi-agent briefing generation.

    Spawns parallel workers for data fetching, then coordinates deduplication,
    intelligence enrichment, synthesis, rendering, and distribution.
    """

    DEFAULT_SECTION_ORDER = [
        "stocks", "news", "community_picks", "newsletters", "top_papers", "blogs",
    ]

    def __init__(
        self,
        config: Dict[str, Any],
        dry_run: bool = False,
        run_date: Optional[datetime] = None,
    ):
        self.config = config
        self.dry_run = dry_run
        self.run_date = run_date
        self.errors: List[str] = []

        # LLM + intelligence
        llm_config = config.get("llm", config.get("bedrock", {}))
        self.llm = LLMClient(llm_config)
        self.intelligence = BriefingIntelligence(self.llm, config)

        # Podcast
        self.podcast_generator = PodcastGenerator(config.get("podcast", {}))

        # State store (Supabase-backed, file fallback)
        self._state_store = _SupabaseStateStore()

        # OTel tracing
        setup_tracing(config)
        self._tracer = get_tracer()

        # Status tracking
        self.status: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "papers_found": 0,
            "blogs_found": 0,
            "stocks_fetched": 0,
            "news_found": 0,
            "newsletters_found": 0,
            "github_trending_found": 0,
            "intelligence_enabled": self.intelligence.available,
            "errors": [],
            "pdf_generated": False,
            "email_sent": False,
            "elapsed_seconds": 0,
        }

    # ------------------------------------------------------------------
    # Worker orchestration
    # ------------------------------------------------------------------

    def _spawn_workers(self) -> List[Dict[str, Any]]:
        """Spawn all workers in parallel and collect findings."""
        workers = [
            PapersWorker(self.config, ref_date=self.run_date),
            BlogsWorker(self.config, ref_date=self.run_date),
            NewsMarketWorker(self.config, ref_date=self.run_date),
            GmailWorker(self.config, ref_date=self.run_date),
            GitHubTrendingWorker(self.config, ref_date=self.run_date),
        ]

        # Keep reference to GmailWorker for mark_digested()
        self._gmail_worker = workers[3]

        findings: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = {executor.submit(w.run): w for w in workers}
            for future in as_completed(futures):
                worker = futures[future]
                try:
                    finding = future.result()
                    findings.append(finding)
                    logger.info(
                        f"[{finding['worker']}] completed in "
                        f"{finding['metadata']['processing_time']:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"Worker {worker.worker_name} raised exception: {e}")
                    findings.append({
                        "worker": worker.worker_name,
                        "status": "error",
                        "items": [],
                        "metadata": {
                            "processing_time": 0, "token_count": 0,
                            "items_found": 0, "items_kept": 0,
                        },
                        "synthesis": "",
                        "error": str(e),
                    })
        return findings

    def _extract_items(self, findings: List[Dict[str, Any]]) -> tuple:
        """Extract (papers, blogs, news, stocks, newsletters, github_trending) from findings."""
        papers: list = []
        blogs: list = []
        news: list = []
        stocks: list = []
        newsletters: list = []
        github_trending: list = []

        for f in findings:
            if f["worker"] == "papers_worker":
                papers = f.get("items", [])
            elif f["worker"] == "blogs_worker":
                blogs = f.get("items", [])
            elif f["worker"] == "news_market_worker":
                items = f.get("items", {})
                news = items.get("news", []) if isinstance(items, dict) else []
                stocks = items.get("stocks", []) if isinstance(items, dict) else []
            elif f["worker"] == "gmail_worker":
                newsletters = f.get("items", [])
            elif f["worker"] == "github_trending_worker":
                github_trending = f.get("items", [])

        return papers, blogs, news, stocks, newsletters, github_trending

    # ------------------------------------------------------------------
    # Deduplication (migrated from v1)
    # ------------------------------------------------------------------

    @staticmethod
    def deduplicate_news_and_blogs(
        news: List[Dict[str, Any]], blogs: List[Dict[str, Any]]
    ) -> tuple:
        """Remove duplicate content between news and blog sections."""
        blog_domains: set = set()
        blog_titles_lower: set = set()
        for blog in blogs:
            link = blog.get("link", "")
            if link:
                try:
                    blog_domains.add(urlparse(link).netloc.lower())
                except Exception:
                    pass
            title = blog.get("title", "").lower().strip()
            if title:
                blog_titles_lower.add(title)

        deduped = []
        for article in news:
            url = article.get("url", "")
            title = article.get("title", "").lower().strip()
            if title and title in blog_titles_lower:
                continue
            if url:
                try:
                    if urlparse(url).netloc.lower() in blog_domains:
                        continue
                except Exception:
                    pass
            deduped.append(article)

        removed = len(news) - len(deduped)
        if removed:
            logger.info(f"Dedup: removed {removed} news articles duplicated in blogs")
        return deduped, blogs

    @staticmethod
    def deduplicate_similar_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove papers with very similar titles (>85% match)."""
        if len(papers) <= 1:
            return papers
        deduped = []
        for paper in papers:
            title = paper.get("title", "").lower()
            if not any(
                SequenceMatcher(None, title, k.get("title", "").lower()).ratio() > 0.85
                for k in deduped
            ):
                deduped.append(paper)
        removed = len(papers) - len(deduped)
        if removed:
            logger.info(f"Dedup: removed {removed} near-duplicate papers")
        return deduped

    @staticmethod
    def _dedup_against_previous(
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        previous_state: Dict[str, Any],
        github_trending: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple:
        """Remove items that appeared in yesterday's briefing."""
        if not previous_state:
            if github_trending is not None:
                return papers, blogs, news, github_trending
            return papers, blogs, news
        prev_papers = {t.lower() for t in previous_state.get("top_paper_titles", [])}
        prev_blogs = {t.lower() for t in previous_state.get("top_blog_titles", [])}
        prev_news = {t.lower() for t in previous_state.get("top_news_titles", [])}
        prev_github = {t.lower() for t in previous_state.get("top_github_titles", [])}

        def _filter(items, prev, key="title"):
            before = len(items)
            out = [i for i in items if i.get(key, "").lower() not in prev]
            removed = before - len(out)
            if removed:
                logger.info(f"Cross-day dedup: removed {removed} items seen yesterday")
            return out

        filtered_papers = _filter(papers, prev_papers)
        filtered_blogs = _filter(blogs, prev_blogs)
        filtered_news = _filter(news, prev_news)
        if github_trending is not None:
            filtered_github = _filter(github_trending, prev_github)
            return filtered_papers, filtered_blogs, filtered_news, filtered_github
        return filtered_papers, filtered_blogs, filtered_news

    # ------------------------------------------------------------------
    # Community picks builder
    # ------------------------------------------------------------------

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
            if BriefingCoordinator._URL_NOISE.search(url):
                continue
            out.append(url)
        return out[:10]

    @staticmethod
    def _build_community_picks(
        newsletters: List[Dict[str, Any]],
        github_trending: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge newsletter-extracted links and GitHub trending repos."""
        picks: List[Dict[str, Any]] = []
        seen_urls: set = set()
        for nl in newsletters:
            source = nl.get("source", "Newsletter")
            # Prefer structured (url, title) pairs extracted from HTML
            links: List = nl.get("links") or []
            if links:
                for url, title in links:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        picks.append({
                            "url": url,
                            "title": title,
                            "source": source,
                            "summary": "",
                            "source_type": "newsletter",
                        })
            else:
                # Fallback: extract URLs from plain text snippet
                text = (nl.get("snippet") or "") + " " + (nl.get("summary") or "")
                for url in BriefingCoordinator._extract_urls_from_text(text):
                    if url not in seen_urls:
                        seen_urls.add(url)
                        picks.append({
                            "url": url,
                            "title": nl.get("title", "Newsletter article"),
                            "source": source,
                            "summary": "",
                            "source_type": "newsletter",
                        })
        for repo in github_trending:
            if repo.get("link"):
                picks.append({
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
        return picks

    # ------------------------------------------------------------------
    # Intelligence helpers
    # ------------------------------------------------------------------

    def _enrich_papers(self, papers: list, topics: list) -> list:
        """Run paper summarization + semantic scoring."""
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
        result = self.intelligence.bedrock.invoke(prompt, tier="light", max_tokens=150)
        return result.strip() if result else ""

    def _ensure_paper_summaries(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        result = self.intelligence.bedrock.invoke(prompt, tier="light", max_tokens=500)
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
        return papers

    # ------------------------------------------------------------------
    # Markdown rendering (migrated from v1)
    # ------------------------------------------------------------------

    @staticmethod
    def _render_stars(score: int) -> str:
        if score is None:
            return ""
        score = max(0, min(score, 5))
        return "★" * score + "☆" * (5 - score)

    @staticmethod
    def _clean_summary(summary: str, title: str, source: str = "") -> str:
        """Remove title/source echo from LLM-generated summary."""
        if not summary:
            return summary
        s = summary.lstrip("* ").strip()
        if s.lower().startswith("summary:"):
            s = s[8:].lstrip("* ").strip()
        if not title:
            return s
        title_lower = title.lower()[:40]
        if s.lower().startswith(title_lower):
            rest = s[len(title):].strip()
            if rest.startswith("(") and ")" in rest:
                rest = rest[rest.index(")") + 1:].strip()
            if rest.startswith(("-", ":", "\u2013")):
                rest = rest[1:].strip()
            return rest if rest else summary
        return s

    def _get_limit(self, key: str, default: int) -> int:
        env_val = os.environ.get(f"BRIEFING_{key.upper()}")
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                pass
        return self.config.get(key, default)

    def _render_stocks(self, stocks: List[Dict[str, Any]], market_trend: str = "") -> str:
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

    def _render_news(self, news: List[Dict[str, Any]]) -> str:
        max_news = self._get_limit("max_news_render", 5)
        md = ["## AI & Tech News\n\n"]
        for article in news[:max_news]:
            title = article.get("title", "")
            url = article.get("url", "")
            summary = self._clean_summary(article.get("brief_summary", ""), title)
            if url:
                md.append(f"**[{title}]({url})**\n")
            else:
                md.append(f"**{title}**\n")
            if summary:
                md.append(f"{summary}\n")
            md.append("\n")
        return "".join(md)

    def _render_blogs(self, blogs: List[Dict[str, Any]]) -> str:
        max_blogs = self._get_limit("max_blogs_render", 5)
        md = ["## Blog Updates\n\n"]
        sorted_blogs = sorted(blogs[:max_blogs], key=lambda x: x.get("score_combined", 0), reverse=True)
        if any(b.get("score_combined") for b in sorted_blogs):
            sorted_blogs = [b for b in sorted_blogs if b.get("score_combined", 0) >= 3]
        for article in sorted_blogs:
            title = article.get("title", "")
            source = article.get("source", "")
            link = article.get("link", "")
            score = article.get("score_combined")
            summary = self._clean_summary(article.get("brief_summary", ""), title, source)
            score_tag = f" {self._render_stars(score)}" if score else ""
            if link:
                md.append(f"**[{title}]({link})** *({source})*{score_tag}\n")
            else:
                md.append(f"**{title}** *({source})*{score_tag}\n")
            if summary:
                md.append(f"{summary}\n")
            md.append("\n")
        return "".join(md)

    def _render_community_picks(self, items: List[Dict[str, Any]]) -> str:
        max_items = self._get_limit("max_community_picks", 8)
        sorted_items = sorted(items, key=lambda x: x.get("score_combined", 0), reverse=True)
        if any(i.get("score_combined") for i in sorted_items):
            sorted_items = [i for i in sorted_items if i.get("score_combined", 0) >= 3]
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
                item.get("brief_summary") or item.get("summary") or "", title, source
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

    def _render_top_papers(self, top_papers: List[Dict[str, Any]]) -> str:
        max_top = self._get_limit("max_top_papers_render", 3)
        md = ["## Top Papers\n\n"]
        sorted_papers = sorted(top_papers[:max_top], key=lambda x: x.get("score_combined", 0), reverse=True)
        if any(p.get("score_combined") for p in sorted_papers):
            sorted_papers = [p for p in sorted_papers if p.get("score_combined", 0) >= 3]
        for i, paper in enumerate(sorted_papers, 1):
            title = paper.get("title", "")
            authors = paper.get("authors", [])
            arxiv_url = paper.get("arxiv_url", "")
            brief_summary = paper.get("brief_summary", "")
            relevance_reason = paper.get("relevance_reason", "")
            score = paper.get("score_combined")
            score_tag = f" {self._render_stars(score)}" if score else ""
            if arxiv_url:
                md.append(f"### {i}. [{title}]({arxiv_url}){score_tag}\n")
            else:
                md.append(f"### {i}. {title}{score_tag}\n")
            if authors:
                md.append(f"*{', '.join(authors[:3])}*\n\n")
            if brief_summary:
                md.append(f"{brief_summary}\n\n")
            elif relevance_reason:
                md.append(f"{relevance_reason}\n\n")
            md.append("\n\n")
        return "".join(md)

    def _render_papers(self, papers: List[Dict[str, Any]]) -> str:
        max_papers = self._get_limit("max_papers_render", 5)
        md = ["## Recent Papers\n\n"]
        for paper in papers[:max_papers]:
            title = paper.get("title", "")
            authors = paper.get("authors", [])
            arxiv_url = paper.get("arxiv_url", "")
            brief_summary = paper.get("brief_summary", "")
            md.append(f"**{title}**")
            if authors:
                md.append(f" *{', '.join(authors[:2])}*")
            if arxiv_url:
                md.append(f" [arxiv]({arxiv_url})")
            md.append("\n")
            if brief_summary:
                md.append(f"{brief_summary}\n")
            md.append("\n")
        return "".join(md)

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
        """Generate markdown briefing from all data."""
        md: List[str] = []

        # Editorial intro
        if synthesis and synthesis.get("editorial_intro"):
            intro = synthesis["editorial_intro"].strip()
            lines = intro.split("\n")
            cleaned = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "executive summary" in stripped.lower():
                    continue
                if "morning briefing" in stripped.lower() or "ai briefing" in stripped.lower():
                    continue
                date_stripped = stripped.lstrip("–—-*# ").strip()
                if re.match(r"^\d{4}-\d{2}-\d{2}$", date_stripped):
                    continue
                cleaned.append(line)
            intro = "\n".join(cleaned).strip()
            md.append("## Executive Summary\n\n")
            md.append(f"{intro}\n\n")

        section_data = {
            "stocks": stocks,
            "news": news,
            "newsletters": newsletters or [],
            "community_picks": community_picks or [],
            "blogs": blogs,
            "top_papers": top_papers,
            "papers": papers,
        }
        for section in self.DEFAULT_SECTION_ORDER:
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

        if weekly_deep_dive:
            md.append("## This Week in AI\n\n")
            md.append(f"{weekly_deep_dive}\n\n")

        if self.errors:
            md.append("## Errors\n\n")
            for error in self.errors:
                md.append(f"- {error}\n")
            md.append("\n")

        return "".join(md)

    @staticmethod
    def _inject_podcast_section(markdown_content: str, podcast_url: str) -> str:
        section = (
            "\n\n## Audio Overview\n\n"
            f"[Listen on NotebookLM]({podcast_url}) "
            "*(audio generating — ready in ~10 min)*\n"
        )
        return markdown_content.rstrip() + section

    # ------------------------------------------------------------------
    # Obsidian-based context (replaces file-based memory)
    # ------------------------------------------------------------------

    def _load_obsidian_context(self, date: datetime) -> Dict[str, Any]:
        """Load cross-day context from Obsidian vault for synthesis enrichment.

        Reads yesterday's briefing frontmatter and tracked entity pages to
        provide narrative continuity. All reads are best-effort.
        """
        obsidian_config = self.config.get("obsidian", {})
        if not obsidian_config.get("enabled", False):
            return {}

        api_url = obsidian_config.get("api_url", "http://localhost:27123")
        api_key = os.environ.get("OBSIDIAN_API_KEY", "")
        if not api_key:
            return {}

        try:
            writer = ObsidianWriter(api_url, api_key, obsidian_config)
        except Exception:
            return {}

        context: Dict[str, Any] = {}

        # Read yesterday's briefing frontmatter
        from datetime import timedelta
        yesterday = date - timedelta(days=1)
        folder = obsidian_config.get("briefing_folder", "Sources/Briefings")
        yesterday_path = (
            f"{folder}/{yesterday.strftime('%Y')}/{yesterday.strftime('%m')}/"
            f"Personal-Briefing-{yesterday.strftime('%Y-%m-%d')}.md"
        )
        content = writer._get_note(yesterday_path)
        if content:
            fm = ObsidianWriter._extract_frontmatter(content)
            context["yesterday_themes"] = fm.get("emerging-themes", [])
            context["yesterday_top_papers"] = fm.get("top-papers", [])
            context["yesterday_entities"] = fm.get("entity-mentions", [])

        # Read tracked entity pages for timeline context
        entity_context = {}
        for entity in self.config.get("tracked_entities", [])[:5]:
            name = entity.get("name", "")
            if not name:
                continue
            vault_name = ObsidianWriter._to_vault_name(name)
            entity_content = writer._get_note(f"Wiki/Entities/{vault_name}.md")
            if entity_content:
                # Extract last 3 timeline headings
                headings = [
                    line.strip("# ").strip()
                    for line in entity_content.split("\n")
                    if line.startswith("### 20")  # date headings
                ]
                entity_context[name] = headings[-3:]
        if entity_context:
            context["entity_timelines"] = entity_context

        return context

    # ------------------------------------------------------------------
    # Distribution & publishing
    # ------------------------------------------------------------------

    def distribute_briefing(
        self, markdown_content: str, pdf_path: Optional[str], subject: str
    ) -> Dict[str, bool]:
        if self.dry_run:
            logger.info("Dry run: Skipping distribution")
            return {}
        sender_email = os.environ.get("GMAIL_USER")
        sender_password = os.environ.get("GMAIL_APP_PASSWORD")
        if not sender_email or not sender_password:
            logger.warning("Gmail credentials not set, skipping distribution")
            return {}

        # Guard against duplicate sends on K8s job retry
        today = datetime.now().strftime("%Y-%m-%d")
        if self._state_store.email_sent_today(today):
            logger.warning(f"Email already sent today ({today}), skipping duplicate send")
            self.status["email_sent"] = True
            return {}

        try:
            distributor = EmailDistributor(sender_email=sender_email, sender_password=sender_password)
            results = distributor.distribute(
                config=self.config, markdown_content=markdown_content,
                pdf_path=pdf_path, subject=subject, dry_run=self.dry_run,
            )
            sent = sum(1 for v in results.values() if v)
            self.status["email_sent"] = sent > 0
            self.status["distribution"] = {"sent": sent, "total": len(results), "details": results}
            logger.info(f"Distribution: {sent}/{len(results)} channels delivered")

            # Persist sent date immediately so retries don't re-send
            if sent > 0:
                self._state_store.mark_email_sent(today)

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
                markdown_content=markdown_content, date=date, status=self.status,
                emerging_themes=emerging_themes, top_papers=top_papers,
                entity_mentions=entity_mentions, trending_topics=trending_topics,
                weekly_deep_dive=weekly_deep_dive, briefing_name=briefing_name,
                weekly_briefing_names=weekly_briefing_names, podcast_url=podcast_url,
            )
            self.status["obsidian"] = results
            logger.info(f"Obsidian publish: briefing={'ok' if results.get('briefing') else 'fail'}")
            return results
        except Exception as e:
            logger.error(f"Obsidian publish failed: {e}")
            self.errors.append(f"Obsidian publish: {e}")
            return {}

    def generate_pdf(self, markdown_content: str, output_path: str) -> bool:
        try:
            pdf_config = self.config.get("pdf", {})
            page_format = self.config.get("output_format", "kindle")
            generator = PDFGenerator(
                page_format=page_format,
                font_size=pdf_config.get("font_size", 10),
                line_spacing=pdf_config.get("line_spacing", 1.5),
            )
            generator.generate_pdf(markdown_content, output_path)
            self.status["pdf_generated"] = True
            return True
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            self.errors.append(f"PDF generation: {e}")
            return False

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _format_filename(self, now: datetime) -> str:
        file_naming = self.config.get("file_naming", "Personal-Briefing-{yyyy}.{mm}.{dd}")
        known_vars = {
            "yyyy": now.strftime("%Y"),
            "mm": now.strftime("%m"),
            "dd": now.strftime("%d"),
            "type": "Daily",
        }
        return file_naming.format_map(known_vars)

    def _load_previous_state(self) -> Dict[str, Any]:
        return self._state_store.load()

    def _save_state(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        stocks: List[Dict[str, Any]],
        emerging_themes: List[str],
        trending_topics: Optional[Dict[str, Any]] = None,
        weekly_items: Optional[List[Dict[str, Any]]] = None,
        github_trending: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._state_store.save(
            papers=papers,
            blogs=blogs,
            news=news,
            stocks=stocks,
            emerging_themes=emerging_themes,
            trending_topics=trending_topics,
            weekly_items=weekly_items,
            github_trending=github_trending,
        )

    def save_status(self, output_dir: str = ".") -> None:
        self.status["errors"] = self.errors
        status_path = Path(output_dir) / "status.json"
        try:
            with open(status_path, "w") as f:
                json.dump(self.status, f, indent=2)
            logger.info(f"Status saved: {status_path}")
        except IOError as e:
            logger.warning(f"Failed to save status: {e}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Run the complete briefing pipeline."""
        start_time = time.time()
        logger.info("=== Starting Morning Briefing (v0.2 Coordinator + Workers) ===")
        now = self.run_date or datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        if self.run_date:
            logger.info(f"Rerun mode: generating briefing for {today_str}")

        with self._tracer.start_as_current_span("personal.briefing.run") as root_span:
            root_span.set_attribute("briefing.date", today_str)
            root_span.set_attribute("briefing.dry_run", self.dry_run)

            # Load previous state
            previous_state = self._load_previous_state()

            # Topic expansion
            topics = self.config.get("arxiv_topics", [])
            if self.intelligence.available:
                logger.info("=== Intelligence: Expanding Topics ===")
                topics = self.intelligence.expand_topics(topics)

            # --- Spawn all workers in parallel ---
            with self._tracer.start_as_current_span("personal.fetch"):
                logger.info("=== Spawning parallel workers ===")
                findings = self._spawn_workers()

            # Check failures
            failed = [f for f in findings if f["status"] == "error"]
            if len(failed) == len(findings):
                logger.error("All workers failed. Aborting.")
                root_span.set_status(trace.StatusCode.ERROR, "All workers failed")
                return 2
            if failed:
                logger.warning(f"{len(failed)} worker(s) failed: {[f['worker'] for f in failed]}")

            # Extract items
            papers, blogs, news, stocks, newsletters, github_trending = self._extract_items(findings)

            # Update status counts
            self.status["papers_found"] = len(papers)
            self.status["blogs_found"] = len(blogs)
            self.status["news_found"] = len(news)
            self.status["stocks_fetched"] = len(stocks)
            self.status["newsletters_found"] = len(newsletters)
            self.status["github_trending_found"] = len(github_trending)

            # --- Dynamic news queries + additional news fetch ---
            news_queries = self.config.get("news_queries", [])
            if self.intelligence.available:
                logger.info("=== Intelligence: Dynamic Queries ===")
                news_queries = self.intelligence.generate_dynamic_queries(
                    previous_state, news_queries
                )

            with self._tracer.start_as_current_span("personal.fetch.news"):
                api_key = os.environ.get("BRAVE_API_KEY")
                if api_key and news_queries:
                    try:
                        aggregator = NewsAggregator(
                            api_key=api_key, queries=news_queries,
                            max_results=self.config.get("max_news", 15),
                        )
                        extra_news = aggregator.aggregate_all_queries()
                        news.extend(extra_news)
                        self.status["news_found"] = len(news)
                    except Exception as e:
                        logger.error(f"News aggregation failed: {e}")
                        self.errors.append(f"News aggregation: {e}")

            # --- Deduplication ---
            news, blogs = self.deduplicate_news_and_blogs(news, blogs)
            papers = self.deduplicate_similar_papers(papers)
            papers, blogs, news, github_trending = self._dedup_against_previous(
                papers, blogs, news, previous_state, github_trending
            )

            # --- Build community picks ---
            community_picks = self._build_community_picks(newsletters, github_trending)
            if community_picks:
                nl_count = len([c for c in community_picks if c["source_type"] == "newsletter"])
                gh_count = len([c for c in community_picks if c["source_type"] == "github"])
                logger.info(f"Built {len(community_picks)} community picks ({nl_count} newsletter, {gh_count} GitHub)")

            # --- Intelligence enrichment ---
            synthesis: Dict[str, Any] = {}
            emerging_themes: List[str] = []
            if self.intelligence.available:
                with self._tracer.start_as_current_span("personal.enrich"):
                    logger.info("=== Intelligence: Enriching ===")

                    # Relevance filter
                    interest_profile = self.config.get("interest_profile")
                    if interest_profile:
                        papers = self.intelligence.filter_papers_by_relevance(papers, interest_profile)

                    # Parallel batch 1
                    interest_topics = self.config.get("interest_profile", [])
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        fp = pool.submit(self._enrich_papers, papers, topics)
                        fn = pool.submit(self.intelligence.rank_and_summarize_news, news, topics)
                        fb = pool.submit(self.intelligence.rank_and_summarize_blogs, blogs, topics)
                        fc = pool.submit(self.intelligence.rank_source_links, community_picks, interest_topics)

                        papers = fp.result()
                        news = fn.result()
                        blogs = fb.result()
                        community_picks = fc.result()

                    # Parallel batch 2
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        fs = pool.submit(self.intelligence.correlate_stocks_and_news, stocks, news)
                        ft = pool.submit(
                            self.intelligence.detect_emerging_themes,
                            papers, blogs, news, newsletters, github_trending,
                        )
                        stocks = fs.result()
                        emerging_themes = ft.result()

                    # Trending tracking
                    previous_state, papers, blogs, news = self.intelligence.track_trending(
                        papers, blogs, news, previous_state, newsletters, github_trending,
                    )

            # Market trend
            market_trend = ""
            if self.intelligence.available and stocks:
                market_trend = self._analyze_market_trend(stocks)

            # Score papers
            top_papers: List[Dict[str, Any]] = []
            if papers:
                scorer_topics = self.config.get("arxiv_topics", [])
                weights = self.config.get("paper_scoring", {})
                num_picks = self.config.get("num_paper_picks", 3)
                scorer = PaperScorer(topics=scorer_topics, weights=weights, num_picks=num_picks)
                top_papers = scorer.get_top_picks(papers)

            # Synthesis
            if self.intelligence.available:
                with self._tracer.start_as_current_span("personal.synthesize"):
                    top_papers = self.intelligence.assess_reproduction_feasibility(top_papers)
                    top_papers = self._ensure_paper_summaries(top_papers[:3]) + top_papers[3:]

                    # Load cross-day context from Obsidian vault
                    obsidian_context = self._load_obsidian_context(now)

                    synthesis = self.intelligence.synthesize_briefing(
                        papers, blogs[:5], stocks, news[:5], top_papers[:3],
                        emerging_themes=emerging_themes,
                        previous_state=previous_state,
                        newsletters=newsletters,
                        github_trending=github_trending,
                        obsidian_context=obsidian_context,
                    )

                    tracked_entities = self.config.get("tracked_entities", [])
                    entity_mentions: List[Dict[str, Any]] = []
                    if tracked_entities:
                        entity_mentions = self.intelligence.detect_entity_mentions(
                            papers, blogs, news, tracked_entities, newsletters, github_trending,
                        )
                        synthesis["entity_mentions"] = entity_mentions

            # Weekly deep dive (Saturday)
            is_saturday = now.weekday() == 5
            weekly_deep_dive = ""
            weekly_items = previous_state.get("weekly_items", [])
            for paper in top_papers[:3]:
                weekly_items.append({"date": today_str, "type": "paper", "title": paper.get("title", "")})
            for article in news[:3]:
                weekly_items.append({"date": today_str, "type": "news", "title": article.get("title", "")})
            if is_saturday and self.intelligence.available and weekly_items:
                weekly_deep_dive = self.intelligence.generate_weekly_deep_dive(weekly_items)
                weekly_items = []

            # Check data
            has_data = any([papers, blogs, stocks, news, newsletters, github_trending])
            if not has_data:
                logger.error("No data collected from any source")
                self.status["elapsed_seconds"] = round(time.time() - start_time, 1)
                self.save_status()
                root_span.set_status(trace.StatusCode.ERROR, "No data collected")
                return 2

            podcast_url: Optional[str] = None
            with self._tracer.start_as_current_span("personal.render"):
                filename = self._format_filename(now)
                markdown_content = self.generate_markdown_briefing(
                    papers, blogs, stocks, news, top_papers, synthesis,
                    market_trend=market_trend, weekly_deep_dive=weekly_deep_dive,
                    newsletters=newsletters, community_picks=community_picks,
                )

                # Podcast
                if self.podcast_generator.enabled:
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
                    _max_community = self._get_limit("max_community_picks", 8)
                    _rendered_community = sorted(
                        community_picks,
                        key=lambda x: x.get("score_combined", 0), reverse=True,
                    )
                    if any(c.get("score_combined") for c in _rendered_community):
                        _rendered_community = [c for c in _rendered_community if c.get("score_combined", 0) >= 3]
                    _rendered_community = _rendered_community[:_max_community]

                    source_urls = (
                        [p.get("url") or p.get("arxiv_url") for p in _rendered_top
                         if p.get("url") or p.get("arxiv_url")]
                        + [b["link"] for b in _rendered_blogs if b.get("link")]
                        + [n["url"] for n in news[:_max_news] if n.get("url")]
                        + [c.get("url") or c.get("link") for c in _rendered_community
                           if c.get("url") or c.get("link")]
                    )[:10]

                    podcast_url = self.podcast_generator.generate(markdown_content, now, source_urls)
                    if podcast_url:
                        markdown_content = self._inject_podcast_section(markdown_content, podcast_url)
                        self.status["podcast_url"] = podcast_url
                        logger.info(f"Podcast URL injected: {podcast_url}")
                    elif self.podcast_generator.enabled:
                        self.status["podcast_error"] = "failed — check logs (possible auth expiry)"
                        logger.warning("Podcast generation failed — briefing continues without podcast link")

                # Save markdown
                md_path = f"{filename}.md"
                try:
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    logger.info(f"Saved markdown: {md_path}")
                except IOError as e:
                    logger.warning(f"Failed to save markdown: {e}")

                # PDF (skip for HTML-only)
                pdf_path = None
                if self.config.get("output_format", "kindle") != "html":
                    pdf_path = f"{filename}.pdf"
                    if not self.generate_pdf(markdown_content, pdf_path):
                        logger.error("Failed to generate PDF")
                        self.status["elapsed_seconds"] = round(time.time() - start_time, 1)
                        self.save_status()
                        root_span.set_status(trace.StatusCode.ERROR, "PDF generation failed")
                        return 2

            # Distribute
            self.distribute_briefing(markdown_content, pdf_path, filename)

        # Obsidian (outside root span — independent side-effect)
        self.publish_to_obsidian(
            markdown_content=markdown_content, date=now, top_papers=top_papers,
            emerging_themes=emerging_themes,
            entity_mentions=synthesis.get("entity_mentions", []),
            trending_topics=previous_state.get("trending_topics", {}),
            weekly_deep_dive=weekly_deep_dive, briefing_name=filename,
            weekly_items=weekly_items, podcast_url=podcast_url,
        )

        # Mark newsletters as read
        ns_config = self.config.get("newsletter_source", {})
        if (
            hasattr(self, "_gmail_worker")
            and self._gmail_worker.scanner
            and ns_config.get("mark_read", True)
        ):
            self._gmail_worker.scanner.mark_digested()

        # Save state
        self._save_state(
            top_papers, blogs, news, stocks, emerging_themes,
            trending_topics=previous_state.get("trending_topics", {}),
            weekly_items=weekly_items,
            github_trending=github_trending,
        )

        # Finalize
        elapsed = time.time() - start_time
        self.status["elapsed_seconds"] = round(elapsed, 1)
        self.save_status()

        logger.info(f"=== Briefing Complete in {elapsed:.1f}s ===")
        if self.errors:
            logger.warning(f"Completed with {len(self.errors)} errors")
            return 1
        logger.info("Completed successfully")
        return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Morning Briefing v0.2 (Coordinator + Workers)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Skip email distribution")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--date", type=str, default=None, metavar="YYYY-MM-DD",
        help="Generate briefing for a past date (search window: [DATE - days_back, DATE])",
    )
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    config = load_config(args.config)

    is_valid, messages = validate_config(config)
    if not is_valid:
        logger.error("Configuration is invalid. Fix errors above and retry.")
        return 2

    check_environment(config, dry_run=args.dry_run)

    run_date = None
    if args.date:
        try:
            run_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid --date format '{args.date}': expected YYYY-MM-DD")
            return 2

    max_retries = config.get("max_retries", 2)
    retry_backoffs = [30, 120]  # seconds between attempts

    for attempt in range(max_retries + 1):
        coordinator = BriefingCoordinator(config=config, dry_run=args.dry_run, run_date=run_date)
        result = coordinator.run()
        if result == 0:
            return 0
        if result == 2:
            # Fatal config/data error — don't retry
            return 2
        if attempt < max_retries:
            wait = retry_backoffs[min(attempt, len(retry_backoffs) - 1)]
            logger.warning(
                "Attempt %d/%d failed (exit %d) — retrying in %ds",
                attempt + 1, max_retries + 1, result, wait,
            )
            time.sleep(wait)

    logger.error("All %d attempt(s) failed", max_retries + 1)
    return result


if __name__ == "__main__":
    sys.exit(main())
