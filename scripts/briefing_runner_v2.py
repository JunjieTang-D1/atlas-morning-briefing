#!/usr/bin/env python3
"""
Morning briefing runner v0.2 - Coordinator + Parallel Workers Architecture.

KEY CHANGES FROM V0.1:
- Coordinator pattern: coordinator READS findings and synthesizes (no lazy delegation)
- Workers are fully self-contained (fetch + enrich independently)
- All workers run in parallel
- Memory system for cross-day learning
- Skeptical memory: hints not truth, verify before using

Based on Claude Code leaked architecture patterns:
- Coordinator never writes code/fetches data directly
- Workers have self-contained prompts with purpose, context, output format
- Synthesis happens AFTER reading all findings
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.workers.papers_worker import PapersWorker
from scripts.workers.blogs_worker import BlogsWorker
from scripts.workers.news_market_worker import NewsMarketWorker
from scripts.llm_client import LLMClient
from scripts.pdf_generator import PDFGenerator
from scripts.email_distributor import EmailDistributor
from scripts.config_validator import validate_config, check_environment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s)")
logger = logging.getLogger(__name__)

STATE_FILENAME = ".atlas-state.json"
MEMORY_DIR = Path("briefing-memory")


class BriefingCoordinator:
    """
    Coordinator for v0.2 multi-agent briefing generation.

    The coordinator:
    1. Spawns parallel workers
    2. Waits for all findings
    3. READS all findings (not lazy delegation)
    4. Synthesizes: executive summary, correlations, emerging themes
    5. Updates memory after briefing
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dry_run: bool = False,
        run_date: Optional[datetime] = None,
    ):
        """
        Initialize BriefingCoordinator.

        Args:
            config: Configuration dictionary
            dry_run: If True, skip email distribution
            run_date: Optional date for historical reruns (None = today)
        """
        self.config = config
        self.dry_run = dry_run
        self.run_date = run_date
        self.llm = LLMClient(config.get("llm", {}))

        # Initialize memory system
        self.memory_dir = MEMORY_DIR
        self.memory_dir.mkdir(exist_ok=True)

    def run(self) -> int:
        """
        Run the v0.2 coordinator workflow.

        Returns:
            Exit code (0=success, 1=partial failure, 2=total failure)
        """
        start_time = time.time()
        logger.info("=== Morning Briefing v0.2 - Coordinator + Parallel Workers ===")

        # Load memory (hints for workers)
        memory = self._load_memory()

        # Spawn all workers in parallel
        logger.info("=== Spawning parallel workers ===")
        findings = self._spawn_workers()

        # Check if all workers succeeded
        failed_workers = [f for f in findings if f["status"] == "error"]
        if len(failed_workers) == len(findings):
            logger.error("All workers failed. Aborting.")
            return 2

        if failed_workers:
            logger.warning(f"{len(failed_workers)} worker(s) failed: {[f['worker'] for f in failed_workers]}")

        # Extract items from findings
        papers, blogs, news, stocks = self._extract_items(findings)

        # Coordinator synthesis (NOT lazy delegation - reads all findings)
        logger.info("=== Coordinator Synthesis ===")
        synthesis = self._synthesize_findings(findings, papers, blogs, news, stocks, memory)

        # Generate briefing document
        logger.info("=== Generating briefing document ===")
        briefing_content = self._generate_briefing(synthesis, papers, blogs, news, stocks)

        # Generate PDF
        output_filename = self._get_output_filename()
        pdf_path = self._generate_pdf(briefing_content, output_filename)

        # Distribute email (if not dry run)
        if not self.dry_run:
            self._distribute_email(briefing_content, output_filename)

        # Update memory with today's findings
        self._update_memory(synthesis, papers, blogs, news, stocks)

        # Save state for cross-day deduplication
        self._save_state(papers, blogs, news)

        elapsed = time.time() - start_time
        logger.info(f"=== Briefing completed in {elapsed:.1f}s ===")

        # Calculate metrics
        total_tokens = sum(f["metadata"]["token_count"] for f in findings)
        logger.info(f"Total LLM tokens used: {total_tokens}")

        return 0 if not failed_workers else 1

    def _spawn_workers(self) -> List[Dict[str, Any]]:
        """
        Spawn all workers in parallel and collect findings.

        Returns:
            List of finding dictionaries from all workers
        """
        workers = [
            PapersWorker(self.config, ref_date=self.run_date),
            BlogsWorker(self.config, ref_date=self.run_date),
            NewsMarketWorker(self.config, ref_date=self.run_date),
        ]

        findings = []
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = {executor.submit(worker.execute): worker for worker in workers}

            for future in as_completed(futures):
                worker = futures[future]
                try:
                    finding = future.result()
                    findings.append(finding)
                    logger.info(f"[{finding['worker']}] completed in {finding['metadata']['processing_time']:.1f}s")
                except Exception as e:
                    logger.error(f"Worker {worker.worker_name} raised exception: {e}")
                    findings.append({
                        "worker": worker.worker_name,
                        "status": "error",
                        "items": [],
                        "metadata": {"processing_time": 0, "token_count": 0, "items_found": 0, "items_kept": 0},
                        "synthesis": "",
                        "error": str(e)
                    })

        return findings

    def _extract_items(self, findings: List[Dict[str, Any]]) -> tuple:
        """
        Extract papers, blogs, news, stocks from findings.

        Args:
            findings: List of finding dictionaries

        Returns:
            (papers, blogs, news, stocks) tuple
        """
        papers = []
        blogs = []
        news = []
        stocks = []

        for finding in findings:
            if finding["worker"] == "papers_worker":
                papers = finding.get("items", [])
            elif finding["worker"] == "blogs_worker":
                blogs = finding.get("items", [])
            elif finding["worker"] == "news_market_worker":
                items = finding.get("items", {})
                news = items.get("news", [])
                stocks = items.get("stocks", [])

        return papers, blogs, news, stocks

    def _synthesize_findings(
        self,
        findings: List[Dict[str, Any]],
        papers: list,
        blogs: list,
        news: list,
        stocks: list,
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinator reads ALL findings and synthesizes (not lazy delegation).

        Args:
            findings: Raw findings from workers
            papers: Papers list
            blogs: Blogs list
            news: News list
            stocks: Stocks list
            memory: Memory hints

        Returns:
            Synthesis dictionary with executive summary, correlations, themes
        """
        logger.info("Coordinator reading all worker findings...")

        # Read each worker's synthesis
        worker_syntheses = {f["worker"]: f["synthesis"] for f in findings}

        # Detect emerging themes across all sources
        emerging_themes = self._detect_emerging_themes(papers, blogs, news)

        # Generate executive summary (coordinator's unique value-add)
        executive_summary = self._generate_executive_summary(
            worker_syntheses,
            emerging_themes,
            stocks
        )

        # Market trend analysis
        market_trend = self._analyze_market_trend(stocks, news)

        return {
            "executive_summary": executive_summary,
            "worker_syntheses": worker_syntheses,
            "emerging_themes": emerging_themes,
            "market_trend": market_trend,
            "total_items": len(papers) + len(blogs) + len(news) + len(stocks)
        }

    def _detect_emerging_themes(self, papers: list, blogs: list, news: list) -> List[str]:
        """
        Detect emerging themes across papers, blogs, and news.

        Args:
            papers: Papers list
            blogs: Blogs list
            news: News list

        Returns:
            List of emerging theme strings
        """
        if not self.llm.available:
            return []

        # Build prompt with key items from each source
        top_papers = sorted(papers, key=lambda p: p.get("score", 0), reverse=True)[:3]
        top_blogs = sorted(blogs, key=lambda b: b.get("llm_score", 0), reverse=True)[:3]
        top_news = sorted(news, key=lambda n: n.get("llm_score", 0), reverse=True)[:3]

        prompt = "Detect 2-3 emerging themes that appear across these sources:\n\n"
        prompt += "TOP PAPERS:\n"
        for p in top_papers:
            prompt += f"- {p.get('title', 'Unknown')}\n"
        prompt += "\nTOP BLOGS:\n"
        for b in top_blogs:
            prompt += f"- {b.get('title', 'Unknown')}\n"
        prompt += "\nTOP NEWS:\n"
        for n in top_news:
            prompt += f"- {n.get('title', 'Unknown')}\n"
        prompt += "\nReturn ONLY a comma-separated list of 2-3 themes (no explanation)."

        response = self.llm.invoke(prompt, tier="light")
        if not response:
            return []
        themes = [t.strip() for t in response.split(",")]
        return themes[:3]

    def _generate_executive_summary(
        self,
        worker_syntheses: Dict[str, str],
        emerging_themes: List[str],
        stocks: list
    ) -> str:
        """
        Generate executive summary by reading worker syntheses (not lazy delegation).

        Args:
            worker_syntheses: Dict of worker -> synthesis
            emerging_themes: Detected themes
            stocks: Stock data

        Returns:
            Executive summary string
        """
        if not self.llm.available:
            return "Executive summary unavailable (LLM offline)"

        prompt = "You are writing the executive summary for today's morning briefing.\n\n"
        prompt += "WORKER FINDINGS:\n"
        for worker, synthesis in worker_syntheses.items():
            prompt += f"\n{worker}: {synthesis}\n"

        if emerging_themes:
            prompt += f"\nEMERGING THEMES: {', '.join(emerging_themes)}\n"

        if stocks:
            gainers = [s for s in stocks if s.get("change_pct", 0) > 0]
            prompt += f"\nMARKET: {len(gainers)}/{len(stocks)} stocks up\n"

        prompt += "\nWrite a 2-3 sentence executive summary highlighting the most important insights. Be specific."

        return self.llm.invoke(prompt, tier="medium") or "Executive summary unavailable"

    def _analyze_market_trend(self, stocks: list, news: list) -> str:
        """
        Analyze market trend with context from news.

        Args:
            stocks: Stock data
            news: News articles

        Returns:
            Market trend analysis string
        """
        if not stocks or not self.llm.available:
            return ""

        gainers = [s for s in stocks if s.get("change_pct", 0) > 0]
        losers = [s for s in stocks if s.get("change_pct", 0) < 0]

        prompt = f"Analyze today's market trend:\n\n"
        prompt += f"STOCKS: {len(gainers)} up, {len(losers)} down\n"
        for s in stocks:
            prompt += f"- {s.get('symbol', 'Unknown')}: {s.get('change_pct', 0):.2f}%\n"

        if news:
            prompt += f"\nRELEVANT NEWS:\n"
            for n in news[:3]:
                prompt += f"- {n.get('title', 'Unknown')}\n"

        prompt += "\nProvide a 1-2 sentence market trend analysis."

        return self.llm.invoke(prompt, tier="light") or ""

    def _generate_briefing(
        self,
        synthesis: Dict[str, Any],
        papers: list,
        blogs: list,
        news: list,
        stocks: list
    ) -> str:
        """
        Generate final briefing markdown content.

        Args:
            synthesis: Synthesis dictionary
            papers: Papers list
            blogs: Blogs list
            news: News list
            stocks: Stocks list

        Returns:
            Markdown content string
        """
        content = f"# Morning Briefing - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"
        content += "## Executive Summary\n\n"
        content += synthesis["executive_summary"] + "\n\n"

        if synthesis["emerging_themes"]:
            content += "**Emerging Themes:** " + ", ".join(synthesis["emerging_themes"]) + "\n\n"

        # Financial section
        if stocks:
            content += "## Financial Markets\n\n"
            content += synthesis["market_trend"] + "\n\n"
            for stock in stocks:
                pct = stock.get("change_pct", 0)
                arrow = "↑" if pct > 0 else "↓" if pct < 0 else "→"
                content += f"- **{stock.get('symbol', 'Unknown')}**: {arrow} {pct:.2f}%"
                if stock.get("ai_insight"):
                    content += f" — {stock.get('ai_insight')}"
                content += "\n"
            content += "\n"

        # News section
        if news:
            content += "## News\n\n"
            for item in news[:5]:
                score = item.get("llm_score", 0)
                stars = "★" * score + "☆" * (5 - score)
                content += f"### {item.get('title', 'Unknown')} {stars}\n\n"
                if item.get("llm_summary"):
                    content += f"{item['llm_summary']}\n\n"
                content += f"[Read more]({item.get('url', '#')})\n\n"

        # Papers section
        if papers:
            content += "## Research Papers\n\n"
            top_papers = sorted(papers, key=lambda p: p.get("score", 0), reverse=True)[:5]
            for paper in top_papers:
                content += f"### {paper.get('title', 'Unknown')}\n\n"
                if paper.get("llm_summary"):
                    content += f"{paper['llm_summary']}\n\n"
                content += f"**Authors:** {', '.join(paper.get('authors', []))}\n\n"
                content += f"[ArXiv]({paper.get('link', '#')})\n\n"

        # Blogs section
        if blogs:
            content += "## Blog Posts\n\n"
            for blog in blogs[:5]:
                score = blog.get("llm_score", 0)
                stars = "★" * score + "☆" * (5 - score)
                content += f"### {blog.get('title', 'Unknown')} {stars}\n\n"
                content += f"**Source:** {blog.get('source', 'Unknown')}\n\n"
                if blog.get("llm_summary"):
                    content += f"{blog['llm_summary']}\n\n"
                content += f"[Read more]({blog.get('link', '#')})\n\n"

        return content

    def _generate_pdf(self, content: str, filename: str) -> Path:
        """Generate PDF from markdown content."""
        output_path = f"{filename}.pdf"
        logger.info(f"Generating PDF: {output_path}")
        # Extract PDF config
        pdf_config = self.config.get("pdf", {})
        page_format = self.config.get("output_format", "kindle")
        font_size = pdf_config.get("font_size", 10)
        line_spacing = pdf_config.get("line_spacing", 1.5)

        generator = PDFGenerator(
            page_format=page_format,
            font_size=font_size,
            line_spacing=line_spacing
        )
        generator.generate_pdf(content, output_path)
        logger.info(f"PDF saved to: {output_path}")
        return Path(output_path)

    def _distribute_email(self, content: str, filename: str):
        """Distribute briefing via email."""
        logger.info("Distributing briefing via email")
        distributor = EmailDistributor(self.config)
        distributor.send_briefing(content, filename)

    def _get_output_filename(self) -> str:
        """Get output filename based on config pattern."""
        pattern = self.config.get("file_naming", "Personal-Briefing-{yyyy}.{mm}.{dd}")
        now = self.run_date or datetime.now(timezone.utc)
        return pattern.format(yyyy=now.year, mm=f"{now.month:02d}", dd=f"{now.day:02d}")

    def _load_memory(self) -> Dict[str, Any]:
        """Load memory hints (not truth - verify before using)."""
        memory_file = self.memory_dir / "MEMORY.md"
        if not memory_file.exists():
            return {}

        # For now, just return empty dict (memory system implemented next)
        return {}

    def _update_memory(
        self,
        synthesis: Dict[str, Any],
        papers: list,
        blogs: list,
        news: list,
        stocks: list
    ):
        """Update memory with today's findings."""
        # Implemented in next step
        pass

    def _save_state(self, papers: list, blogs: list, news: list):
        """Save state for cross-day deduplication."""
        state = {
            "date": datetime.now(timezone.utc).isoformat(),
            "paper_titles": [p.get("title", "") for p in papers],
            "blog_titles": [b.get("title", "") for b in blogs],
            "news_titles": [n.get("title", "") for n in news]
        }
        with open(STATE_FILENAME, "w") as f:
            json.dump(state, f, indent=2)


def main():
    """Main entry point for v0.2 briefing runner."""
    parser = argparse.ArgumentParser(description="Morning Briefing v0.2 (Coordinator + Workers)")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Skip email distribution")
    parser.add_argument("--date", type=str, help="Rerun for a past date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Load and validate config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    validate_config(config)
    check_environment(config)

    # Parse optional --date
    run_date = None
    if args.date:
        try:
            run_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid --date format '{args.date}': expected YYYY-MM-DD")
            sys.exit(2)

    # Run coordinator
    coordinator = BriefingCoordinator(config, dry_run=args.dry_run, run_date=run_date)
    exit_code = coordinator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
