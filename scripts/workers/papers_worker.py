#!/usr/bin/env python3
"""
Papers worker for v0.2 multi-agent architecture.

PURPOSE: Fetch ArXiv papers matching configured topics, enrich with LLM scoring
and relevance filtering, and return structured findings.

Self-contained worker that does NOT delegate back to coordinator.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.arxiv_scanner import ArxivScanner
from scripts.paper_scorer import PaperScorer
from scripts.llm_client import LLMClient
from scripts.intelligence import BriefingIntelligence
from scripts.workers.base_worker import BaseWorker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s)")
logger = logging.getLogger(__name__)


class PapersWorker(BaseWorker):
    """Fetches and enriches ArXiv papers independently."""

    def __init__(self, config: Dict[str, Any], ref_date: Optional[datetime] = None):
        """
        Initialize PapersWorker.

        Args:
            config: Full configuration dictionary
            ref_date: Reference date for historical reruns
        """
        super().__init__(config, "papers_worker", ref_date=ref_date)
        self.topics = config.get("arxiv_topics", [])
        self.days_back = config.get("arxiv_days_back", 3)
        self.max_papers = config.get("max_papers", 30)

    def execute(self) -> Dict[str, Any]:
        """
        Execute papers workflow: fetch + enrich + score + filter.

        Returns:
            Finding dict with enriched papers
        """
        self._start_timing()
        token_count = 0

        try:
            logger.info(f"[{self.worker_name}] Starting ArXiv paper scan")

            # Step 1: Fetch papers from ArXiv
            scanner = ArxivScanner(
                topics=self.topics,
                days_back=self.days_back,
                max_results=self.max_papers,
                end_date=self.ref_date,
            )
            papers = scanner.scan_all_topics()
            items_found = len(papers)
            logger.info(f"[{self.worker_name}] Fetched {items_found} papers")

            if not papers:
                return self._create_finding(
                    status="success",
                    items=[],
                    synthesis="No papers found matching configured topics",
                    items_found=0
                )

            # Step 2: Initialize intelligence layer for enrichment
            llm = LLMClient(self.config.get("llm", {}))
            intelligence = BriefingIntelligence(llm, self.config)

            if not intelligence.available:
                logger.warning(f"[{self.worker_name}] Intelligence layer unavailable, skipping enrichment")
                # Fallback: basic TF-IDF scoring only
                scorer = PaperScorer(self.config)
                papers = scorer.score_papers(papers, self.topics)
                return self._create_finding(
                    status="success",
                    items=papers,
                    synthesis=f"Found {len(papers)} papers. LLM enrichment unavailable.",
                    items_found=items_found
                )

            # Step 3: Relevance filtering (Stage 1)
            interest_profile = self.config.get("interest_profile")
            if interest_profile:
                logger.info(f"[{self.worker_name}] Stage 1: Relevance filtering")
                papers = intelligence.filter_papers_by_relevance(papers, interest_profile)
                logger.info(f"[{self.worker_name}] After relevance filter: {len(papers)} papers")

            # Step 4: Semantic scoring and summarization
            logger.info(f"[{self.worker_name}] Enriching papers with LLM")
            papers = intelligence.summarize_papers(papers)
            papers = intelligence.score_papers_semantically(papers, self.topics)

            # Estimate token usage (rough): ~500 tokens per paper for summarization
            token_count = len(papers) * 500

            # Step 5: Final scoring (combines TF-IDF + semantic)
            weights = self.config.get("paper_scoring", {})
            num_picks = self.config.get("num_paper_picks", 3)
            scorer = PaperScorer(topics=self.topics, weights=weights, num_picks=num_picks)
            papers = scorer.score_papers(papers)

            # Step 6: Generate synthesis
            top_papers = sorted(papers, key=lambda p: p.get("score", 0), reverse=True)[:5]
            synthesis = self._generate_synthesis(top_papers)

            logger.info(f"[{self.worker_name}] Completed. {len(papers)} papers enriched.")

            return self._create_finding(
                status="success",
                items=papers,
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

    def _generate_synthesis(self, top_papers: list) -> str:
        """
        Generate synthesis summary from top papers.

        Args:
            top_papers: Top 5 papers by score

        Returns:
            Summary string
        """
        if not top_papers:
            return "No high-scoring papers found"

        topics = set()
        for paper in top_papers:
            if "category" in paper:
                topics.add(paper["category"])

        synthesis = f"Found {len(top_papers)} high-relevance papers. "
        if topics:
            synthesis += f"Key areas: {', '.join(list(topics)[:3])}. "
        synthesis += f"Top paper: '{top_papers[0].get('title', 'Unknown')}' (score: {top_papers[0].get('score', 0):.1f})"

        return synthesis
