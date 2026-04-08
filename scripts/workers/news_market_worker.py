#!/usr/bin/env python3
"""
News and market worker for v0.2 multi-agent architecture.

PURPOSE: Fetch news articles and stock data, enrich with LLM ranking and
correlation analysis, and return structured findings.

Self-contained worker that does NOT delegate back to coordinator.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.news_aggregator import NewsAggregator
from scripts.stock_fetcher import StockFetcher
from scripts.llm_client import LLMClient
from scripts.intelligence import BriefingIntelligence
from scripts.workers.base_worker import BaseWorker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s)")
logger = logging.getLogger(__name__)


class NewsMarketWorker(BaseWorker):
    """Fetches and enriches news + stock data independently."""

    def __init__(self, config: Dict[str, Any], ref_date: Optional[datetime] = None):
        """
        Initialize NewsMarketWorker.

        Args:
            config: Full configuration dictionary
            ref_date: Reference date for historical reruns (unused for news)
        """
        super().__init__(config, "news_market_worker", ref_date=ref_date)
        self.news_queries = config.get("news_queries", [])
        self.stocks = config.get("stocks", [])
        self.max_news = config.get("max_news", 15)

    def execute(self) -> Dict[str, Any]:
        """
        Execute news+market workflow: fetch + enrich + correlate.

        Returns:
            Finding dict with enriched news and stocks
        """
        self._start_timing()
        token_count = 0

        try:
            logger.info(f"[{self.worker_name}] Starting news and market data fetch")

            # Step 1: Fetch news articles
            news_aggregator = NewsAggregator(self.news_queries, self.config)
            news = news_aggregator.aggregate_all_queries()
            news_found = len(news)
            logger.info(f"[{self.worker_name}] Fetched {news_found} news articles")

            # Step 2: Fetch stock data
            stock_fetcher = StockFetcher(self.stocks, self.config)
            stocks = stock_fetcher.fetch_all_stocks()
            stocks_found = len(stocks)
            logger.info(f"[{self.worker_name}] Fetched {stocks_found} stock prices")

            # Step 3: Initialize intelligence layer for enrichment
            llm = LLMClient(self.config.get("llm", {}))
            intelligence = BriefingIntelligence(llm, self.config)

            if not intelligence.available:
                logger.warning(f"[{self.worker_name}] Intelligence layer unavailable, returning raw data")
                return self._create_finding(
                    status="success",
                    items={"news": news[:self.max_news], "stocks": stocks},
                    synthesis=f"Found {news_found} news articles and {stocks_found} stocks. LLM enrichment unavailable.",
                    items_found=news_found + stocks_found
                )

            # Step 4: Rank and summarize news with LLM
            logger.info(f"[{self.worker_name}] Enriching news with LLM")
            topics = self.config.get("arxiv_topics", [])
            news = intelligence.rank_and_summarize_news(news, topics)
            token_count += len(news) * 400  # ~400 tokens per news item

            # Step 5: Correlate stocks and news
            logger.info(f"[{self.worker_name}] Correlating stocks with news")
            stocks = intelligence.correlate_stocks_and_news(stocks, news)
            token_count += len(stocks) * 300  # ~300 tokens per stock correlation

            # Step 6: Filter to top news
            news = [n for n in news if n.get("llm_score", 0) >= 3]
            news = sorted(news, key=lambda n: n.get("llm_score", 0), reverse=True)[:self.max_news]

            # Step 7: Generate synthesis
            synthesis = self._generate_synthesis(news, stocks)

            logger.info(f"[{self.worker_name}] Completed. {len(news)} news + {len(stocks)} stocks enriched.")

            return self._create_finding(
                status="success",
                items={"news": news, "stocks": stocks},
                synthesis=synthesis,
                token_count=token_count,
                items_found=news_found + stocks_found
            )

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error: {e}")
            return self._create_finding(
                status="error",
                items={"news": [], "stocks": []},
                error=str(e)
            )

    def _generate_synthesis(self, news: list, stocks: list) -> str:
        """
        Generate synthesis summary from news and stocks.

        Args:
            news: Enriched news articles
            stocks: Enriched stock data

        Returns:
            Summary string
        """
        news_summary = f"{len(news)} high-relevance news articles"
        stocks_summary = f"{len(stocks)} stocks tracked"

        # Calculate market trend
        gainers = [s for s in stocks if s.get("change_pct", 0) > 0]
        losers = [s for s in stocks if s.get("change_pct", 0) < 0]

        if gainers and losers:
            trend = f"{len(gainers)} up, {len(losers)} down"
        elif gainers:
            trend = "bullish (all up)"
        elif losers:
            trend = "bearish (all down)"
        else:
            trend = "flat"

        synthesis = f"{news_summary}. {stocks_summary} ({trend}). "

        if news:
            synthesis += f"Top story: '{news[0].get('title', 'Unknown')}'"

        return synthesis
