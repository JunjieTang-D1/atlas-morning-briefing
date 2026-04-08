"""Worker modules for v0.2 multi-agent architecture."""

from .base_worker import BaseWorker
from .papers_worker import PapersWorker
from .blogs_worker import BlogsWorker
from .news_market_worker import NewsMarketWorker
from .gmail_worker import GmailWorker
from .github_trending_worker import GitHubTrendingWorker

__all__ = [
    "BaseWorker",
    "PapersWorker",
    "BlogsWorker",
    "NewsMarketWorker",
    "GmailWorker",
    "GitHubTrendingWorker",
]
