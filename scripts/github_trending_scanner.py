#!/usr/bin/env python3
"""
GitHub trending scanner.

Scrapes github.com/trending directly and returns the daily trending repos
as briefing source items. No API key or external service required.
"""

import logging
from html.parser import HTMLParser
from typing import Any, Dict, List
from urllib.request import Request, urlopen
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_TRENDING_URL = "https://github.com/trending?since=daily&spoken_language_code=en"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class _TrendingParser(HTMLParser):
    """Minimal state-machine parser for github.com/trending."""

    def __init__(self) -> None:
        super().__init__()
        self.repos: List[Dict[str, Any]] = []
        self._in_article = False
        self._in_h2 = False
        self._in_h2_a = False
        self._in_desc_p = False
        self._in_lang_span = False
        self._in_stars_today = False
        self._current: Dict[str, Any] = {}
        self._depth = 0
        self._article_depth = 0
        self._h2_a_href = ""

    def handle_starttag(self, tag: str, attrs: List) -> None:
        attr = dict(attrs)
        self._depth += 1

        if tag == "article" and "Box-row" in attr.get("class", ""):
            self._in_article = True
            self._article_depth = self._depth
            self._current = {}
            return

        if not self._in_article:
            return

        if tag == "h2":
            self._in_h2 = True
        elif tag == "a" and self._in_h2:
            self._in_h2_a = True
            href = attr.get("href", "").strip()
            self._h2_a_href = f"https://github.com{href}" if href.startswith("/") else href
        elif tag == "p" and not self._current.get("description"):
            self._in_desc_p = True
        elif tag == "span":
            itemprop = attr.get("itemprop", "")
            aria = attr.get("aria-label", "")
            if itemprop == "programmingLanguage":
                self._in_lang_span = True
            elif "stars today" in aria.lower() or "star" in aria.lower():
                self._in_stars_today = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._in_article and self._depth == self._article_depth:
            if self._current.get("title"):
                self.repos.append(self._current)
            self._in_article = False
            self._current = {}
        if tag == "h2":
            self._in_h2 = False
        if tag == "a" and self._in_h2_a:
            self._in_h2_a = False
        if tag == "p":
            self._in_desc_p = False
        if tag == "span":
            self._in_lang_span = False
            self._in_stars_today = False
        self._depth -= 1

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or not self._in_article:
            return

        if self._in_h2_a and not self._current.get("title"):
            # href like /owner/repo — use last two segments as title
            parts = [p for p in self._h2_a_href.rstrip("/").split("/") if p]
            if len(parts) >= 2:
                owner, repo = parts[-2], parts[-1]
                self._current["title"] = f"{owner}/{repo}"
                self._current["link"] = self._h2_a_href
        elif self._in_desc_p and not self._current.get("description"):
            self._current["description"] = text
        elif self._in_lang_span and not self._current.get("language"):
            self._current["language"] = text
        elif self._in_stars_today and not self._current.get("stars_today"):
            self._current["stars_today"] = text.replace("stars today", "").strip()


class GitHubTrendingScanner:
    """Scrapes github.com/trending and returns daily trending repos."""

    def __init__(self, max_items: int = 20):
        self.max_items = max_items

    def scan(self) -> List[Dict[str, Any]]:
        """
        Fetch today's trending repos from github.com/trending.

        Returns:
            List of article-like dicts compatible with the blog/news format.
        """
        try:
            req = Request(_TRENDING_URL, headers={"User-Agent": _USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except URLError as e:
            logger.error(f"Failed to fetch GitHub trending: {e}")
            return []

        parser = _TrendingParser()
        parser.feed(html)

        items = []
        for repo in parser.repos[: self.max_items]:
            stars = repo.get("stars_today", "")
            language = repo.get("language", "")
            items.append(
                {
                    "source": "GitHub Trending",
                    "title": repo.get("title", ""),
                    "link": repo.get("link", ""),
                    "summary": repo.get("description", ""),
                    "published": "",
                    "stars": stars,
                    "language": language,
                }
            )

        logger.info(f"Found {len(items)} trending repos")
        return items
