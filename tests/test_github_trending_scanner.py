# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for GitHub trending scanner module."""

from io import BytesIO
from unittest.mock import patch, MagicMock

from scripts.github_trending_scanner import GitHubTrendingScanner


_SAMPLE_HTML = """
<html><body>
<article class="Box-row">
  <h2><a href="/anthropics/claude-code">anthropics / claude-code</a></h2>
  <p>CLI for Claude</p>
  <span itemprop="programmingLanguage">TypeScript</span>
  <span aria-label="1,234 stars today">1,234 stars today</span>
</article>
<article class="Box-row">
  <h2><a href="/langchain-ai/langgraph">langchain-ai / langgraph</a></h2>
  <p>Agent orchestration</p>
  <span itemprop="programmingLanguage">Python</span>
  <span aria-label="567 stars today">567 stars today</span>
</article>
</body></html>
"""


class TestGitHubTrendingScannerInit:
    def test_default_init(self):
        scanner = GitHubTrendingScanner()
        assert scanner.max_items == 20

    def test_custom_init(self):
        scanner = GitHubTrendingScanner(max_items=10)
        assert scanner.max_items == 10


class TestGitHubTrendingScan:
    @patch("scripts.github_trending_scanner.urlopen")
    def test_successful_scan(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _SAMPLE_HTML.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        scanner = GitHubTrendingScanner()
        items = scanner.scan()

        assert len(items) == 2
        assert items[0]["title"] == "anthropics/claude-code"
        assert items[0]["source"] == "GitHub Trending"
        assert items[0]["link"] == "https://github.com/anthropics/claude-code"

    @patch("scripts.github_trending_scanner.urlopen")
    def test_network_error_returns_empty(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection failed")

        scanner = GitHubTrendingScanner()
        assert scanner.scan() == []

    @patch("scripts.github_trending_scanner.urlopen")
    def test_max_items_respected(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _SAMPLE_HTML.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        scanner = GitHubTrendingScanner(max_items=1)
        items = scanner.scan()
        assert len(items) == 1

    @patch("scripts.github_trending_scanner.urlopen")
    def test_empty_html_returns_empty(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><body></body></html>"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        scanner = GitHubTrendingScanner()
        assert scanner.scan() == []
