# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for GitHub trending scanner module."""

from unittest.mock import patch, MagicMock

import requests as req_lib
from scripts.github_trending_scanner import GitHubTrendingScanner


class TestGitHubTrendingScannerInit:
    def test_default_init(self):
        scanner = GitHubTrendingScanner()
        assert scanner.max_items == 20

    def test_custom_init(self):
        scanner = GitHubTrendingScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            max_items=10,
        )
        assert scanner.supabase_url == "https://test.supabase.co"
        assert scanner.max_items == 10


class TestGitHubTrendingScan:
    def test_no_credentials_returns_empty(self):
        scanner = GitHubTrendingScanner(supabase_url="", supabase_key="")
        assert scanner.scan() == []

    @patch("scripts.github_trending_scanner.requests.get")
    def test_successful_scan(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "title": "anthropics/claude-code",
                "summary": "CLI for Claude",
                "url": "https://github.com/anthropics/claude-code",
                "metadata": {"stars": "5000", "language": "TypeScript"},
                "created_at": "2026-03-30T08:00:00Z",
            },
            {
                "title": "langchain-ai/langgraph",
                "summary": "Agent orchestration",
                "url": "https://github.com/langchain-ai/langgraph",
                "metadata": {"stars": "3000", "language": "Python"},
                "created_at": "2026-03-30T09:00:00Z",
            },
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        scanner = GitHubTrendingScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        items = scanner.scan()

        assert len(items) == 2
        assert items[0]["title"] == "anthropics/claude-code"
        assert items[0]["stars"] == "5000"
        assert items[0]["language"] == "TypeScript"
        assert items[0]["source"] == "GitHub Trending"

        call_url = mock_get.call_args[0][0]
        assert "source=eq.github_trending" in call_url

    @patch("scripts.github_trending_scanner.requests.get")
    def test_api_error_returns_empty(self, mock_get):
        mock_get.side_effect = req_lib.ConnectionError("Connection failed")
        scanner = GitHubTrendingScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        assert scanner.scan() == []

    @patch("scripts.github_trending_scanner.requests.get")
    def test_missing_metadata_handled(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "title": "some/repo",
                "summary": "desc",
                "url": "https://github.com/some/repo",
                "metadata": None,
                "created_at": "2026-03-30T08:00:00Z",
            },
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        scanner = GitHubTrendingScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        items = scanner.scan()
        assert len(items) == 1
        assert items[0]["stars"] == ""
        assert items[0]["language"] == ""
