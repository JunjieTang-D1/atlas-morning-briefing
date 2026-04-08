"""Tests for GitHubTrendingWorker."""

from unittest.mock import patch, MagicMock

from scripts.workers.github_trending_worker import GitHubTrendingWorker


class TestGitHubTrendingWorkerInit:
    def test_default_disabled(self):
        worker = GitHubTrendingWorker({})
        assert not worker.enabled
        assert worker.worker_name == "github_trending_worker"

    def test_enabled_from_config(self):
        config = {"github_trending": {"enabled": True, "max_items": 10}}
        worker = GitHubTrendingWorker(config)
        assert worker.enabled
        assert worker.max_items == 10


class TestGitHubTrendingWorkerExecute:
    def test_disabled_returns_empty(self):
        worker = GitHubTrendingWorker({})
        finding = worker.execute()
        assert finding["status"] == "success"
        assert finding["items"] == []

    @patch("scripts.workers.github_trending_worker.GitHubTrendingScanner")
    def test_successful_scan(self, mock_scanner_cls):
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = [
            {"source": "GitHub Trending", "title": "owner/repo", "link": "https://github.com/owner/repo"},
        ]
        mock_scanner_cls.return_value = mock_scanner

        config = {"github_trending": {"enabled": True, "max_items": 20}}
        worker = GitHubTrendingWorker(config)
        finding = worker.execute()

        assert finding["status"] == "success"
        assert len(finding["items"]) == 1

    @patch("scripts.workers.github_trending_worker.GitHubTrendingScanner")
    def test_scan_error(self, mock_scanner_cls):
        mock_scanner_cls.side_effect = Exception("Network error")

        config = {"github_trending": {"enabled": True}}
        worker = GitHubTrendingWorker(config)
        finding = worker.execute()

        assert finding["status"] == "error"
        assert "Network error" in finding["error"]
