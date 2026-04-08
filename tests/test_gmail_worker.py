"""Tests for GmailWorker."""

from unittest.mock import patch, MagicMock

from scripts.workers.gmail_worker import GmailWorker


class TestGmailWorkerInit:
    def test_default_disabled(self):
        worker = GmailWorker({})
        assert not worker.enabled
        assert worker.worker_name == "gmail_worker"

    def test_enabled_from_config(self):
        config = {"newsletter_source": {"enabled": True, "source_label": "Newsletters", "max_items": 5}}
        worker = GmailWorker(config)
        assert worker.enabled
        assert worker.source_label == "Newsletters"
        assert worker.max_items == 5


class TestGmailWorkerExecute:
    def test_disabled_returns_empty(self):
        worker = GmailWorker({})
        finding = worker.execute()
        assert finding["status"] == "success"
        assert finding["items"] == []
        assert finding["worker"] == "gmail_worker"

    @patch("scripts.workers.gmail_worker.GmailScanner")
    def test_successful_scan(self, mock_scanner_cls):
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = [
            {"title": "AI Weekly", "source": "newsletter@example.com", "summary": "AI news"},
        ]
        mock_scanner_cls.return_value = mock_scanner

        config = {"newsletter_source": {"enabled": True, "max_items": 10}}
        worker = GmailWorker(config)
        finding = worker.execute()

        assert finding["status"] == "success"
        assert len(finding["items"]) == 1
        assert finding["items"][0]["title"] == "AI Weekly"
        assert worker.scanner is mock_scanner

    @patch("scripts.workers.gmail_worker.GmailScanner")
    def test_scan_error(self, mock_scanner_cls):
        mock_scanner_cls.side_effect = Exception("IMAP connection refused")

        config = {"newsletter_source": {"enabled": True}}
        worker = GmailWorker(config)
        finding = worker.execute()

        assert finding["status"] == "error"
        assert "IMAP connection refused" in finding["error"]
