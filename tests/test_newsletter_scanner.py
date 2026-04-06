# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for newsletter scanner module."""

from unittest.mock import patch, MagicMock

import requests as req_lib
from scripts.newsletter_scanner import NewsletterScanner


class TestNewsletterScannerInit:
    def test_default_init(self):
        scanner = NewsletterScanner()
        assert scanner.max_items == 20
        assert scanner._fetched_ids == []

    def test_custom_init(self):
        scanner = NewsletterScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            max_items=10,
        )
        assert scanner.supabase_url == "https://test.supabase.co"
        assert scanner.supabase_key == "test-key"
        assert scanner.max_items == 10


class TestNewsletterScan:
    def test_no_credentials_returns_empty(self):
        scanner = NewsletterScanner(supabase_url="", supabase_key="")
        assert scanner.scan() == []

    @patch("scripts.newsletter_scanner.requests.get")
    def test_successful_scan(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "uuid-1",
                "subject": "AI Weekly",
                "from_name": "TechDigest",
                "from_address": "digest@tech.com",
                "summary": "Latest AI news",
                "category": "tech",
                "created_at": "2026-03-30T08:00:00Z",
            },
            {
                "id": "uuid-2",
                "subject": "ML Update",
                "from_name": None,
                "from_address": "ml@update.com",
                "summary": "ML roundup",
                "category": "ml",
                "created_at": "2026-03-30T09:00:00Z",
            },
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        scanner = NewsletterScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        items = scanner.scan()

        assert len(items) == 2
        assert items[0]["title"] == "AI Weekly"
        assert items[0]["source"] == "TechDigest"
        assert items[1]["source"] == "ml@update.com"  # Falls back to from_address
        assert scanner._fetched_ids == ["uuid-1", "uuid-2"]

        # Verify Supabase query parameters
        call_url = mock_get.call_args[0][0]
        assert "should_include_in_digest=eq.true" in call_url
        assert "digest_included_at=is.null" in call_url

    @patch("scripts.newsletter_scanner.requests.get")
    def test_api_error_returns_empty(self, mock_get):
        mock_get.side_effect = req_lib.ConnectionError("Connection failed")
        scanner = NewsletterScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        assert scanner.scan() == []


class TestMarkDigested:
    def test_no_ids_returns_true(self):
        scanner = NewsletterScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        assert scanner.mark_digested() is True

    @patch("scripts.newsletter_scanner.requests.patch")
    def test_successful_mark(self, mock_patch):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_patch.return_value = mock_resp

        scanner = NewsletterScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        scanner._fetched_ids = ["uuid-1", "uuid-2"]
        assert scanner.mark_digested() is True

        call_url = mock_patch.call_args[0][0]
        assert "id=in.(uuid-1,uuid-2)" in call_url

    @patch("scripts.newsletter_scanner.requests.patch")
    def test_failed_mark(self, mock_patch):
        mock_patch.side_effect = req_lib.ConnectionError("PATCH failed")
        scanner = NewsletterScanner(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        scanner._fetched_ids = ["uuid-1"]
        assert scanner.mark_digested() is False
