# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for Gmail scanner module."""

import imaplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

from scripts.gmail_scanner import GmailScanner


def _make_raw_email(
    subject: str = "Test Newsletter",
    from_addr: str = "Sender Name <sender@example.com>",
    date: str = "Mon, 07 Apr 2026 10:00:00 +0000",
    body: str = "This is the email body content.",
    html_body: str | None = None,
) -> bytes:
    """Build a minimal RFC822 email as bytes."""
    if html_body:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
    else:
        msg = MIMEText(body, "plain")

    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["Date"] = date
    return msg.as_bytes()


def _imap_mock(unseen_nums: list[bytes], raw_emails: list[bytes]) -> MagicMock:
    """Build a mock IMAP4_SSL instance for the given unseen message numbers."""
    imap = MagicMock()
    imap.login.return_value = ("OK", [b"Logged in"])
    imap.select.return_value = ("OK", [b"1 EXISTS"])
    imap.search.return_value = ("OK", [b" ".join(unseen_nums)] if unseen_nums else [b""])

    def fetch_side_effect(num, _fmt):
        idx = unseen_nums.index(num)
        return ("OK", [(b"FLAGS", raw_emails[idx])])

    imap.fetch.side_effect = fetch_side_effect
    imap.store.return_value = ("OK", [])
    imap.close.return_value = ("OK", [])
    imap.logout.return_value = ("BYE", [])
    return imap


class TestGmailScannerInit:
    def test_default_init(self):
        scanner = GmailScanner()
        assert scanner.max_items == 20
        assert scanner.source_label == "INBOX"
        assert scanner._fetched_nums == []

    def test_custom_init(self):
        scanner = GmailScanner(
            gmail_user="user@gmail.com",
            gmail_password="secret",
            source_label="Newsletters",
            max_items=50,
        )
        assert scanner.gmail_user == "user@gmail.com"
        assert scanner.source_label == "Newsletters"
        assert scanner.max_items == 50


class TestGmailScan:
    def test_no_credentials_returns_empty(self):
        scanner = GmailScanner(gmail_user="", gmail_password="")
        assert scanner.scan() == []

    @patch("scripts.gmail_scanner.imaplib.IMAP4_SSL")
    def test_login_failure_returns_empty(self, mock_ssl):
        imap = MagicMock()
        imap.login.side_effect = imaplib.IMAP4.error("authentication failed")
        mock_ssl.return_value = imap

        scanner = GmailScanner(gmail_user="u@g.com", gmail_password="pw")
        assert scanner.scan() == []

    @patch("scripts.gmail_scanner.imaplib.IMAP4_SSL")
    def test_no_unread_returns_empty(self, mock_ssl):
        imap = MagicMock()
        imap.login.return_value = ("OK", [])
        imap.select.return_value = ("OK", [b"0 EXISTS"])
        imap.search.return_value = ("OK", [b""])
        mock_ssl.return_value = imap

        scanner = GmailScanner(gmail_user="u@g.com", gmail_password="pw")
        assert scanner.scan() == []

    @patch("scripts.gmail_scanner.imaplib.IMAP4_SSL")
    def test_successful_scan_returns_items(self, mock_ssl):
        raw = _make_raw_email(
            subject="AI Weekly",
            from_addr="TechDigest <digest@tech.com>",
            body="Top AI stories this week.",
        )
        nums = [b"1"]
        mock_ssl.return_value = _imap_mock(nums, [raw])

        scanner = GmailScanner(gmail_user="u@g.com", gmail_password="pw")
        items = scanner.scan()

        assert len(items) == 1
        assert items[0]["title"] == "AI Weekly"
        assert items[0]["source"] == "TechDigest"
        assert "AI stories" in items[0]["snippet"]
        assert items[0]["link"] == ""
        assert items[0]["summary"] == ""
        assert scanner._fetched_nums == [b"1"]

    @patch("scripts.gmail_scanner.imaplib.IMAP4_SSL")
    def test_respects_max_items(self, mock_ssl):
        nums = [b"1", b"2", b"3", b"4", b"5"]
        raws = [_make_raw_email(subject=f"NL {i}", body=f"Body {i}") for i in range(5)]
        mock_ssl.return_value = _imap_mock(nums, raws)

        scanner = GmailScanner(gmail_user="u@g.com", gmail_password="pw", max_items=2)
        items = scanner.scan()

        # Takes last 2 (newest) and reverses: nums [4, 5] → fetched as [5, 4]
        assert len(items) == 2

    @patch("scripts.gmail_scanner.imaplib.IMAP4_SSL")
    def test_html_fallback_strips_tags(self, mock_ssl):
        raw = _make_raw_email(
            subject="HTML Newsletter",
            from_addr="sender@example.com",
            body="",
            html_body="<h1>Hello</h1><p>This is <b>bold</b> content.</p>",
        )
        mock_ssl.return_value = _imap_mock([b"1"], [raw])

        scanner = GmailScanner(gmail_user="u@g.com", gmail_password="pw")
        items = scanner.scan()

        assert len(items) == 1
        assert "<h1>" not in items[0]["snippet"]
        assert "Hello" in items[0]["snippet"]

    @patch("scripts.gmail_scanner.imaplib.IMAP4_SSL")
    def test_snippet_truncated_to_max(self, mock_ssl):
        long_body = "x" * 1000
        raw = _make_raw_email(body=long_body)
        mock_ssl.return_value = _imap_mock([b"1"], [raw])

        scanner = GmailScanner(gmail_user="u@g.com", gmail_password="pw")
        items = scanner.scan()

        assert len(items[0]["snippet"]) <= 500


class TestMarkDigested:
    def test_no_fetched_nums_returns_true(self):
        scanner = GmailScanner(gmail_user="u@g.com", gmail_password="pw")
        assert scanner.mark_digested() is True

    def test_no_credentials_returns_false(self):
        scanner = GmailScanner(gmail_user="", gmail_password="")
        scanner._fetched_nums = [b"1"]
        assert scanner.mark_digested() is False

    @patch("scripts.gmail_scanner.imaplib.IMAP4_SSL")
    def test_marks_emails_as_seen(self, mock_ssl):
        imap = MagicMock()
        imap.login.return_value = ("OK", [])
        imap.select.return_value = ("OK", [b"1 EXISTS"])
        imap.store.return_value = ("OK", [])
        mock_ssl.return_value = imap

        scanner = GmailScanner(gmail_user="u@g.com", gmail_password="pw")
        scanner._fetched_nums = [b"1", b"2"]
        result = scanner.mark_digested()

        assert result is True
        imap.store.assert_any_call(b"1", "+FLAGS", "\\Seen")
        imap.store.assert_any_call(b"2", "+FLAGS", "\\Seen")

    @patch("scripts.gmail_scanner.imaplib.IMAP4_SSL")
    def test_imap_error_returns_false(self, mock_ssl):
        imap = MagicMock()
        imap.login.side_effect = imaplib.IMAP4.error("login failed")
        mock_ssl.return_value = imap

        scanner = GmailScanner(gmail_user="u@g.com", gmail_password="pw")
        scanner._fetched_nums = [b"1"]
        assert scanner.mark_digested() is False
