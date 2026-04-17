#!/usr/bin/env python3
"""
Gmail scanner.

Fetches unread newsletter emails directly from Gmail via IMAP and returns
them as briefing source items. Uses GMAIL_USER + GMAIL_APP_PASSWORD credentials
(same as email_distributor.py for SMTP sending).
"""

import email
import email.message
import html
import imaplib
import logging
import os
import re
from email.header import decode_header, make_header
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_IMAP_HOST = "imap.gmail.com"
_IMAP_PORT = 993
_SNIPPET_MAX = 500

_LINK_NOISE = re.compile(
    r'(unsubscribe|track|pixel|click\.|utm_|mailto:|/login|/signin'
    r'|twitter\.com|x\.com|t\.co|linkedin\.com|facebook\.com|instagram\.com'
    r'|view.*browser|email.*client|manage.*pref)',
    re.IGNORECASE,
)

_LINK_TEXT_NOISE = re.compile(
    r'^(work with us|follow|subscribe|unsubscribe|view (in|online)|'
    r'manage pref|privacy|terms|advertise|sponsor|forward|tweet|share|'
    r'read online|contact us|become a member|refer a friend)',
    re.IGNORECASE,
)


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _decode_header_str(value: Optional[str]) -> str:
    """Decode a possibly RFC2047-encoded header value to a plain string."""
    if not value:
        return ""
    return str(make_header(decode_header(value)))


class _LinkExtractor(HTMLParser):
    """Extract (url, link_text) pairs from HTML, skipping noise links."""

    def __init__(self) -> None:
        super().__init__()
        self._current_href: Optional[str] = None
        self._current_text: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self._seen: set = set()

    def handle_starttag(self, tag: str, attrs: List) -> None:
        if tag == "a":
            attr = dict(attrs)
            href = attr.get("href", "").strip()
            if href.startswith("http") and not _LINK_NOISE.search(href) and len(href) >= 20:
                self._current_href = href
                self._current_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href:
            text = " ".join(self._current_text).strip()
            text = re.sub(r"\s+", " ", html.unescape(text)).strip()
            url = self._current_href
            if url not in self._seen and text and len(text) > 5 and not _LINK_TEXT_NOISE.search(text):
                self._seen.add(url)
                self.links.append((url, text))
            self._current_href = None
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)


def _extract_links_from_html(raw_html: str) -> List[Tuple[str, str]]:
    """Extract (url, title) pairs from newsletter HTML, filtering noise."""
    parser = _LinkExtractor()
    try:
        parser.feed(raw_html)
    except Exception:
        pass
    return parser.links


def _extract_body(msg: email.message.Message) -> str:
    """Extract plaintext body from a parsed email message."""
    # Prefer text/plain parts (skip empty)
    for part in msg.walk():
        if part.get_content_type() == "text/plain" and "attachment" not in part.get(
            "Content-Disposition", ""
        ):
            charset = part.get_content_charset() or "utf-8"
            try:
                text = part.get_payload(decode=True).decode(charset, errors="replace").strip()
                if text:
                    return text
            except Exception:
                pass

    # Fall back to text/html, stripping tags
    for part in msg.walk():
        if part.get_content_type() == "text/html" and "attachment" not in part.get(
            "Content-Disposition", ""
        ):
            charset = part.get_content_charset() or "utf-8"
            try:
                raw = part.get_payload(decode=True).decode(charset, errors="replace")
                return _strip_html(raw)
            except Exception:
                pass

    return ""


def _extract_html_part(msg: email.message.Message) -> str:
    """Return raw HTML body from the email, if present."""
    for part in msg.walk():
        if part.get_content_type() == "text/html" and "attachment" not in part.get(
            "Content-Disposition", ""
        ):
            charset = part.get_content_charset() or "utf-8"
            try:
                return part.get_payload(decode=True).decode(charset, errors="replace")
            except Exception:
                pass
    return ""


class GmailScanner:
    """Fetches unread newsletter emails from Gmail via IMAP."""

    def __init__(
        self,
        gmail_user: Optional[str] = None,
        gmail_password: Optional[str] = None,
        source_label: str = "INBOX",
        max_items: int = 20,
    ):
        """
        Initialize GmailScanner.

        Args:
            gmail_user: Gmail address. Falls back to GMAIL_USER env var.
            gmail_password: App password. Falls back to GMAIL_APP_PASSWORD env var.
            source_label: IMAP mailbox / Gmail label to scan (e.g. "INBOX",
                "Newsletters", "[Gmail]/All Mail").
            max_items: Maximum number of unread emails to fetch per run.
        """
        self.gmail_user = gmail_user or os.environ.get("GMAIL_USER", "")
        self.gmail_password = gmail_password or os.environ.get("GMAIL_APP_PASSWORD", "")
        self.source_label = source_label
        self.max_items = max_items
        self._fetched_nums: List[bytes] = []

    def scan(self) -> List[Dict[str, Any]]:
        """
        Fetch unread emails from the configured Gmail label.

        Returns:
            List of article-like dicts compatible with the blog/news format.
        """
        if not self.gmail_user or not self.gmail_password:
            logger.warning(
                "Gmail credentials not configured, skipping newsletter scan"
            )
            return []

        try:
            imap = imaplib.IMAP4_SSL(_IMAP_HOST, _IMAP_PORT)
            imap.login(self.gmail_user, self.gmail_password)
        except imaplib.IMAP4.error as e:
            logger.error(f"Gmail IMAP login failed: {e}")
            return []

        try:
            status, _ = imap.select(self.source_label, readonly=False)
            if status != "OK":
                logger.error(f"Cannot select mailbox '{self.source_label}'")
                return []

            _, data = imap.search(None, "UNSEEN")
            nums = data[0].split() if data[0] else []

            # Newest-first: IMAP numbers are ascending by arrival; take last N then reverse
            nums = nums[-self.max_items :][::-1]

            if not nums:
                logger.info(f"No unread emails in '{self.source_label}'")
                return []

            items = []
            for num in nums:
                try:
                    _, msg_data = imap.fetch(num, "(RFC822)")
                    raw = msg_data[0][1]
                    msg = email.message_from_bytes(raw)

                    subject = _decode_header_str(msg.get("Subject"))
                    from_full = _decode_header_str(msg.get("From"))
                    date_str = msg.get("Date", "")

                    # Friendly sender name: strip angle-bracket address if present
                    from_name = from_full.split("<")[0].strip().strip('"') or from_full

                    body = _extract_body(msg)
                    snippet = body[: _SNIPPET_MAX].strip()

                    # Extract (url, title) pairs from HTML for community picks
                    raw_html = _extract_html_part(msg)
                    links = _extract_links_from_html(raw_html) if raw_html else []

                    self._fetched_nums.append(num)
                    items.append(
                        {
                            "source": from_name,
                            "title": subject,
                            "link": "",
                            "summary": "",
                            "snippet": snippet,
                            "links": links,
                            "published": date_str,
                            "category": "",
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse email {num}: {e}")

            logger.info(
                f"Found {len(items)} unread emails in '{self.source_label}'"
            )
            return items

        finally:
            try:
                imap.close()
                imap.logout()
            except Exception:
                pass

    def mark_digested(self) -> bool:
        """
        Mark fetched emails as read (\\Seen) in Gmail.

        Should be called after the briefing has been successfully generated
        and distributed.

        Returns:
            True if successful, False otherwise.
        """
        if not self._fetched_nums:
            return True

        if not self.gmail_user or not self.gmail_password:
            return False

        try:
            imap = imaplib.IMAP4_SSL(_IMAP_HOST, _IMAP_PORT)
            imap.login(self.gmail_user, self.gmail_password)
            imap.select(self.source_label, readonly=False)
            for num in self._fetched_nums:
                imap.store(num, "+FLAGS", "\\Seen")
            imap.close()
            imap.logout()
            logger.info(f"Marked {len(self._fetched_nums)} emails as read")
            return True
        except Exception as e:
            logger.error(f"Failed to mark emails as read: {e}")
            return False
