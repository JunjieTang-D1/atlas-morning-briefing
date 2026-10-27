# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for arxiv_scanner module."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts import arxiv_scanner

# With deepxiv-sdk installed the module rebinds ArxivScanner to DeepXivScanner;
# these tests target the legacy arXiv API scanner either way.
ArxivScanner = getattr(arxiv_scanner, "_OriginalArxivScanner", arxiv_scanner.ArxivScanner)


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sample_xml() -> str:
    """Atom feed with a recent entry, a stale entry and an undated entry."""
    recent, old = _iso(1), _iso(400)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Evaluating Multi-Agent Systems</title>
    <summary>We propose a benchmark for agent evaluation.</summary>
    <published>{recent}</published>
    <updated>{recent}</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.AI"/>
    <category term="cs.MA"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Old Paper on Something</title>
    <summary>This is an old paper.</summary>
    <published>{old}</published>
    <updated>{old}</updated>
    <author><name>Charlie</name></author>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00003v1</id>
    <title>No Date Paper</title>
    <summary>Missing published date.</summary>
    <author><name>Diana</name></author>
  </entry>
</feed>"""


@pytest.fixture
def scanner():
    return ArxivScanner(topics=["Agent Evaluation"], days_back=7, max_results=10)


def _search(scanner, monkeypatch, body: str):
    """Run search_topic against a canned API response (no network)."""
    monkeypatch.setattr(scanner, "_throttled_get", lambda params: SimpleNamespace(text=body))
    return scanner.search_topic("Agent Evaluation")


class TestSearchTopicParsing:
    def test_parses_valid_entry(self, scanner, monkeypatch):
        papers = _search(scanner, monkeypatch, _sample_xml())
        paper = papers[0]
        assert paper["title"] == "Evaluating Multi-Agent Systems"
        assert paper["authors"] == ["Alice Smith", "Bob Jones"]
        assert "cs.AI" in paper["categories"]
        assert paper["pdf_link"] == "http://arxiv.org/pdf/2401.00001v1"
        assert paper["source"] == "arxiv_api"

    def test_filters_by_date(self, scanner, monkeypatch):
        titles = [p["title"] for p in _search(scanner, monkeypatch, _sample_xml())]
        assert "Evaluating Multi-Agent Systems" in titles
        assert "Old Paper on Something" not in titles

    def test_keeps_entries_without_date(self, scanner, monkeypatch):
        # An entry with neither <updated> nor <published> cannot be judged stale,
        # so the scanner keeps it rather than silently dropping it.
        titles = [p["title"] for p in _search(scanner, monkeypatch, _sample_xml())]
        assert "No Date Paper" in titles

    def test_handles_malformed_xml(self, scanner, monkeypatch):
        assert _search(scanner, monkeypatch, "not xml") == []

    def test_quotes_multi_word_topic(self, scanner, monkeypatch):
        seen = {}

        def fake_get(params):
            seen.update(params)
            return SimpleNamespace(text=_sample_xml())

        monkeypatch.setattr(scanner, "_throttled_get", fake_get)
        scanner.search_topic("Agent Evaluation")
        assert seen["search_query"] == 'all:"Agent Evaluation"'


class TestScanAllTopics:
    def test_deduplicates_papers(self, scanner, monkeypatch):
        paper = {
            "id": "http://arxiv.org/abs/2401.00001v1",
            "title": "Test",
            "summary": "",
            "authors": [],
            "published": "",
            "updated": "",
            "categories": [],
            "pdf_link": "",
            "arxiv_url": "",
        }
        monkeypatch.setattr(scanner, "search_topic", lambda topic: [paper, paper])
        scanner.topics = ["topic1", "topic2"]
        papers = scanner.scan_all_topics()
        assert len(papers) == 1
