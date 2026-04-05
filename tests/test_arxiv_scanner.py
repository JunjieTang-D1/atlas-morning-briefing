# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for arxiv_scanner module."""

import pytest
from scripts.arxiv_scanner import ArxivScanner


SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Evaluating Multi-Agent Systems</title>
    <summary>We propose a benchmark for agent evaluation.</summary>
    <published>2026-03-05T00:00:00Z</published>
    <updated>2026-03-05T00:00:00Z</updated>
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
    <published>2020-01-01T00:00:00Z</published>
    <updated>2020-01-01T00:00:00Z</updated>
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


class TestParseArxivResponse:
    def test_parses_valid_entry(self, scanner):
        from datetime import datetime, timezone
        # Use a fixed date before the sample paper (2026-03-05) to avoid time-dependent failures
        start_date = datetime(2026, 3, 1, tzinfo=timezone.utc)
        papers = scanner._parse_arxiv_response(SAMPLE_ARXIV_XML, start_date)
        assert len(papers) >= 1
        paper = papers[0]
        assert paper["title"] == "Evaluating Multi-Agent Systems"
        assert paper["authors"] == ["Alice Smith", "Bob Jones"]
        assert "cs.AI" in paper["categories"]
        assert paper["pdf_link"] == "http://arxiv.org/pdf/2401.00001v1"

    def test_filters_by_date(self, scanner):
        from datetime import datetime, timedelta, timezone
        # Start date after the old paper
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        papers = scanner._parse_arxiv_response(SAMPLE_ARXIV_XML, start_date)
        titles = [p["title"] for p in papers]
        assert "Evaluating Multi-Agent Systems" in titles
        assert "Old Paper on Something" not in titles

    def test_skips_entries_without_date(self, scanner):
        from datetime import datetime, timezone
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        papers = scanner._parse_arxiv_response(SAMPLE_ARXIV_XML, start_date)
        titles = [p["title"] for p in papers]
        assert "No Date Paper" not in titles

    def test_handles_malformed_xml(self, scanner):
        from datetime import datetime, timezone
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        papers = scanner._parse_arxiv_response("not xml", start_date)
        assert papers == []


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
