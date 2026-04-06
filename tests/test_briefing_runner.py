# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for briefing_runner module."""

import pytest
from scripts.briefing_runner import BriefingRunner


@pytest.fixture
def minimal_config():
    return {
        "arxiv_topics": ["Agent Evaluation"],
        "blog_feeds": [],
        "stocks": [],
        "news_queries": [],
        "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
        "num_paper_picks": 2,
        "max_papers": 5,
        "arxiv_days_back": 7,
        "output_format": "kindle",
        "file_naming": "Personal-Briefing-{yyyy}.{mm}.{dd}",
        "pdf": {"font_size": 10, "line_spacing": 1.5},
        "llm": {"enabled": False},
    }


@pytest.fixture
def runner(minimal_config):
    return BriefingRunner(config=minimal_config, dry_run=True)


class TestDeduplicateNewsAndBlogs:
    def test_removes_duplicate_title(self, runner):
        news = [
            {"title": "Big AI News", "url": "http://news.com/1"},
            {"title": "Other News", "url": "http://news.com/2"},
        ]
        blogs = [
            {"title": "Big AI News", "link": "http://blog.com/big-ai"},
        ]
        deduped_news, _ = runner.deduplicate_news_and_blogs(news, blogs)
        assert len(deduped_news) == 1
        assert deduped_news[0]["title"] == "Other News"

    def test_removes_same_domain(self, runner):
        news = [
            {"title": "Anthropic Update", "url": "https://www.anthropic.com/news/update"},
            {"title": "Other News", "url": "http://other.com/1"},
        ]
        blogs = [
            {"title": "Blog Post", "link": "https://www.anthropic.com/blog/post"},
        ]
        deduped_news, _ = runner.deduplicate_news_and_blogs(news, blogs)
        assert len(deduped_news) == 1
        assert deduped_news[0]["title"] == "Other News"

    def test_no_blogs_returns_all_news(self, runner):
        news = [{"title": "News 1", "url": "http://a.com"}, {"title": "News 2", "url": "http://b.com"}]
        deduped_news, _ = runner.deduplicate_news_and_blogs(news, [])
        assert len(deduped_news) == 2

    def test_empty_inputs(self, runner):
        deduped_news, deduped_blogs = runner.deduplicate_news_and_blogs([], [])
        assert deduped_news == []
        assert deduped_blogs == []


class TestGenerateMarkdownBriefing:
    def test_generates_title(self, runner):
        md = runner.generate_markdown_briefing([], [], [], [], [])
        assert "Executive Summary" in md or md == ""  # title removed from markdown body

    def test_includes_stocks(self, runner):
        stocks = [{"symbol": "AMZN", "name": "Amazon", "current_price": 200.0, "change": 5.0, "percent_change": 2.5}]
        md = runner.generate_markdown_briefing([], [], stocks, [], [])
        assert "Financial Market Overview" in md
        assert "AMZN" in md
        assert "$200.00" in md

    def test_includes_stock_correlation(self, runner):
        stocks = [{
            "symbol": "NVDA", "name": "NVIDIA", "current_price": 100.0,
            "change": -5.0, "percent_change": -5.0,
            "news_correlation": "Export controls tightened",
        }]
        md = runner.generate_markdown_briefing([], [], stocks, [], [])
        assert "Export controls tightened" in md

    def test_includes_news(self, runner):
        news = [{"title": "AI Breakthrough", "url": "http://example.com", "source": "Reuters"}]
        md = runner.generate_markdown_briefing([], [], [], news, [])
        assert "AI & Tech News" in md
        assert "AI Breakthrough" in md

    def test_includes_blogs(self, runner):
        blogs = [{"title": "New Post", "source": "Anthropic", "link": "http://a.com", "summary": "Summary text"}]
        md = runner.generate_markdown_briefing([], blogs, [], [], [])
        assert "Blog Updates" in md
        assert "New Post" in md

    def test_includes_top_papers(self, runner):
        top_papers = [{
            "title": "Great Paper",
            "authors": ["Alice"],
            "score": 8.5,
            "score_combined": 4,
            "reproduction_difficulty": "S",
            "score_breakdown": {"has_code": True, "topic_match": 0.9, "recency": 0.95},
            "arxiv_url": "http://arxiv.org/abs/1",
            "pdf_link": "http://arxiv.org/pdf/1",
        }]
        md = runner.generate_markdown_briefing([], [], [], [], top_papers)
        assert "Top Papers" in md
        assert "Great Paper" in md

    def test_includes_paper_brief_summary(self, runner):
        top_papers = [{
            "title": "Paper",
            "authors": [],
            "score": 5.0,
            "score_combined": 4,
            "reproduction_difficulty": "M",
            "score_breakdown": {"has_code": False, "topic_match": 0.5, "recency": 0.5},
            "brief_summary": "This paper proposes a novel method.",
            "relevance_reason": "Directly matches agent evaluation",
            "arxiv_url": "",
            "pdf_link": "",
        }]
        md = runner.generate_markdown_briefing([], [], [], [], top_papers)
        assert "This paper proposes a novel method." in md

    def test_includes_synthesis(self, runner):
        synthesis = {
            "editorial_intro": "Today's briefing highlights a surge in agent evaluation papers.",
        }
        md = runner.generate_markdown_briefing([], [], [], [], [], synthesis)
        assert "Today's briefing highlights" in md
        assert "Executive Summary" in md

    def test_intelligence_badge_when_disabled(self, runner):
        md = runner.generate_markdown_briefing([], [], [], [], [])
        assert "Amazon Bedrock" not in md

    def test_includes_errors(self, runner):
        runner.errors = ["ArXiv scan failed"]
        md = runner.generate_markdown_briefing([], [], [], [], [])
        assert "Errors" in md
        assert "ArXiv scan failed" in md


class TestStatus:
    def test_initial_status(self, runner):
        assert runner.status["papers_found"] == 0
        assert runner.status["intelligence_enabled"] is False
        assert runner.status["pdf_generated"] is False

    def test_save_status(self, runner, tmp_path):
        runner.save_status(str(tmp_path))
        import json
        status_path = tmp_path / "status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert "timestamp" in status
        assert "elapsed_seconds" in status
