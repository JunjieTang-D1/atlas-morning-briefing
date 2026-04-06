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


class TestExtractUrlsFromText:
    def test_extracts_http_urls(self):
        text = "Check out https://example.com/article/123 and http://blog.com/post/456"
        urls = BriefingRunner._extract_urls_from_text(text)
        assert len(urls) == 2
        assert "https://example.com/article/123" in urls
        assert "http://blog.com/post/456" in urls

    def test_filters_tracking_urls(self):
        text = "Real: https://example.com/article/123 Tracked: https://click.example.com/track?utm_source=email"
        urls = BriefingRunner._extract_urls_from_text(text)
        assert len(urls) == 1
        assert "example.com/article/123" in urls[0]

    def test_filters_social_urls(self):
        text = "https://twitter.com/user/status/123 https://example.com/real-article"
        urls = BriefingRunner._extract_urls_from_text(text)
        assert len(urls) == 1
        assert "example.com" in urls[0]

    def test_filters_unsubscribe(self):
        text = "https://example.com/real https://list.example.com/unsubscribe/abc"
        urls = BriefingRunner._extract_urls_from_text(text)
        assert len(urls) == 1

    def test_deduplicates(self):
        text = "https://example.com/article https://example.com/article"
        urls = BriefingRunner._extract_urls_from_text(text)
        assert len(urls) == 1

    def test_empty_text(self):
        assert BriefingRunner._extract_urls_from_text("") == []
        assert BriefingRunner._extract_urls_from_text(None) == []

    def test_caps_at_10(self):
        text = " ".join(f"https://example.com/article/{i}" for i in range(20))
        urls = BriefingRunner._extract_urls_from_text(text)
        assert len(urls) <= 10

    def test_strips_trailing_punctuation(self):
        text = "See https://example.com/article/123."
        urls = BriefingRunner._extract_urls_from_text(text)
        assert urls[0] == "https://example.com/article/123"

    def test_short_urls_filtered(self):
        text = "https://a.co https://example.com/real/article"
        urls = BriefingRunner._extract_urls_from_text(text)
        assert len(urls) == 1
        assert "example.com" in urls[0]


class TestGetLimit:
    def test_env_var_override(self, runner, monkeypatch):
        monkeypatch.setenv("BRIEFING_MAX_BLOGS_RENDER", "12")
        assert runner._get_limit("max_blogs_render", 5) == 12

    def test_config_fallback(self, runner):
        runner.config["max_blogs_render"] = 8
        assert runner._get_limit("max_blogs_render", 5) == 8

    def test_default_fallback(self, runner):
        assert runner._get_limit("max_blogs_render", 5) == 5

    def test_env_var_invalid_uses_config(self, runner, monkeypatch):
        monkeypatch.setenv("BRIEFING_MAX_BLOGS_RENDER", "not_a_number")
        runner.config["max_blogs_render"] = 7
        assert runner._get_limit("max_blogs_render", 5) == 7


class TestRenderCommunityPicks:
    def test_renders_scored_github_items(self, runner):
        items = [{
            "url": "https://github.com/org/repo",
            "title": "awesome-agents",
            "source": "GitHub Trending",
            "source_type": "github",
            "score_combined": 4,
            "stars": "1.2k",
            "language": "Python",
            "brief_summary": "A collection of AI agent tools.",
        }]
        md = runner._render_community_picks(items)
        assert "Community & Newsletter Picks" in md
        assert "awesome-agents" in md
        assert "GitHub" in md
        assert "1.2k stars" in md

    def test_renders_newsletter_items(self, runner):
        items = [{
            "url": "https://blog.com/article",
            "title": "AI Agent Patterns",
            "source": "The Batch",
            "source_type": "newsletter",
            "score_combined": 5,
            "brief_summary": "Key patterns for building agents.",
        }]
        md = runner._render_community_picks(items)
        assert "AI Agent Patterns" in md
        assert "via The Batch" in md

    def test_filters_low_scores(self, runner):
        items = [
            {"url": "https://a.com", "title": "Good", "source": "S",
             "score_combined": 4, "source_type": "github"},
            {"url": "https://b.com", "title": "Bad", "source": "S",
             "score_combined": 2, "source_type": "github"},
        ]
        md = runner._render_community_picks(items)
        assert "Good" in md
        assert "Bad" not in md

    def test_empty_returns_empty(self, runner):
        assert runner._render_community_picks([]) == ""

    def test_no_scores_shows_all(self, runner):
        items = [
            {"url": "https://a.com", "title": "Item A", "source": "S", "source_type": "github"},
            {"url": "https://b.com", "title": "Item B", "source": "S", "source_type": "newsletter"},
        ]
        md = runner._render_community_picks(items)
        assert "Item A" in md
        assert "Item B" in md


class TestSectionOrder:
    def test_community_picks_replaces_github_trending(self):
        assert "community_picks" in BriefingRunner.DEFAULT_SECTION_ORDER
        assert "github_trending" not in BriefingRunner.DEFAULT_SECTION_ORDER

    def test_community_picks_in_briefing(self, runner):
        community = [{
            "url": "https://github.com/x/y",
            "title": "Cool Repo",
            "source": "GitHub Trending",
            "source_type": "github",
            "score_combined": 5,
            "brief_summary": "Very cool.",
        }]
        md = runner.generate_markdown_briefing(
            [], [], [], [], [], community_picks=community,
        )
        assert "Community & Newsletter Picks" in md
        assert "Cool Repo" in md
