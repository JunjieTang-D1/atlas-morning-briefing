# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for config_validator module."""

from scripts.config_validator import validate_config, check_environment


class TestValidateConfig:
    def test_valid_config(self):
        config = {
            "arxiv_topics": ["Agent Evaluation"],
            "blog_feeds": [{"name": "Test", "url": "http://example.com/rss"}],
            "stocks": ["AMZN"],
            "news_queries": ["AI"],
            "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
            "output_format": "kindle",
            "arxiv_days_back": 7,
            "max_papers": 20,
            "max_blogs": 10,
            "max_news": 15,
            "num_paper_picks": 3,
        }
        is_valid, messages = validate_config(config)
        assert is_valid is True

    def test_invalid_arxiv_topics_type(self):
        config = {"arxiv_topics": "not a list"}
        is_valid, messages = validate_config(config)
        assert is_valid is False
        assert any("arxiv_topics" in m for m in messages)

    def test_invalid_int_field(self):
        config = {"arxiv_topics": ["test"], "arxiv_days_back": "seven"}
        is_valid, messages = validate_config(config)
        assert is_valid is False
        assert any("arxiv_days_back" in m for m in messages)

    def test_invalid_blog_feed_missing_url(self):
        config = {
            "arxiv_topics": ["test"],
            "blog_feeds": [{"name": "Test"}],
        }
        is_valid, messages = validate_config(config)
        assert is_valid is False
        assert any("blog_feeds" in m for m in messages)

    def test_invalid_blog_feed_not_dict(self):
        config = {
            "arxiv_topics": ["test"],
            "blog_feeds": ["not a dict"],
        }
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_invalid_output_format(self):
        config = {"arxiv_topics": ["test"], "output_format": "tabloid"}
        is_valid, messages = validate_config(config)
        assert is_valid is False
        assert any("output_format" in m for m in messages)

    def test_invalid_paper_scoring_type(self):
        config = {"arxiv_topics": ["test"], "paper_scoring": "bad"}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_invalid_paper_scoring_value(self):
        config = {"arxiv_topics": ["test"], "paper_scoring": {"has_code": "five"}}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_invalid_pdf_config(self):
        config = {"arxiv_topics": ["test"], "pdf": "bad"}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_invalid_llm_config(self):
        config = {"arxiv_topics": ["test"], "llm": "bad"}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_warning_for_many_stocks(self):
        config = {
            "arxiv_topics": ["test"],
            "stocks": [f"TICK{i}" for i in range(35)],
        }
        is_valid, messages = validate_config(config)
        assert is_valid is True  # Warning, not error
        assert any("tickers" in m for m in messages)

    def test_warning_for_empty_topics(self):
        config = {"arxiv_topics": []}
        is_valid, messages = validate_config(config)
        assert is_valid is True  # Warning, not error
        assert any("empty" in m for m in messages)

    def test_empty_config(self):
        config = {}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_valid_llm_config(self):
        config = {
            "arxiv_topics": ["test"],
            "llm": {
                "enabled": True,
                "primary": {"base_url": "https://api.minimax.io/anthropic", "model": "MiniMax-M2.7"},
                "fallback": {"models": {"heavy": "some-model", "medium": "some-model", "light": "some-model"}},
            },
        }
        is_valid, messages = validate_config(config)
        assert is_valid is True


class TestCheckEnvironment:
    def test_warns_missing_finnhub(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        config = {"stocks": ["AMZN"]}
        warnings = check_environment(config)
        assert any("FINNHUB_API_KEY" in w for w in warnings)

    def test_warns_missing_brave(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        config = {"news_queries": ["AI"]}
        warnings = check_environment(config)
        assert any("BRAVE_API_KEY" in w for w in warnings)

    def test_warns_missing_gmail_not_dry_run(self, monkeypatch):
        monkeypatch.delenv("GMAIL_USER", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        config = {}
        warnings = check_environment(config, dry_run=False)
        assert any("GMAIL_USER" in w for w in warnings)

    def test_no_gmail_warning_on_dry_run(self, monkeypatch):
        monkeypatch.delenv("GMAIL_USER", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        config = {}
        warnings = check_environment(config, dry_run=True)
        assert not any("GMAIL_USER" in w for w in warnings)

    def test_no_warnings_when_no_features(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        config = {}  # No stocks or news configured
        warnings = check_environment(config, dry_run=True)
        assert len(warnings) == 0
