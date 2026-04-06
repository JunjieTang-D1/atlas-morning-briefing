# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for intelligence module."""

import pytest
from unittest.mock import MagicMock, patch
from scripts.intelligence import BriefingIntelligence, _parse_numbered_list


class TestExtractScore:
    def test_standard_format(self):
        score, text = BriefingIntelligence.extract_score("SCORE:4/5 Great paper on agents.")
        assert score == 4
        assert text == "Great paper on agents."

    def test_lowercase_variant(self):
        score, text = BriefingIntelligence.extract_score("Score: 3/5 Decent work.")
        assert score == 3
        assert text == "Decent work."

    def test_no_score(self):
        score, text = BriefingIntelligence.extract_score("Just a plain summary.")
        assert score is None
        assert text == "Just a plain summary."

    def test_empty_string(self):
        score, text = BriefingIntelligence.extract_score("")
        assert score is None
        assert text == ""


class TestParseRankedResponse:
    def test_basic_parsing(self):
        text = "[1] First item summary.\n[2] Second item summary."
        result = BriefingIntelligence._parse_ranked_response(text)
        assert len(result) == 2
        assert result[0] == (0, "First item summary.")
        assert result[1] == (1, "Second item summary.")

    def test_bold_markers(self):
        text = "**[1]** Bold first item.\n**[2]** Bold second."
        result = BriefingIntelligence._parse_ranked_response(text)
        assert len(result) == 2
        assert result[0][0] == 0
        assert "Bold first item" in result[0][1]

    def test_multiline_items(self):
        text = "[1] First line of item one.\nContinuation of item one.\n[2] Item two."
        result = BriefingIntelligence._parse_ranked_response(text)
        assert len(result) == 2
        assert "First line" in result[0][1]
        assert "Continuation" in result[0][1]

    def test_empty_input(self):
        assert BriefingIntelligence._parse_ranked_response("") == []

    def test_skips_empty_items(self):
        text = "[1] Real content.\n[2] \n[3] Also real."
        result = BriefingIntelligence._parse_ranked_response(text)
        # [2] has no content so it's skipped
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[1][0] == 2

    def test_numbered_sub_items_stripped(self):
        text = "[1] Summary here.\n1. Sub-point one.\n2. Sub-point two."
        result = BriefingIntelligence._parse_ranked_response(text)
        assert len(result) == 1
        assert "Sub-point one" in result[0][1]


class TestParseNumberedList:
    def test_bracket_format(self):
        text = "[1] First item.\n[2] Second item.\n[3] Third item."
        result = _parse_numbered_list(text, 3)
        assert len(result) == 3
        assert result[0] == "First item."

    def test_dot_format(self):
        text = "1. First.\n2. Second."
        result = _parse_numbered_list(text, 2)
        assert len(result) == 2
        assert result[0] == "First."

    def test_limits_to_expected(self):
        text = "[1] A\n[2] B\n[3] C\n[4] D"
        result = _parse_numbered_list(text, 2)
        assert len(result) == 2

    def test_multiline_item(self):
        text = "[1] Start of item.\nMore of the item.\n[2] Next."
        result = _parse_numbered_list(text, 2)
        assert len(result) == 2
        assert "Start of item. More of the item." == result[0]


@pytest.fixture
def mock_intelligence():
    """Create a BriefingIntelligence with mocked LLM client."""
    mock_llm = MagicMock()
    mock_llm.enabled = True
    mock_llm.invoke = MagicMock(return_value="THEME: Test emerging theme")
    intel = BriefingIntelligence(mock_llm, {"arxiv_topics": ["Agentic AI"]})
    return intel


class TestDetectEmergingThemesWithAllSources:
    def test_includes_newsletter_titles_in_prompt(self, mock_intelligence):
        newsletters = [{"title": "Newsletter about agent memory"}]
        github = [{"title": "awesome-agents"}]

        mock_intelligence.detect_emerging_themes(
            papers=[{"title": "Paper A"}],
            blogs=[{"title": "Blog B"}],
            news=[{"title": "News C"}],
            newsletters=newsletters,
            github_trending=github,
        )

        prompt = mock_intelligence.bedrock.invoke.call_args[0][0]
        assert "[newsletter] Newsletter about agent memory" in prompt
        assert "[github] awesome-agents" in prompt
        assert "newsletters" in prompt.lower()

    def test_works_without_optional_sources(self, mock_intelligence):
        result = mock_intelligence.detect_emerging_themes(
            papers=[{"title": "Paper A"}], blogs=[], news=[],
        )
        assert isinstance(result, list)


class TestTrackTrendingWithAllSources:
    def test_includes_newsletter_and_github_items(self, mock_intelligence):
        mock_intelligence.bedrock.invoke.return_value = "[1] NEW test-topic"
        newsletters = [{"title": "Newsletter trending topic"}]
        github = [{"title": "trending-repo"}]

        state = {"trending_topics": {}}
        mock_intelligence.track_trending(
            papers=[{"title": "Paper"}],
            blogs=[{"title": "Blog"}],
            news=[{"title": "News"}],
            state=state,
            newsletters=newsletters,
            github_trending=github,
        )

        prompt = mock_intelligence.bedrock.invoke.call_args[0][0]
        assert "[newsletter] Newsletter trending topic" in prompt
        assert "[github] trending-repo" in prompt

    def test_works_without_optional_sources(self, mock_intelligence):
        mock_intelligence.bedrock.invoke.return_value = ""
        state = {"trending_topics": {}}
        result = mock_intelligence.track_trending(
            papers=[{"title": "P"}], blogs=[], news=[], state=state,
        )
        assert len(result) == 4  # tuple of 4


class TestDetectEntityMentionsWithAllSources:
    def test_scans_newsletter_and_github_items(self, mock_intelligence):
        tracked = [{"name": "Anthropic", "type": "company"}]
        newsletters = [{"title": "Anthropic launches new feature", "summary": ""}]
        github = [{"title": "anthropic-sdk", "summary": "SDK by Anthropic"}]

        result = mock_intelligence.detect_entity_mentions(
            papers=[], blogs=[], news=[], tracked_entities=tracked,
            newsletters=newsletters, github_trending=github,
        )

        assert len(result) == 1
        assert result[0]["name"] == "Anthropic"
        assert result[0]["count"] >= 2  # found in newsletter title + github summary

    def test_works_without_optional_sources(self, mock_intelligence):
        tracked = [{"name": "OpenAI", "type": "company"}]
        result = mock_intelligence.detect_entity_mentions(
            papers=[{"title": "OpenAI paper", "summary": ""}],
            blogs=[], news=[], tracked_entities=tracked,
        )
        assert len(result) == 1


class TestRankSourceLinks:
    def test_returns_scored_items(self, mock_intelligence):
        mock_intelligence.bedrock.invoke.return_value = (
            "[1] SCORE:4/5 Highly relevant AI agent repo.\n"
            "[2] SCORE:2/5 Tangentially related."
        )
        items = [
            {"url": "https://github.com/org/agent-tool", "title": "agent-tool",
             "source": "GitHub Trending", "source_type": "github"},
            {"url": "https://example.com/article", "title": "Some article",
             "source": "Newsletter", "source_type": "newsletter"},
        ]
        result = mock_intelligence.rank_source_links(items, ["Agentic AI"])
        assert len(result) >= 1
        assert result[0]["score_combined"] == 4

    def test_empty_items_returns_empty(self, mock_intelligence):
        assert mock_intelligence.rank_source_links([], ["Agentic AI"]) == []

    def test_llm_failure_returns_empty(self, mock_intelligence):
        mock_intelligence.bedrock.invoke.return_value = None
        items = [{"url": "https://example.com", "title": "Test", "source": "Test"}]
        assert mock_intelligence.rank_source_links(items, ["AI"]) == []

    def test_mixed_types_scored_together(self, mock_intelligence):
        mock_intelligence.bedrock.invoke.return_value = (
            "[1] SCORE:5/5 Core AI framework.\n"
            "[2] SCORE:4/5 Good newsletter article.\n"
            "[3] SCORE:3/5 Tangential."
        )
        items = [
            {"url": "https://github.com/org/repo", "title": "AI Framework",
             "source": "GitHub Trending", "source_type": "github"},
            {"url": "https://blog.com/article", "title": "Newsletter Pick",
             "source": "The Batch", "source_type": "newsletter"},
            {"url": "https://other.com/thing", "title": "Other Link",
             "source": "Import AI", "source_type": "newsletter"},
        ]
        result = mock_intelligence.rank_source_links(items, ["AI"])
        assert len(result) == 3
        assert result[0]["score_combined"] == 5
        assert result[1]["score_combined"] == 4

    def test_handles_dict_topics(self, mock_intelligence):
        mock_intelligence.bedrock.invoke.return_value = "[1] SCORE:3/5 Relevant."
        items = [{"url": "https://ex.com", "title": "T", "source": "S"}]
        topics = [{"topic": "Agentic AI", "weight": 1.0}]
        result = mock_intelligence.rank_source_links(items, topics)
        prompt = mock_intelligence.bedrock.invoke.call_args[0][0]
        assert "Agentic AI" in prompt

    def test_not_available_returns_items_unchanged(self, mock_intelligence):
        mock_intelligence.bedrock.available = False
        items = [
            {"url": "https://ex.com", "title": "T", "source": "S"},
            {"url": "https://ex2.com", "title": "T2", "source": "S2"},
        ]
        result = mock_intelligence.rank_source_links(items, ["AI"])
        # Returns first 10 items unscored when LLM not available
        assert len(result) == 2


class TestCrossSourceSignalsWithAllSources:
    def test_detects_signals_from_newsletters(self, mock_intelligence):
        papers = [{"title": "Agent memory architectures for production"}]
        newsletters = [{"title": "Agent memory architectures in practice"}]

        result = mock_intelligence._detect_cross_source_signals(
            papers=papers, blogs=[], news=[],
            newsletters=newsletters, github_trending=[],
        )
        assert any("agent memory" in s.lower() for s in result)

    def test_works_without_optional_sources(self, mock_intelligence):
        result = mock_intelligence._detect_cross_source_signals(
            papers=[{"title": "test paper"}], blogs=[], news=[],
        )
        assert isinstance(result, list)
