"""Tests for obsidian_writer module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from scripts.obsidian_writer import ObsidianWriter


@pytest.fixture
def writer():
    return ObsidianWriter(
        api_url="http://localhost:27123",
        api_key="test-key",
        config={"briefing_folder": "Sources/Briefings", "trending_threshold": 3},
    )


@pytest.fixture
def sample_status():
    return {
        "papers_found": 10,
        "blogs_found": 5,
        "news_found": 8,
        "intelligence_enabled": True,
    }


@pytest.fixture
def sample_entity_mentions():
    return [
        {
            "name": "Anthropic",
            "type": "company",
            "count": 5,
            "example_titles": ["Anthropic releases new model", "Claude update"],
        },
        {
            "name": "OpenAI",
            "type": "company",
            "count": 3,
            "example_titles": ["GPT-5 benchmarks"],
        },
    ]


@pytest.fixture
def sample_date():
    return datetime(2026, 4, 5)


class TestEncoding:
    def test_encode_simple_path(self, writer):
        assert writer._encode_path("Wiki/Entities/Anthropic.md") == "Wiki/Entities/Anthropic.md"

    def test_encode_path_with_spaces(self, writer):
        assert writer._encode_path("My Folder/My Note.md") == "My%20Folder/My%20Note.md"

    def test_encode_path_with_special_chars(self, writer):
        result = writer._encode_path("Wiki/Concepts/AI & ML.md")
        assert "%26" in result


class TestVaultName:
    def test_simple_name(self, writer):
        assert writer._to_vault_name("Agent Memory Architectures") == "Agent-Memory-Architectures"

    def test_hyphenated_input(self, writer):
        assert writer._to_vault_name("agent-safety-reliability") == "Agent-Safety-Reliability"

    def test_name_with_dots(self, writer):
        assert writer._to_vault_name("claude 3.5") == "Claude-3.5"

    def test_name_with_special_chars(self, writer):
        result = writer._to_vault_name("AI/ML & Data")
        assert ".." not in result
        assert "/" not in result

    def test_strips_trailing_hyphens(self, writer):
        assert writer._to_vault_name("  test  ") == "Test"


class TestFrontmatter:
    def test_build_frontmatter(self, writer):
        fm = writer._build_frontmatter({"type": "test", "tags": ["a", "b"]})
        assert fm.startswith("---\n")
        assert fm.endswith("---\n")
        assert "type: test" in fm

    def test_extract_frontmatter(self, writer):
        content = "---\ntype: test\ntags:\n  - a\n---\n\n# Body"
        fm = writer._extract_frontmatter(content)
        assert fm["type"] == "test"
        assert fm["tags"] == ["a"]

    def test_extract_frontmatter_none(self, writer):
        assert writer._extract_frontmatter("no frontmatter") == {}

    def test_extract_body(self, writer):
        content = "---\ntype: test\n---\n\n# Body here"
        body = writer._extract_body(content)
        assert body.startswith("# Body here")

    def test_extract_body_no_frontmatter(self, writer):
        assert writer._extract_body("just text") == "just text"

    def test_roundtrip(self, writer):
        fields = {"type": "wiki/entity", "created": "2026-04-05", "tags": ["wiki"]}
        fm_str = writer._build_frontmatter(fields)
        body = "# Test\n\nContent here.\n"
        full = fm_str + "\n" + body
        extracted_fm = writer._extract_frontmatter(full)
        extracted_body = writer._extract_body(full)
        assert extracted_fm["type"] == "wiki/entity"
        assert "Content here." in extracted_body


class TestRestHelpers:
    @patch("scripts.obsidian_writer.requests.get")
    def test_get_note_returns_content(self, mock_get, writer):
        mock_get.return_value = MagicMock(status_code=200, text="# Note", raise_for_status=lambda: None)
        assert writer._get_note("test.md") == "# Note"

    @patch("scripts.obsidian_writer.requests.get")
    def test_get_note_returns_none_on_404(self, mock_get, writer):
        mock_get.return_value = MagicMock(status_code=404)
        assert writer._get_note("missing.md") is None

    @patch("scripts.obsidian_writer.requests.get")
    def test_get_note_returns_none_on_error(self, mock_get, writer):
        import requests as req
        mock_get.side_effect = req.ConnectionError("refused")
        assert writer._get_note("test.md") is None

    @patch("scripts.obsidian_writer.requests.put")
    def test_put_note_returns_true(self, mock_put, writer):
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        assert writer._put_note("test.md", "content") is True

    @patch("scripts.obsidian_writer.requests.put")
    def test_put_note_returns_false_on_error(self, mock_put, writer):
        import requests as req
        mock_put.side_effect = req.ConnectionError("refused")
        assert writer._put_note("test.md", "content") is False



class TestFrontmatterEdgeCases:
    def test_extract_frontmatter_unclosed(self, writer):
        assert writer._extract_frontmatter("---\ntype: test\nno closing") == {}

    def test_extract_frontmatter_invalid_yaml(self, writer):
        assert writer._extract_frontmatter("---\n: :\n  bad: [yaml\n---\n") == {}

    def test_extract_frontmatter_empty_yaml(self, writer):
        assert writer._extract_frontmatter("---\n---\n") == {}

    def test_extract_body_unclosed_frontmatter(self, writer):
        content = "---\ntype: test\nno closing marker"
        assert writer._extract_body(content) == content


class TestWriteDailyBriefing:
    @patch("scripts.obsidian_writer.requests.put")
    def test_writes_correct_path(self, mock_put, writer, sample_date, sample_status):
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        result = writer.write_daily_briefing(
            "# Briefing", sample_date, sample_status, ["theme1"], [], [],
        )
        assert result is True
        call_url = mock_put.call_args[0][0]
        assert "Sources/Briefings/2026/04/Personal-Briefing-2026-04-05.md" in call_url

    @patch("scripts.obsidian_writer.requests.put")
    def test_frontmatter_contains_metadata(self, mock_put, writer, sample_date, sample_status):
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        writer.write_daily_briefing(
            "# Briefing", sample_date, sample_status, ["emerging theme"], [], [],
        )
        content = mock_put.call_args[1].get("data") or mock_put.call_args[0][1] if len(mock_put.call_args[0]) > 1 else None
        if content is None:
            content = mock_put.call_args.kwargs.get("data", b"")
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        assert "type: source/briefing" in content
        assert "papers-found: 10" in content
        assert "emerging theme" in content

    @patch("scripts.obsidian_writer.requests.put")
    def test_includes_entity_wikilinks(self, mock_put, writer, sample_date, sample_status, sample_entity_mentions):
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        writer.write_daily_briefing(
            "# Briefing", sample_date, sample_status, [], [], sample_entity_mentions,
        )
        content = mock_put.call_args.kwargs.get("data", b"").decode("utf-8")
        assert "[[Anthropic]]" in content
        assert "[[Openai]]" in content

    @patch("scripts.obsidian_writer.requests.put")
    def test_connection_error_returns_false(self, mock_put, writer, sample_date, sample_status):
        import requests as req
        mock_put.side_effect = req.ConnectionError("refused")
        result = writer.write_daily_briefing(
            "# Briefing", sample_date, sample_status, [], [], [],
        )
        assert result is False


class TestUpdateEntityTimelines:
    @patch("scripts.obsidian_writer.requests.put")
    @patch("scripts.obsidian_writer.requests.get")
    def test_appends_to_existing_entity(self, mock_get, mock_put, writer, sample_date):
        existing = "---\ntype: wiki/entity\n---\n\n# Anthropic\n\nA company.\n"
        mock_get.return_value = MagicMock(status_code=200, text=existing, raise_for_status=lambda: None)
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)

        mentions = [{"name": "Anthropic", "count": 5, "example_titles": ["New model release"]}]
        results = writer.update_entity_timelines(mentions, sample_date, "Personal-Briefing-2026-04-05")

        assert results["Anthropic"] is True
        content = mock_put.call_args.kwargs.get("data", b"").decode("utf-8")
        assert "### 2026-04-05" in content
        assert "[[Personal-Briefing-2026-04-05]]" in content
        assert "Mentioned 5 times" in content

    @patch("scripts.obsidian_writer.requests.get")
    def test_skips_nonexistent_entity(self, mock_get, writer, sample_date):
        mock_get.return_value = MagicMock(status_code=404)
        mentions = [{"name": "Unknown", "count": 1, "example_titles": []}]
        results = writer.update_entity_timelines(mentions, sample_date, "Personal-Briefing-2026-04-05")
        assert results == {}  # skipped entirely, not in results

    @patch("scripts.obsidian_writer.requests.get")
    def test_duplicate_day_guard(self, mock_get, writer, sample_date):
        existing = "---\ntype: wiki/entity\n---\n\n# Test\n\n### 2026-04-05\n- Already here\n"
        mock_get.return_value = MagicMock(status_code=200, text=existing, raise_for_status=lambda: None)

        mentions = [{"name": "Test", "count": 1, "example_titles": []}]
        results = writer.update_entity_timelines(mentions, sample_date, "Personal-Briefing-2026-04-05")
        assert results["Test"] is True  # reported success, no PUT needed

    @patch("scripts.obsidian_writer.requests.get")
    def test_empty_mentions_returns_empty(self, mock_get, writer, sample_date):
        results = writer.update_entity_timelines([], sample_date, "Personal-Briefing-2026-04-05")
        assert results == {}
        mock_get.assert_not_called()


class TestPromoteConcepts:
    @patch("scripts.obsidian_writer.requests.put")
    @patch("scripts.obsidian_writer.requests.get")
    def test_creates_new_concept_at_threshold(self, mock_get, mock_put, writer, sample_date):
        mock_get.return_value = MagicMock(status_code=404)
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)

        trending = {
            "agent-memory": {"first_seen": "2026-04-01", "count": 3, "last_seen": "2026-04-05"},
        }
        themes = ["Agent memory architectures are gaining traction"]
        results = writer.promote_concepts(trending, themes, sample_date, "Personal-Briefing-2026-04-05")

        assert results["agent-memory"] is True
        content = mock_put.call_args.kwargs.get("data", b"").decode("utf-8")
        assert "type: wiki/concept" in content
        assert "auto-promoted" in content
        assert "agent memory" in content.lower()

    @patch("scripts.obsidian_writer.requests.get")
    def test_skips_below_threshold(self, mock_get, writer, sample_date):
        trending = {
            "short-lived": {"first_seen": "2026-04-04", "count": 2, "last_seen": "2026-04-05"},
        }
        results = writer.promote_concepts(trending, [], sample_date, "Personal-Briefing-2026-04-05")
        assert results == {}
        mock_get.assert_not_called()

    @patch("scripts.obsidian_writer.requests.put")
    @patch("scripts.obsidian_writer.requests.get")
    def test_appends_to_existing_concept(self, mock_get, mock_put, writer, sample_date):
        existing = "---\ntype: wiki/concept\ncreated: '2026-04-01'\nupdated: '2026-04-03'\ndetection-count: 3\nsources:\n- '[[Personal-Briefing-2026-04-03]]'\ntags:\n- wiki\n- concept\n- auto-promoted\n---\n\n# Agent Memory\n\nSome content.\n"
        mock_get.return_value = MagicMock(status_code=200, text=existing, raise_for_status=lambda: None)
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)

        trending = {
            "agent-memory": {"first_seen": "2026-04-01", "count": 4, "last_seen": "2026-04-05"},
        }
        results = writer.promote_concepts(trending, [], sample_date, "Personal-Briefing-2026-04-05")

        assert results["agent-memory"] is True
        content = mock_put.call_args.kwargs.get("data", b"").decode("utf-8")
        assert "### 2026-04-05" in content
        assert "[[Personal-Briefing-2026-04-05]]" in content


class TestWriteWeeklySynthesis:
    @patch("scripts.obsidian_writer.requests.put")
    def test_writes_weekly_note(self, mock_put, writer):
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        date = datetime(2026, 4, 4)  # Saturday

        result = writer.write_weekly_synthesis(
            "# Weekly Deep Dive\n\nContent here.",
            date,
            ["Personal-Briefing-2026-03-30", "Personal-Briefing-2026-04-04"],
        )
        assert result is True
        call_url = mock_put.call_args[0][0]
        assert "Weekly-Digest-2026-W14" in call_url
        content = mock_put.call_args.kwargs.get("data", b"").decode("utf-8")
        assert "type: wiki/synthesis" in content
        assert "weekly-digest" in content
        assert "[[Personal-Briefing-2026-03-30]]" in content

    def test_empty_content_returns_false(self, writer):
        result = writer.write_weekly_synthesis("", datetime(2026, 4, 4), [])
        assert result is False


class TestPublish:
    @patch("scripts.obsidian_writer.requests.put")
    @patch("scripts.obsidian_writer.requests.get")
    def test_orchestrates_all_operations(self, mock_get, mock_put, writer, sample_date, sample_status):
        mock_get.return_value = MagicMock(status_code=404)
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)

        results = writer.publish(
            markdown_content="# Briefing",
            date=sample_date,
            status=sample_status,
            emerging_themes=["theme1"],
            top_papers=[],
            entity_mentions=[],
            trending_topics={},
            weekly_deep_dive="",
            briefing_name="Personal-Briefing-2026-04-05",
            weekly_briefing_names=[],
        )
        assert results["briefing"] is True
        assert results["entities"] == {}
        assert results["concepts"] == {}
        assert results["weekly_synthesis"] is None

    @patch("scripts.obsidian_writer.requests.put")
    @patch("scripts.obsidian_writer.requests.get")
    def test_partial_failure_doesnt_block_others(self, mock_get, mock_put, writer, sample_date, sample_status):
        # First PUT (briefing) fails, second (entity) succeeds
        import requests as req
        existing_entity = "---\ntype: wiki/entity\n---\n\n# Anthropic\n"
        mock_get.return_value = MagicMock(status_code=200, text=existing_entity, raise_for_status=lambda: None)

        call_count = [0]
        def put_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise req.ConnectionError("first call fails")
            return MagicMock(status_code=200, raise_for_status=lambda: None)

        mock_put.side_effect = put_side_effect

        results = writer.publish(
            markdown_content="# Briefing",
            date=sample_date,
            status=sample_status,
            emerging_themes=[],
            top_papers=[],
            entity_mentions=[{"name": "Anthropic", "count": 3, "example_titles": []}],
            trending_topics={},
            weekly_deep_dive="",
            briefing_name="Personal-Briefing-2026-04-05",
            weekly_briefing_names=[],
        )
        # Briefing failed but entities still attempted
        assert results["briefing"] is False
        assert "Anthropic" in results["entities"]

    @patch("scripts.obsidian_writer.requests.put")
    @patch("scripts.obsidian_writer.requests.get")
    def test_weekly_synthesis_included_when_present(self, mock_get, mock_put, writer, sample_date, sample_status):
        mock_get.return_value = MagicMock(status_code=404)
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)

        results = writer.publish(
            markdown_content="# Briefing",
            date=sample_date,
            status=sample_status,
            emerging_themes=[],
            top_papers=[],
            entity_mentions=[],
            trending_topics={},
            weekly_deep_dive="# Weekly Deep Dive\n\nContent.",
            briefing_name="Personal-Briefing-2026-04-05",
            weekly_briefing_names=["Personal-Briefing-2026-04-01"],
        )
        assert results["weekly_synthesis"] is True

    @patch("scripts.obsidian_writer.requests.put")
    @patch("scripts.obsidian_writer.requests.get")
    def test_concept_promotion_included(self, mock_get, mock_put, writer, sample_date, sample_status):
        mock_get.return_value = MagicMock(status_code=404)
        mock_put.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)

        results = writer.publish(
            markdown_content="# Briefing",
            date=sample_date,
            status=sample_status,
            emerging_themes=["Agent safety is trending"],
            top_papers=[],
            entity_mentions=[],
            trending_topics={"agent-safety": {"first_seen": "2026-04-01", "count": 3, "last_seen": "2026-04-05"}},
            weekly_deep_dive="",
            briefing_name="Personal-Briefing-2026-04-05",
            weekly_briefing_names=[],
        )
        assert "agent-safety" in results["concepts"]
        assert results["concepts"]["agent-safety"] is True

    def test_dry_run_skips_all(self, writer, sample_date, sample_status):
        results = writer.publish(
            markdown_content="# Briefing",
            date=sample_date,
            status=sample_status,
            emerging_themes=[],
            top_papers=[],
            entity_mentions=[],
            trending_topics={},
            weekly_deep_dive="",
            briefing_name="Personal-Briefing-2026-04-05",
            weekly_briefing_names=[],
            dry_run=True,
        )
        assert results == {"dry_run": True}
