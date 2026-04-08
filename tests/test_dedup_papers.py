# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for paper deduplication in briefing_runner."""

import pytest
from scripts.briefing_runner import BriefingCoordinator


@pytest.fixture
def runner():
    config = {
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
        "podcast": {"enabled": False},
        "obsidian": {"enabled": False},
    }
    return BriefingCoordinator(config=config, dry_run=True)


class TestDeduplicateSimilarPapers:
    def test_removes_near_duplicate_titles(self, runner):
        papers = [
            {"title": "A Novel Approach to Agent Evaluation in Multi-Agent Systems", "id": "1"},
            {"title": "A Novel Approach to Agent Evaluation in Multi-Agent System", "id": "2"},
            {"title": "Completely Different Paper on Robotics", "id": "3"},
        ]
        result = runner.deduplicate_similar_papers(papers)
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "3"

    def test_keeps_different_papers(self, runner):
        papers = [
            {"title": "Paper About Transformers", "id": "1"},
            {"title": "Paper About Reinforcement Learning", "id": "2"},
            {"title": "Paper About Computer Vision", "id": "3"},
        ]
        result = runner.deduplicate_similar_papers(papers)
        assert len(result) == 3

    def test_empty_list(self, runner):
        assert runner.deduplicate_similar_papers([]) == []

    def test_single_paper(self, runner):
        papers = [{"title": "Only Paper", "id": "1"}]
        result = runner.deduplicate_similar_papers(papers)
        assert len(result) == 1

    def test_exact_duplicate_titles(self, runner):
        papers = [
            {"title": "Exact Same Title", "id": "1"},
            {"title": "Exact Same Title", "id": "2"},
        ]
        result = runner.deduplicate_similar_papers(papers)
        assert len(result) == 1
        assert result[0]["id"] == "1"
