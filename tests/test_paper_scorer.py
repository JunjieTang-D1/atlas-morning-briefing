# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for paper_scorer module."""

import pytest
from scripts.paper_scorer import PaperScorer


@pytest.fixture
def default_scorer():
    return PaperScorer(
        topics=["Agent Evaluation", "Multi-Agent Systems"],
        weights={"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
        num_picks=2,
    )


@pytest.fixture
def sample_papers():
    return [
        {
            "id": "2401.00001",
            "title": "Evaluating Multi-Agent Systems with Benchmarks",
            "summary": "We propose a benchmark for evaluating multi-agent systems. "
            "Code available at github.com/example/benchmark.",
            "authors": ["Alice", "Bob"],
            "published": "2026-03-05T00:00:00Z",
            "categories": ["cs.AI"],
            "arxiv_url": "http://arxiv.org/abs/2401.00001",
            "pdf_link": "http://arxiv.org/pdf/2401.00001.pdf",
        },
        {
            "id": "2401.00002",
            "title": "Quantum Computing for Drug Discovery",
            "summary": "We demonstrate quantum advantage for molecular simulation.",
            "authors": ["Charlie"],
            "published": "2026-03-04T00:00:00Z",
            "categories": ["quant-ph"],
            "arxiv_url": "http://arxiv.org/abs/2401.00002",
            "pdf_link": "http://arxiv.org/pdf/2401.00002.pdf",
        },
        {
            "id": "2401.00003",
            "title": "Tool Use in Large Language Models",
            "summary": "A simple and efficient approach to tool use evaluation. "
            "Source code is provided.",
            "authors": ["Diana", "Eve"],
            "published": "2026-03-06T00:00:00Z",
            "categories": ["cs.CL"],
            "arxiv_url": "http://arxiv.org/abs/2401.00003",
            "pdf_link": "http://arxiv.org/pdf/2401.00003.pdf",
        },
    ]


class TestHasCodeRepository:
    def test_detects_github_link(self, default_scorer):
        paper = {"title": "Test", "summary": "Code at github.com/user/repo"}
        assert default_scorer.has_code_repository(paper) is True

    def test_detects_gitlab_link(self, default_scorer):
        paper = {"title": "Test", "summary": "Code at gitlab.com/user/repo"}
        assert default_scorer.has_code_repository(paper) is True

    def test_detects_huggingface_link(self, default_scorer):
        paper = {"title": "Test", "summary": "Model at huggingface.co/user"}
        assert default_scorer.has_code_repository(paper) is True

    def test_detects_code_available_text(self, default_scorer):
        paper = {"title": "Test", "summary": "Code available upon request"}
        assert default_scorer.has_code_repository(paper) is True

    def test_detects_source_code_text(self, default_scorer):
        paper = {"title": "Source code is provided", "summary": "Abstract text"}
        assert default_scorer.has_code_repository(paper) is True

    def test_no_code_reference(self, default_scorer):
        paper = {"title": "Test Paper", "summary": "We propose a new method."}
        assert default_scorer.has_code_repository(paper) is False

    def test_empty_fields(self, default_scorer):
        paper = {"title": "", "summary": ""}
        assert default_scorer.has_code_repository(paper) is False


class TestCalculateTopicMatch:
    def test_returns_scores_for_all_papers(self, default_scorer, sample_papers):
        scores = default_scorer.calculate_topic_match(sample_papers)
        assert len(scores) == len(sample_papers)

    def test_relevant_paper_scores_higher(self, default_scorer, sample_papers):
        scores = default_scorer.calculate_topic_match(sample_papers)
        # Paper 0 (agent evaluation) should score higher than Paper 1 (quantum)
        assert scores[0] > scores[1]

    def test_empty_papers_returns_empty(self, default_scorer):
        scores = default_scorer.calculate_topic_match([])
        assert scores == []

    def test_all_scores_between_0_and_1(self, default_scorer, sample_papers):
        scores = default_scorer.calculate_topic_match(sample_papers)
        for score in scores:
            assert 0.0 <= score <= 1.0


class TestCalculateRecencyScore:
    def test_recent_paper_scores_high(self, default_scorer):
        # Use today's date so this test doesn't become stale over time
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        paper = {"published": today}
        score = default_scorer.calculate_recency_score(paper)
        assert score > 0.9

    def test_old_paper_scores_low(self, default_scorer):
        paper = {"published": "2025-01-01T00:00:00Z"}
        score = default_scorer.calculate_recency_score(paper)
        assert score < 0.1

    def test_no_date_returns_zero(self, default_scorer):
        paper = {"published": ""}
        score = default_scorer.calculate_recency_score(paper)
        assert score == 0.0

    def test_invalid_date_returns_zero(self, default_scorer):
        paper = {"published": "not-a-date"}
        score = default_scorer.calculate_recency_score(paper)
        assert score == 0.0


class TestEstimateReproductionDifficulty:
    def test_simple_paper_is_small(self, default_scorer):
        paper = {"summary": "A simple and lightweight approach to text classification"}
        assert default_scorer.estimate_reproduction_difficulty(paper) == "S"

    def test_large_paper_is_large(self, default_scorer):
        paper = {"summary": "We train on a large-scale cluster with 1000 gpu hours"}
        assert default_scorer.estimate_reproduction_difficulty(paper) == "L"

    def test_xl_paper(self, default_scorer):
        paper = {"summary": "Distributed training across a tpu pod with petabyte data"}
        assert default_scorer.estimate_reproduction_difficulty(paper) == "XL"

    def test_default_is_medium(self, default_scorer):
        paper = {"summary": "We propose a novel architecture for image recognition"}
        assert default_scorer.estimate_reproduction_difficulty(paper) == "M"


class TestScorePapers:
    def test_returns_scored_papers(self, default_scorer, sample_papers):
        scored = default_scorer.score_papers(sample_papers)
        assert len(scored) == len(sample_papers)
        for paper in scored:
            assert "score" in paper
            assert "score_breakdown" in paper
            assert "reproduction_difficulty" in paper

    def test_papers_sorted_by_score(self, default_scorer, sample_papers):
        scored = default_scorer.score_papers(sample_papers)
        for i in range(len(scored) - 1):
            assert scored[i]["score"] >= scored[i + 1]["score"]

    def test_code_paper_scores_higher(self, default_scorer, sample_papers):
        scored = default_scorer.score_papers(sample_papers)
        # Papers with code should generally score higher
        code_papers = [p for p in scored if p["score_breakdown"]["has_code"]]
        no_code_papers = [p for p in scored if not p["score_breakdown"]["has_code"]]
        if code_papers and no_code_papers:
            assert code_papers[0]["score"] > no_code_papers[-1]["score"]

    def test_empty_returns_empty(self, default_scorer):
        assert default_scorer.score_papers([]) == []


class TestGetTopPicks:
    def test_returns_correct_number(self, default_scorer, sample_papers):
        picks = default_scorer.get_top_picks(sample_papers)
        assert len(picks) == default_scorer.num_picks

    def test_returns_highest_scored(self, default_scorer, sample_papers):
        picks = default_scorer.get_top_picks(sample_papers)
        all_scored = default_scorer.score_papers(sample_papers)
        for i, pick in enumerate(picks):
            assert pick["score"] == all_scored[i]["score"]
