#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Paper scorer.

Scores papers for reproduction value based on multiple criteria.
"""

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.utils import load_config
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PaperScorer:
    """Scores papers for reproduction value."""

    # Common code hosting patterns
    CODE_PATTERNS = [
        r"github\.com/[\w-]+/[\w-]+",
        r"gitlab\.com/[\w-]+/[\w-]+",
        r"huggingface\.co/[\w-]+",
        r"code available",
        r"source code",
    ]

    def __init__(
        self,
        topics: List[str],
        weights: Dict[str, float],
        num_picks: int = 3,
    ):
        """
        Initialize PaperScorer.

        Args:
            topics: List of topics to match against
            weights: Scoring weights for different criteria
            num_picks: Number of top papers to return
        """
        self.topics = topics
        self.weights = weights
        self.num_picks = num_picks
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def has_code_repository(self, paper: Dict[str, Any]) -> bool:
        """
        Check if paper has a linked code repository.

        Args:
            paper: Paper dictionary

        Returns:
            True if code repository is found
        """
        text_fields = [
            paper.get("summary", ""),
            paper.get("title", ""),
        ]
        combined_text = " ".join(text_fields).lower()

        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True

        return False

    def calculate_topic_match(self, papers: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate topic match scores for all papers using TF-IDF.

        Args:
            papers: List of paper dictionaries

        Returns:
            List of topic match scores (0-1)
        """
        if not papers:
            return []

        # Combine title and summary for each paper
        paper_texts = [
            f"{p.get('title', '')} {p.get('summary', '')}"
            for p in papers
        ]

        # Add topic strings
        topic_text = " ".join(self.topics)

        try:
            # Fit vectorizer on all texts including topic
            all_texts = paper_texts + [topic_text]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Calculate cosine similarity between each paper and topics
            topic_vector = tfidf_matrix[-1]
            paper_vectors = tfidf_matrix[:-1]

            similarities = cosine_similarity(paper_vectors, topic_vector.reshape(1, -1))
            scores = similarities.flatten().tolist()

            return scores

        except Exception as e:
            logger.warning(f"Failed to calculate topic match: {e}")
            return [0.0] * len(papers)

    def calculate_recency_score(self, paper: Dict[str, Any]) -> float:
        """
        Calculate recency score based on publication date.

        Args:
            paper: Paper dictionary

        Returns:
            Recency score (higher = more recent)
        """
        published = paper.get("published", "")
        if not published:
            return 0.0

        try:
            pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - pub_date).days

            # Exponential decay: score = e^(-days/30)
            # Papers from today get 1.0, papers from 30 days ago get ~0.37
            score = math.exp(-days_ago / 30.0)
            return score

        except (ValueError, AttributeError) as e:
            logger.debug(f"Failed to parse date '{published}': {e}")
            return 0.0

    # Infrastructure-heavy keywords that signal IMPOSSIBLE to reproduce
    # (multi-node clusters, TPU pods, datacenter-scale only)
    INFRA_IMPOSSIBLE_PATTERNS = [
        r"tpu pod", r"petabyte", r"exascale", r"data center",
        r"64.*gpu", r"128.*gpu", r"256.*gpu",
    ]

    # Keywords indicating theoretical/analysis-only papers (no reproducible system)
    THEORY_PATTERNS = [
        r"we prove", r"we formaliz", r"theoretical analysis",
        r"position paper", r"survey", r"we argue",
    ]

    def estimate_reproduction_difficulty(self, paper: Dict[str, Any]) -> str:
        """
        Estimate reproduction difficulty (S/M/L/XL).

        Args:
            paper: Paper dictionary

        Returns:
            Difficulty level: S, M, L, or XL
        """
        summary = paper.get("summary", "").lower()

        # Heuristics for difficulty
        complexity_indicators = {
            "xl": ["petabyte", "exascale", "distributed training", "tpu pod",
                    "gpu cluster", "kubernetes", "multi-node"],
            "l": ["billion", "large-scale", "cluster", "gpu hours", "8 gpu",
                   "16 gpu", "a100", "h100"],
            "m": ["dataset", "training", "fine-tuning", "benchmark"],
            "s": ["simple", "lightweight", "efficient", "small", "api",
                   "retrieval", "rag"],
        }

        # Check for indicators in order of difficulty
        for level in ["xl", "l", "m", "s"]:
            for indicator in complexity_indicators[level]:
                if indicator in summary:
                    return level.upper()

        # Default to M if no clear indicators
        return "M"

    def calculate_infra_penalty(self, paper: Dict[str, Any]) -> float:
        """
        Calculate infrastructure penalty for papers that are truly impossible.

        Only penalizes datacenter-scale (64+ GPU, TPU pod, petabyte) or
        pure theory papers. Single GPU / small cluster (g5.xlarge, trn1)
        is fine — we have EC2 GPU runner access.

        Args:
            paper: Paper dictionary

        Returns:
            Penalty score (0 = no penalty, -5 = max penalty)
        """
        summary = paper.get("summary", "").lower()
        title = paper.get("title", "").lower()
        combined = f"{title} {summary}"

        penalty = 0.0

        # Only penalize truly impossible infra (datacenter-scale)
        for pattern in self.INFRA_IMPOSSIBLE_PATTERNS:
            if re.search(pattern, combined):
                penalty -= 2.0
                break

        # Check theory-only patterns (no system to reproduce)
        for pattern in self.THEORY_PATTERNS:
            if re.search(pattern, combined):
                penalty -= 1.5
                break

        # No code = penalty (but smaller, since some papers are still worth it)
        if not self.has_code_repository(paper):
            penalty -= 1.0

        return max(-5.0, penalty)

    def score_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score all papers and return sorted by score.

        Args:
            papers: List of paper dictionaries

        Returns:
            List of papers with scores, sorted by total score
        """
        if not papers:
            return []

        # Calculate topic match scores for all papers
        topic_scores = self.calculate_topic_match(papers)

        scored_papers = []

        for i, paper in enumerate(papers):
            # Calculate individual scores
            has_code = self.has_code_repository(paper)
            topic_match = topic_scores[i] if i < len(topic_scores) else 0.0
            recency = self.calculate_recency_score(paper)
            infra_penalty = self.calculate_infra_penalty(paper)

            # Calculate weighted total score
            total_score = (
                (self.weights.get("has_code", 5) * (1.0 if has_code else 0.0))
                + (self.weights.get("topic_match", 3) * topic_match)
                + (self.weights.get("recency", 2) * recency)
                + infra_penalty  # Penalize infra-heavy or theory-only papers
            )

            # Estimate reproduction difficulty
            difficulty = self.estimate_reproduction_difficulty(paper)

            # Add scoring info to paper
            scored_paper = {
                **paper,
                "score": round(total_score, 2),
                "score_breakdown": {
                    "has_code": has_code,
                    "topic_match": round(topic_match, 3),
                    "recency": round(recency, 3),
                    "infra_penalty": round(infra_penalty, 1),
                },
                "reproduction_difficulty": difficulty,
            }

            scored_papers.append(scored_paper)

        # Sort by score (descending)
        scored_papers.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Scored {len(scored_papers)} papers")
        return scored_papers

    def get_top_picks(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get top N paper picks for reproduction.

        Args:
            papers: List of scored paper dictionaries

        Returns:
            Top N papers
        """
        scored = self.score_papers(papers)
        top_picks = scored[: self.num_picks]

        logger.info(f"Selected top {len(top_picks)} papers for reproduction")
        return top_picks



def main() -> int:
    """
    Main entry point for paper_scorer.

    Returns:
        Exit code (0 for success, 1 for partial failure, 2 for total failure)
    """
    parser = argparse.ArgumentParser(description="Score papers for reproduction value")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with papers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scored_papers.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(getattr(logging, args.log_level))

    # Load config
    config = load_config(args.config)

    # Extract settings
    topics = config.get("arxiv_topics", [])
    weights = config.get("paper_scoring", {})
    num_picks = config.get("num_paper_picks", 3)

    if not topics:
        logger.warning("No arxiv_topics configured, using default weights only")

    # Load papers
    try:
        with open(args.input, "r") as f:
            papers = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        return 2
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse input JSON: {e}")
        return 2

    if not papers:
        logger.warning("No papers to score")
        return 1

    # Score papers
    scorer = PaperScorer(topics=topics, weights=weights, num_picks=num_picks)
    scored_papers = scorer.score_papers(papers)

    # Save results
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(scored_papers, f, indent=2)
        logger.info(f"Saved {len(scored_papers)} scored papers to {args.output}")
        return 0
    except IOError as e:
        logger.error(f"Failed to write output file: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
