#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Intelligence layer for morning briefing.

Uses Amazon Bedrock models to add reasoning, synthesis, and summarization
to the briefing pipeline. Falls back gracefully when Amazon Bedrock is unavailable.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from scripts.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an AI research analyst generating a daily morning briefing. "
    "Be concise, insightful, and factual. Use markdown formatting. "
    "Do not invent facts or citations. If information is insufficient, say so."
)


def _sanitize_prompt_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize external input before embedding in LLM prompts.

    Strips prompt injection markers and truncates to prevent abuse.

    Args:
        text: Raw input text from external source.
        max_length: Maximum allowed character length.

    Returns:
        Sanitized text safe for prompt inclusion.
    """
    if not isinstance(text, str):
        return ""
    text = text[:max_length]
    # Strip characters that could be used for prompt injection / XML tag spoofing
    text = re.sub(r"</?(?:system|human|assistant|instructions?|prompt)[^>]*>", "", text, flags=re.IGNORECASE)
    return text


class BriefingIntelligence:
    """Adds LLM-powered intelligence to the briefing pipeline."""

    def __init__(self, llm: LLMClient, config: Dict[str, Any]):
        """
        Initialize BriefingIntelligence.

        Args:
            llm: LLMClient instance.
            config: Full config dictionary.
        """
        self.bedrock = llm
        self.config = config
        self.topics = config.get("arxiv_topics", [])

    @property
    def available(self) -> bool:
        """Check if intelligence features are available."""
        return self.bedrock.available

    @staticmethod
    def extract_score(text: str) -> Tuple[int, str]:
        """Extract SCORE:X/5 from text, return (score_int, cleaned_text)."""
        text = text.strip()
        match = re.match(r"SCORE:\s*(\d)/5\s*(.*)", text, re.DOTALL)
        if match:
            return int(match.group(1)), match.group(2).strip()
        # Also try "Score: X/5" variant
        match = re.match(r"[Ss]core:\s*(\d)/5\s*(.*)", text, re.DOTALL)
        if match:
            return int(match.group(1)), match.group(2).strip()
        return None, text

    @staticmethod
    def _parse_ranked_response(text: str) -> List[Tuple[int, str]]:
        """
        Parse LLM response with [number] prefixed items.

        Handles bold markers (**[1]**), numbered sub-items, and multi-line entries.

        Args:
            text: Raw LLM response text.

        Returns:
            List of (0-based index, text) tuples.
        """
        items = []
        current_idx = -1
        current_lines: List[str] = []

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Strip markdown bold markers for detection
            clean = line.lstrip("*").strip()

            # Check if line starts a new item: [number] or **[number]**
            if clean.startswith("[") and "]" in clean[:8]:
                # Save previous item
                if current_idx >= 0 and current_lines:
                    full_text = " ".join(line for line in current_lines if line)
                    if full_text.strip():
                        items.append((current_idx, full_text.strip()))

                try:
                    bracket_start = clean.index("[")
                    bracket_end = clean.index("]")
                    current_idx = int(clean[bracket_start + 1:bracket_end]) - 1
                    # Get text after the "]" and any trailing ** or title
                    rest = clean[bracket_end + 1:].strip().rstrip("*").strip()
                    current_lines = [rest] if rest else []
                except (ValueError, IndexError):
                    current_idx = -1
                    current_lines = []
            else:
                # Strip numbered sub-items like "1." "2." that are part of summary
                current_lines.append(
                    line.lstrip("0123456789.").strip()
                    if re.match(r"^\d+\.", line)
                    else line
                )

        # Save last item
        if current_idx >= 0 and current_lines:
            full_text = " ".join(line for line in current_lines if line)
            if full_text.strip():
                items.append((current_idx, full_text.strip()))

        return items

    def filter_papers_by_relevance(
        self, papers: List[Dict[str, Any]], interest_profile: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Two-stage paper filtering using Opus.

        Stage 1: Send ALL paper titles+abstracts to Opus in ONE call.
                Score each paper 1-10 for relevance to interest profile.
                Return ONLY papers scoring >= 7.
        Stage 2: Filtered papers go to existing score_papers_semantically() for deep ranking.

        Args:
            papers: List of paper dictionaries with 'title' and 'summary' keys.
            interest_profile: Optional list of interest topics with weights.
                            If None, falls back to arxiv_topics.

        Returns:
            Filtered papers (target: ~30) that are highly relevant.
        """
        if not self.available or not papers:
            return papers

        # Get interest profile from config or fall back to topics
        if interest_profile is None:
            interest_profile = self.config.get("interest_profile", [])

        # If no interest profile, fall back to current behavior
        if not interest_profile:
            logger.info("No interest_profile configured, skipping relevance filtering")
            return papers

        # Build interest profile string
        profile_lines = []
        for item in interest_profile:
            topic = item.get("topic", "")
            weight = item.get("weight", 1.0)
            profile_lines.append(f"- {topic} (weight: {weight})")
        profile_str = "\n".join(profile_lines)

        # Build paper batch (all papers) with sanitized inputs
        paper_lines = []
        for i, p in enumerate(papers):
            title = _sanitize_prompt_input(p.get("title", "Untitled"), max_length=500)
            abstract = _sanitize_prompt_input(p.get("summary", "")[:400], max_length=500)
            paper_lines.append(f"[{i+1}] {title}\n{abstract}")

        papers_block = "\n\n".join(paper_lines)

        prompt = (
            "You are filtering papers for a daily AI research briefing. "
            "Score each paper 1-10 for relevance to this interest profile:\n\n"
            f"<interest_profile>\n{profile_str}\n</interest_profile>\n\n"
            f"<papers>\n{papers_block}\n</papers>\n\n"
            "Return ONLY papers scoring >= 7. For each relevant paper, respond with:\n"
            "[number] score reason\n"
            "Example: [5] 9 Directly addresses multi-agent systems with novel evaluation methodology\n\n"
            "Be selective. Only include papers that strongly match the profile."
        )

        result = self.bedrock.invoke(
            prompt, tier="heavy", max_tokens=2000, system_prompt=SYSTEM_PROMPT,
            name="filter_papers",
        )
        if not result:
            logger.warning("Stage 1 filtering failed, returning all papers")
            return papers

        # Parse filtered results
        filtered_papers = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            try:
                bracket_end = line.index("]")
                idx = int(line[1:bracket_end]) - 1
                rest = line[bracket_end + 1:].strip()
                parts = rest.split(" ", 1)
                score = float(parts[0])
                reason = parts[1] if len(parts) > 1 else ""

                if 0 <= idx < len(papers) and score >= 7:
                    paper = papers[idx].copy()
                    paper["relevance_score"] = score
                    paper["relevance_reason"] = reason
                    filtered_papers.append(paper)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse line: {line}, error: {e}")
                continue

        logger.info(f"Stage 1 filtering: {len(papers)} → {len(filtered_papers)} papers (score >= 7)")
        return filtered_papers if filtered_papers else papers[:30]  # Fallback to top 30 if filtering fails

    def generate_dynamic_queries(
        self, previous_briefing_state: Optional[Dict[str, Any]], static_queries: List[str]
    ) -> List[str]:
        """
        Generate dynamic news queries based on yesterday's top stories.

        Takes yesterday's top stories from state.json + static queries,
        then uses Opus to generate 3 additional targeted follow-up queries.

        Args:
            previous_briefing_state: Previous briefing state with top stories.
            static_queries: Static queries from config.

        Returns:
            Combined list of static + dynamic queries (total ~13 queries).
        """
        if not self.available or not previous_briefing_state or not static_queries:
            return static_queries

        # Extract yesterday's top stories
        prev_date = previous_briefing_state.get("date", "unknown")
        prev_paper_titles = previous_briefing_state.get("top_paper_titles", [])[:5]
        prev_blog_titles = previous_briefing_state.get("top_blog_titles", [])[:5]
        prev_news_titles = previous_briefing_state.get("top_news_titles", [])[:5]
        prev_themes = previous_briefing_state.get("emerging_themes", [])

        # Build context from yesterday's briefing
        context_parts = []
        if prev_paper_titles:
            context_parts.append(f"Top Papers ({prev_date}):\n" + "\n".join(f"- {t}" for t in prev_paper_titles))
        if prev_blog_titles:
            context_parts.append(f"Top Blogs ({prev_date}):\n" + "\n".join(f"- {t}" for t in prev_blog_titles))
        if prev_news_titles:
            context_parts.append(f"Top News ({prev_date}):\n" + "\n".join(f"- {t}" for t in prev_news_titles))
        if prev_themes:
            context_parts.append(f"Emerging Themes: {', '.join(prev_themes)}")

        if not context_parts:
            logger.info("No previous briefing context, using static queries only")
            return static_queries

        context_str = "\n\n".join(context_parts)
        static_queries_str = "\n".join(f"- {q}" for q in static_queries)

        prompt = (
            "You are generating follow-up news queries based on yesterday's AI research briefing.\n\n"
            f"<yesterday_briefing>\n{context_str}\n</yesterday_briefing>\n\n"
            f"<static_queries>\n{static_queries_str}\n</static_queries>\n\n"
            "Generate 3 targeted follow-up queries to track developments in yesterday's hot topics. "
            "Return ONLY the new queries, one per line, no numbering or bullets. "
            "Make them specific and actionable for news search.\n\n"
            "Example outputs:\n"
            "- Claude 3.5 Sonnet benchmark results\n"
            "- AWS Trainium chip adoption enterprise\n"
            "- Multi-agent orchestration frameworks release"
        )

        result = self.bedrock.invoke(prompt, tier="light", max_tokens=300, name="dynamic_queries")
        if not result:
            logger.info("Dynamic query generation failed, using static queries only")
            return static_queries

        # Parse new queries
        new_queries = [
            line.strip().strip("- *")
            for line in result.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Deduplicate and limit
        existing_lower = {q.lower() for q in static_queries}
        dynamic_queries = [
            q for q in new_queries
            if q.lower() not in existing_lower and len(q) > 10
        ][:3]

        if dynamic_queries:
            logger.info(f"Generated {len(dynamic_queries)} dynamic queries: {dynamic_queries}")

        return static_queries + dynamic_queries

    def expand_topics(self, topics: List[str]) -> List[str]:
        """
        Expand user-configured topics with semantically related queries.

        Uses the light tier model for simple brainstorming.

        Args:
            topics: Original topic list from config.

        Returns:
            Expanded topic list (original + new suggestions).
        """
        if not self.available or not topics:
            return topics

        topic_list = "\n".join(f"- {t}" for t in topics)
        prompt = (
            "Given these research topics, suggest 2-3 additional related search "
            "queries that would find relevant papers on arxiv. Return ONLY the "
            "new queries, one per line, no numbering or bullets.\n\n"
            f"<topics>\n{topic_list}\n</topics>"
        )

        result = self.bedrock.invoke(prompt, tier="light", max_tokens=256, name="expand_topics")
        if not result:
            return topics

        new_topics = [
            line.strip().strip("- ")
            for line in result.strip().split("\n")
            if line.strip() and line.strip() != "-"
        ]

        # Deduplicate and limit additions
        existing_lower = {t.lower() for t in topics}
        additions = [
            t for t in new_topics
            if t.lower() not in existing_lower and len(t) > 3
        ][:3]

        if additions:
            logger.info(f"Expanded topics with {len(additions)} suggestions: {additions}")

        return topics + additions

    def summarize_papers(
        self, papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate 1-2 sentence summaries for papers.

        Uses the medium tier model for factual summarization.
        Processes in a single batched call to minimize cost.

        Args:
            papers: List of paper dictionaries with 'title' and 'summary' keys.

        Returns:
            Papers with added 'brief_summary' key.
        """
        if not self.available or not papers:
            return papers

        # Batch papers into a single prompt (limit to top 10 for cost)
        batch = papers[:10]
        paper_texts = []
        for i, p in enumerate(batch):
            title = _sanitize_prompt_input(p.get("title", "Untitled"), max_length=500)
            abstract = _sanitize_prompt_input(p.get("summary", "")[:500], max_length=600)
            paper_texts.append(f"[{i+1}] {title}\n{abstract}")

        papers_block = "\n\n".join(paper_texts)
        prompt = (
            "For each paper below, write a 1-2 sentence summary that captures "
            "the key contribution. Return as a numbered list matching the input "
            "numbering. Be factual -- do not add information not in the abstract.\n\n"
            f"<papers>\n{papers_block}\n</papers>"
        )

        result = self.bedrock.invoke(
            prompt, tier="medium", max_tokens=1500, system_prompt=SYSTEM_PROMPT,
            name="summarize_papers",
        )
        if not result:
            return papers

        # Parse numbered summaries back to papers
        summaries = _parse_numbered_list(result, len(batch))
        for i, summary in enumerate(summaries):
            if i < len(papers):
                papers[i]["brief_summary"] = summary

        logger.info(f"Generated summaries for {len(summaries)} papers")
        return papers

    def score_papers_semantically(
        self, papers: List[Dict[str, Any]], topics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Score papers using semantic understanding instead of TF-IDF.

        Uses the medium tier model to assess relevance.

        Args:
            papers: List of paper dictionaries.
            topics: User's research topics.

        Returns:
            Papers with added 'semantic_score' (0-10) and 'relevance_reason' keys.
        """
        if not self.available or not papers or not topics:
            return papers

        batch = papers[:15]
        paper_lines = []
        for i, p in enumerate(batch):
            title = _sanitize_prompt_input(p.get("title", "Untitled"), max_length=500)
            abstract = _sanitize_prompt_input(p.get("summary", "")[:300], max_length=400)
            paper_lines.append(f"[{i+1}] {title}: {abstract}")

        papers_block = "\n".join(paper_lines)
        prompt = (
            "Rate each paper's relevance to these research interests on a 0-10 scale.\n\n"
            f"<interests>{', '.join(topics)}</interests>\n\n"
            f"<papers>\n{papers_block}\n</papers>\n\n"
            "For each paper, respond with ONLY this format, one per line:\n"
            "[number] score reason\n"
            "Example: [1] 8 Directly addresses agent evaluation methodology"
        )

        result = self.bedrock.invoke(
            prompt, tier="medium", max_tokens=1000, system_prompt=SYSTEM_PROMPT,
            name="score_papers",
        )
        if not result:
            return papers

        # Parse scores
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            try:
                bracket_end = line.index("]")
                idx = int(line[1:bracket_end]) - 1
                rest = line[bracket_end + 1:].strip()
                parts = rest.split(" ", 1)
                score = float(parts[0])
                reason = parts[1] if len(parts) > 1 else ""
                if 0 <= idx < len(papers):
                    papers[idx]["semantic_score"] = min(10.0, max(0.0, score))
                    papers[idx]["relevance_reason"] = reason
            except (ValueError, IndexError):
                continue

        logger.info("Semantic scoring complete")
        return papers

    def assess_reproduction_feasibility(
        self, papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Assess reproduction feasibility and re-rank papers by actionability.

        Uses structured scoring across 5 dimensions to filter out papers
        that are not practically reproducible on our setup (single EC2,
        Amazon Bedrock, no GPU cluster).

        Scoring dimensions (each 1-5):
          1. code_available — Is code open-source and runnable?
          2. data_accessible — Is data open/downloadable (<10GB)?
          3. infra_fit — Can run on single EC2 + Bedrock (no GPU cluster)?
          4. bedrock_ready — Can use Bedrock models (Claude/Titan) directly?
          5. effort — Time to reproduce (5=weekend, 1=months)

        Papers scoring < 15/25 are demoted (moved below higher-scoring ones).
        Papers scoring < 10/25 are dropped from top picks entirely.

        Args:
            papers: Top-scored papers (typically 3-10).

        Returns:
            Papers re-ranked by reproduction feasibility, with structured scores.
        """
        if not self.available or not papers:
            return papers

        paper_texts = []
        for i, p in enumerate(papers):
            title = _sanitize_prompt_input(p.get("title", "Untitled"), max_length=500)
            abstract = _sanitize_prompt_input(p.get("summary", "")[:400], max_length=500)
            has_code = p.get("score_breakdown", {}).get("has_code", False)
            paper_texts.append(
                f"[{i+1}] {title}\nCode available: {has_code}\n{abstract}"
            )

        papers_block = "\n\n".join(paper_texts)
        prompt = (
            "You are evaluating papers for PRACTICAL reproduction on this setup:\n"
            "- Single EC2 GPU instance available (g5.xlarge = 1x A10G 24GB, or trn1.2xlarge = AWS Trainium)\n"
            "- Amazon Bedrock API (Claude Sonnet/Opus, Titan Embeddings)\n"
            "- Python + standard ML libraries, Kubernetes OK if single-node\n"
            "- Budget: <$50 per paper, <1 week effort\n\n"
            "Score each paper on 5 dimensions (1-5 each, 25 max):\n"
            "1. code_available: 5=open repo+README, 3=partial code, 1=no code\n"
            "2. data_accessible: 5=open data <50GB, 3=needs request/large, 1=proprietary\n"
            "3. infra_fit: 5=CPU/API only, 4=single GPU(A10G/Trainium), 3=multi-GPU single node, 2=multi-node cluster, 1=datacenter/TPU pod\n"
            "4. bedrock_ready: 5=can swap in Bedrock models directly, 3=needs adapter, 1=incompatible\n"
            "5. effort: 5=weekend(S), 4=1week(M), 3=2weeks(L), 2=month(XL), 1=impossible\n\n"
            "For each paper respond in this EXACT format (one line each):\n"
            "[number] code:X data:X infra:X bedrock:X effort:X | verdict\n\n"
            "Example: [1] code:5 data:4 infra:5 bedrock:5 effort:4 | Open benchmark + Bedrock RAG, easy to reproduce\n"
            "Example: [2] code:1 data:1 infra:1 bedrock:2 effort:1 | No code, needs GPU cluster, skip\n\n"
            f"<papers>\n{papers_block}\n</papers>"
        )

        result = self.bedrock.invoke(
            prompt, tier="medium", max_tokens=1000, system_prompt=SYSTEM_PROMPT,
            name="assess_repro",
        )
        if not result:
            return papers

        # Parse structured scores
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            try:
                bracket_end = line.index("]")
                idx = int(line[1:bracket_end]) - 1
                rest = line[bracket_end + 1:].strip()

                # Parse scores: code:X data:X infra:X bedrock:X effort:X | verdict
                scores = {}
                verdict = ""
                if "|" in rest:
                    scores_part, verdict = rest.split("|", 1)
                    verdict = verdict.strip()
                else:
                    scores_part = rest

                for dim in ["code", "data", "infra", "bedrock", "effort"]:
                    match = re.search(rf"{dim}:(\d)", scores_part)
                    if match:
                        scores[dim] = int(match.group(1))

                if 0 <= idx < len(papers) and scores:
                    total = sum(scores.values())
                    papers[idx]["repro_scores"] = scores
                    papers[idx]["repro_total"] = total
                    papers[idx]["repro_verdict"] = verdict
                    papers[idx]["reproduction_assessment"] = (
                        f"Score: {total}/25 "
                        f"(code:{scores.get('code',0)} data:{scores.get('data',0)} "
                        f"infra:{scores.get('infra',0)} bedrock:{scores.get('bedrock',0)} "
                        f"effort:{scores.get('effort',0)}) — {verdict}"
                    )
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse repro line: {line}, error: {e}")
                continue

        # Re-rank: sort by repro_total descending, drop papers below threshold
        min_score = self.config.get("repro_min_score", 12)
        scored = [p for p in papers if p.get("repro_total", 0) >= min_score]
        unscored = [p for p in papers if "repro_total" not in p]
        scored.sort(key=lambda x: x.get("repro_total", 0), reverse=True)

        dropped = len(papers) - len(scored) - len(unscored)
        if dropped:
            logger.info(f"Repro gate: dropped {dropped} papers scoring <{min_score}/25")

        result_papers = scored + unscored
        logger.info(
            f"Reproduction feasibility: {len(papers)} assessed, "
            f"{len(scored)} passed gate (≥{min_score}/25), top score: "
            f"{scored[0].get('repro_total', 0) if scored else 'N/A'}/25"
        )
        return result_papers

    def rank_and_summarize_news(
        self, news: List[Dict[str, Any]], topics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rank news by relevance and generate 2-3 sentence summaries for top items.

        Uses the medium tier model.

        Args:
            news: List of news article dictionaries.
            topics: User's research topics for relevance ranking.

        Returns:
            Top 5 news articles, ranked and summarized.
        """
        if not self.available or not news:
            logger.info(f"News ranking skipped: available={self.available}, news_count={len(news)}")
            return news[:5]

        news_lines = []
        for i, article in enumerate(news[:20]):
            title = _sanitize_prompt_input(article.get("title", ""), max_length=300)
            source = _sanitize_prompt_input(article.get("source", ""), max_length=100)
            snippet = _sanitize_prompt_input(
                article.get("description", article.get("snippet", ""))[:200], max_length=300
            )
            news_lines.append(f"[{i+1}] {title} ({source}): {snippet}")

        articles_block = "\n".join(news_lines)
        prompt = (
            "You are curating a daily AI/tech briefing. From these news articles, "
            "select the TOP 5 most important for an AI researcher/engineer.\n\n"
            f"<interests>{', '.join(topics[:5])}</interests>\n\n"
            f"<articles>\n{articles_block}\n</articles>\n\n"
            "For each of your top 5 picks, respond in this exact format:\n"
            "[original_number] 2-3 sentence summary explaining why this matters.\n\n"
            "Rank by importance. Be factual. Do not invent details."
        )

        result = self.bedrock.invoke(
            prompt, tier="medium", max_tokens=1000, system_prompt=SYSTEM_PROMPT,
            name="rank_news",
        )
        if not result:
            return news[:5]

        logger.debug(f"News LLM response:\n{result[:500]}")

        # Parse ranked results using shared parser
        parsed = self._parse_ranked_response(result)
        logger.info(f"News parsing: {len(parsed)} items parsed from LLM response")
        ranked_news = []
        for idx, text in parsed:
            if 0 <= idx < len(news):
                article = news[idx].copy()
                article["brief_summary"] = text
                ranked_news.append(article)

        if ranked_news:
            diversified = self._enforce_source_diversity(ranked_news, max_per_source=2)
            logger.info(f"Ranked and summarized {len(diversified)} news articles")
            return diversified[:5]

        # Retry once with simpler prompt
        logger.warning(f"News ranking parse failed (attempt 1). LLM response: {result[:300]}")
        retry_result = self.bedrock.invoke(
            f"From these articles, pick the 5 most important for an AI researcher. "
            f"Format EXACTLY as: [number] summary sentence.\n\n{articles_block}",
            tier="medium", max_tokens=800, system_prompt=SYSTEM_PROMPT,
            name="rank_news_retry",
        )
        if retry_result:
            parsed_retry = self._parse_ranked_response(retry_result)
            for idx, text in parsed_retry:
                if 0 <= idx < len(news):
                    article = news[idx].copy()
                    article["brief_summary"] = text
                    ranked_news.append(article)
            if ranked_news:
                diversified = self._enforce_source_diversity(ranked_news, max_per_source=2)
                logger.info(f"Ranked {len(diversified)} news on retry")
                return diversified[:5]

        logger.warning("News ranking failed after retry, using description fallback")
        fallback = []
        for article in news[:5]:
            a = article.copy()
            a["brief_summary"] = a.get("description", a.get("snippet", ""))
            fallback.append(a)
        return fallback

    def rank_and_summarize_blogs(
        self, blogs: List[Dict[str, Any]], topics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rank blogs by relevance and generate 1-2 sentence summaries for top items.

        Uses the light tier model to save cost.

        Args:
            blogs: List of blog article dictionaries.
            topics: User's research topics for relevance ranking.

        Returns:
            Top 5 blog articles, ranked and summarized.
        """
        if not self.available or not blogs:
            return blogs[:5]

        blog_lines = []
        for i, article in enumerate(blogs[:15]):
            title = _sanitize_prompt_input(article.get("title", ""), max_length=300)
            source = _sanitize_prompt_input(article.get("source", ""), max_length=100)
            summary = _sanitize_prompt_input(article.get("summary", "")[:200], max_length=300)
            blog_lines.append(f"[{i+1}] {title} ({source}): {summary}")

        blogs_block = "\n".join(blog_lines)
        prompt = (
            "You are curating a daily AI/tech briefing. From these blog posts, "
            "select the TOP 5 most relevant for an AI researcher/engineer.\n\n"
            f"<interests>{', '.join(topics[:5])}</interests>\n\n"
            f"<blogs>\n{blogs_block}\n</blogs>\n\n"
            "For each of your top 5 picks, respond in this exact format:\n"
            "[original_number] SCORE:X/5 1-2 sentence summary of what the post covers.\n\n"
            "SCORE is a combined rating (1-5) of impact, complexity, and innovation. "
            "5 = groundbreaking, 1 = routine.\n"
            "Rank by relevance. Be concise."
        )

        result = self.bedrock.invoke(
            prompt, tier="light", max_tokens=800, system_prompt=SYSTEM_PROMPT,
            name="rank_blogs",
        )
        if not result:
            return blogs[:5]

        # Parse ranked results using shared parser
        parsed = self._parse_ranked_response(result)
        ranked_blogs = []
        for idx, text in parsed:
            if 0 <= idx < len(blogs):
                article = blogs[idx].copy()
                score, summary = self.extract_score(text)
                article["brief_summary"] = summary
                if score:
                    article["score_combined"] = score
                if article["brief_summary"]:
                    ranked_blogs.append(article)

        if ranked_blogs:
            # Enforce source diversity: max 2 per source
            diversified = self._enforce_source_diversity(ranked_blogs, max_per_source=2)
            logger.info(f"Ranked and summarized {len(diversified)} blog articles")
            return diversified[:5]

        return self._enforce_source_diversity(blogs, max_per_source=2)[:5]

    def rank_source_links(
        self, items: List[Dict[str, Any]], topics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Score mixed items (newsletter-extracted links + GitHub repos) for relevance.

        Uses the same LLM scoring pattern as rank_and_summarize_blogs().

        Args:
            items: List of dicts with url/link, title, description/summary, source.
            topics: Interest profile topics (strings or dicts with 'topic' key).

        Returns:
            Items enriched with score_combined + brief_summary, sorted desc.
            Unscored items omitted. Returns [] on LLM failure.
        """
        if not self.available or not items:
            return items[:10] if items else []

        lines = []
        for i, item in enumerate(items[:30], 1):
            title = _sanitize_prompt_input(
                item.get("title") or item.get("name") or "", max_length=150
            )
            desc = _sanitize_prompt_input(
                item.get("description") or item.get("summary") or "", max_length=250
            )
            source = _sanitize_prompt_input(item.get("source") or "", max_length=60)
            url = item.get("url") or item.get("link") or ""
            lines.append(f"[{i}] {title} ({source}): {desc}\n    URL: {url}")

        items_block = "\n".join(lines)
        topics_str = ", ".join(
            t if isinstance(t, str) else t.get("topic", "")
            for t in topics[:6]
        )

        prompt = (
            "Rate these tech/AI resources for relevance to a daily AI "
            "practitioner briefing.\n\n"
            f"<interests>{topics_str}</interests>\n"
            f"<items>\n{items_block}\n</items>\n\n"
            "For each of your top picks, respond in this exact format:\n"
            "[original_number] SCORE:X/5 1-2 sentence summary of why this "
            "is relevant.\n\n"
            "SCORE: 5=core AI/ML content, 3=tangentially relevant, "
            "1=off-topic.\n"
            "Skip tracking URLs, login pages, or noise. "
            "Respond ONLY with the numbered scores, no preamble."
        )

        result = self.bedrock.invoke(
            prompt, tier="light", max_tokens=1200,
            system_prompt=SYSTEM_PROMPT, name="rank_source_links",
        )
        if not result:
            return []

        scored = []
        for idx, text in self._parse_ranked_response(result):
            if 0 <= idx < len(items):
                score, summary = self.extract_score(text)
                if score is not None:
                    item = items[idx].copy()
                    item["score_combined"] = score
                    item["brief_summary"] = summary
                    scored.append(item)

        diversified = self._enforce_source_diversity(
            sorted(scored, key=lambda x: x.get("score_combined", 0), reverse=True),
            max_per_source=5,
        )
        logger.info(f"Ranked {len(diversified)} source links")
        return diversified[:10]

    @staticmethod
    def _enforce_source_diversity(
        items: List[Dict[str, Any]], max_per_source: int = 2
    ) -> List[Dict[str, Any]]:
        """Cap items per source to ensure diversity. Overflow items are dropped."""
        source_count: Dict[str, int] = {}
        result = []
        for item in items:
            source = item.get("source", "unknown")
            count = source_count.get(source, 0)
            if count < max_per_source:
                result.append(item)
                source_count[source] = count + 1
        return result

    def correlate_stocks_and_news(
        self,
        stocks: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Correlate stock movements with news headlines.

        Uses the heavy tier model for cross-domain reasoning.

        Args:
            stocks: Stock data with price changes.
            news: News articles.

        Returns:
            Stocks with added 'news_correlation' key.
        """
        if not self.available or not stocks or not news:
            return stocks

        stock_lines = []
        for s in stocks:
            if "error" in s:
                continue
            symbol = s.get("symbol", "")
            name = s.get("name", symbol)
            pct = s.get("percent_change", 0)
            sign = "+" if pct >= 0 else ""
            stock_lines.append(f"{name} ({symbol}): {sign}{pct:.1f}%")

        if not stock_lines:
            return stocks

        news_lines = [
            f"- {_sanitize_prompt_input(n.get('title', ''), max_length=300)}" for n in news[:15]
        ]

        stocks_block = "\n".join(stock_lines)
        headlines_block = "\n".join(news_lines)
        prompt = (
            "These stocks moved today:\n"
            f"<stocks>\n{stocks_block}\n</stocks>\n\n"
            "Today's headlines:\n"
            f"<headlines>\n{headlines_block}\n</headlines>\n\n"
            "For EVERY stock, write a short driver (max 4 words). "
            "Use the headlines if related, otherwise use general market context "
            "(e.g. 'Broad tech selloff', 'Sector rotation').\n"
            "Respond with one line per stock:\n"
            "SYMBOL | short driver\n"
            "Every stock MUST have a driver. Never leave blank."
        )

        result = self.bedrock.invoke(
            prompt, tier="heavy", max_tokens=500, system_prompt=SYSTEM_PROMPT,
            name="stock_correlation",
        )
        if not result:
            return stocks

        # Parse correlations
        correlations = {}
        for line in result.strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 1)
                symbol = parts[0].strip().upper()
                correlation = parts[1].strip()
                if correlation and correlation.lower() != "no clear driver":
                    correlations[symbol] = correlation

        for stock in stocks:
            symbol = stock.get("symbol", "")
            if symbol in correlations:
                stock["news_correlation"] = correlations[symbol]

        logger.info(f"Correlated {len(correlations)} stocks with news")
        return stocks

    def detect_emerging_themes(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        newsletters: Optional[List[Dict[str, Any]]] = None,
        github_trending: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Identify emerging themes across today's content not in configured topics.

        Args:
            papers: ArXiv papers.
            blogs: Blog articles.
            news: News articles.
            newsletters: Newsletter items from Supabase.
            github_trending: GitHub trending repos from Supabase.

        Returns:
            List of emerging theme descriptions (may be empty).
        """
        if not self.available:
            return []

        titles = []
        for p in papers[:15]:
            titles.append(f"[paper] {p.get('title', '')}")
        for b in blogs[:10]:
            titles.append(f"[blog] {b.get('title', '')}")
        for n in news[:10]:
            titles.append(f"[news] {n.get('title', '')}")
        for nl in (newsletters or [])[:8]:
            titles.append(f"[newsletter] {nl.get('title', '')}")
        for gh in (github_trending or [])[:5]:
            titles.append(f"[github] {gh.get('title', '')}")

        if not titles:
            return []

        titles_block = "\n".join(titles)
        topics_str = ", ".join(self.topics)
        prompt = (
            "Given today's papers, blogs, news, newsletters, and GitHub trending repos, "
            "identify 2-3 emerging themes "
            "or trends that are NOT already covered by these configured topics:\n\n"
            f"<configured_topics>{topics_str}</configured_topics>\n\n"
            f"<content>\n{titles_block}\n</content>\n\n"
            "For each theme, write one line: THEME: brief description\n"
            "Only list genuinely new/emerging themes. If nothing stands out, "
            "respond with NONE."
        )

        result = self.bedrock.invoke(prompt, tier="light", max_tokens=300, name="detect_themes")
        if not result or "NONE" in result.upper():
            return []

        themes = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("THEME:"):
                themes.append(line[6:].strip())

        if themes:
            logger.info(f"Detected emerging themes: {themes}")
        return themes

    def synthesize_briefing(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        stocks: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        top_papers: List[Dict[str, Any]],
        emerging_themes: Optional[List[str]] = None,
        previous_state: Optional[Dict[str, Any]] = None,
        newsletters: Optional[List[Dict[str, Any]]] = None,
        github_trending: Optional[List[Dict[str, Any]]] = None,
        obsidian_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Synthesize cross-section connections and generate editorial content.

        Uses the heavy tier model for deep reasoning.

        Args:
            papers: ArXiv papers (full list for context).
            blogs: Blog articles.
            stocks: Stock data.
            news: News articles.
            top_papers: Top-scored papers.
            emerging_themes: Emerging themes detected from today's content.
            previous_state: Previous briefing state for trend tracking.
            newsletters: Newsletter items.
            github_trending: GitHub trending repos.
            obsidian_context: Cross-day context from Obsidian vault (themes, entities).

        Returns:
            Dictionary with key:
              - 'editorial_intro': Executive summary paragraph for the briefing.
        """
        if not self.available:
            return {}

        # Build a compact summary of all data for the synthesis prompt
        sections = []

        if papers:
            paper_titles = [p.get("title", "") for p in papers[:10]]
            sections.append(
                "PAPERS (" + str(len(papers)) + " total):\n"
                + "\n".join(f"- {t}" for t in paper_titles)
            )

        if blogs:
            blog_items = [
                f"- [{b.get('source', '')}] {b.get('title', '')}"
                for b in blogs[:8]
            ]
            sections.append("BLOGS:\n" + "\n".join(blog_items))

        if stocks:
            stock_items = []
            for s in stocks:
                if "error" not in s:
                    pct = s.get("percent_change", 0)
                    sign = "+" if pct >= 0 else ""
                    corr = s.get("news_correlation", "")
                    line = f"- {s.get('symbol', '')}: {sign}{pct:.1f}%"
                    if corr:
                        line += f" ({corr})"
                    stock_items.append(line)
            if stock_items:
                sections.append("STOCKS:\n" + "\n".join(stock_items))

        if news:
            news_titles = [n.get("title", "") for n in news[:10]]
            sections.append("NEWS:\n" + "\n".join(f"- {t}" for t in news_titles))

        if top_papers:
            top_items = []
            for p in top_papers:
                reason = p.get("relevance_reason", "")
                top_items.append(
                    f"- {p.get('title', '')} (score: {p.get('score', 0):.1f})"
                    + (f" -- {reason}" if reason else "")
                )
            sections.append("TOP PAPERS FOR REPRODUCTION:\n" + "\n".join(top_items))

        if newsletters:
            nl_items = [
                f"- [{item.get('source', '')}] {item.get('title', '')}"
                for item in newsletters[:8]
            ]
            sections.append("NEWSLETTERS:\n" + "\n".join(nl_items))

        if github_trending:
            gh_items = [
                f"- {repo.get('title', '')} ({repo.get('stars', '')} stars, {repo.get('language', '')})"
                for repo in github_trending[:8]
            ]
            sections.append("GITHUB TRENDING:\n" + "\n".join(gh_items))

        if emerging_themes:
            sections.append(
                "EMERGING THEMES (not in configured topics):\n"
                + "\n".join(f"- {t}" for t in emerging_themes)
            )

        if previous_state:
            prev_parts = []
            prev_date = previous_state.get("date", "unknown")
            prev_parts.append(f"PREVIOUS BRIEFING ({prev_date}):")
            prev_themes = previous_state.get("emerging_themes", [])
            if prev_themes:
                prev_parts.append(f"Themes: {', '.join(prev_themes)}")
            prev_stocks = previous_state.get("stock_closes", {})
            if prev_stocks and stocks:
                trend_lines = []
                for s in stocks:
                    sym = s.get("symbol", "")
                    if sym in prev_stocks and "error" not in s:
                        prev_price = prev_stocks[sym]
                        curr_price = s.get("current_price", 0)
                        if prev_price and prev_price > 0:
                            multi_day_pct = ((curr_price - prev_price) / prev_price) * 100
                            trend_lines.append(f"{sym}: {multi_day_pct:+.1f}% over 2 days")
                if trend_lines:
                    prev_parts.append("Multi-day trends: " + "; ".join(trend_lines))
            if len(prev_parts) > 1:
                sections.append("\n".join(prev_parts))

        if obsidian_context:
            vault_parts = []
            yt = obsidian_context.get("yesterday_themes", [])
            if yt:
                vault_parts.append(f"Yesterday's themes: {', '.join(yt)}")
            et = obsidian_context.get("entity_timelines", {})
            for name, dates in et.items():
                if dates:
                    vault_parts.append(f"{name} mentioned on: {', '.join(dates[-3:])}")
            if vault_parts:
                sections.append(
                    "OBSIDIAN VAULT CONTEXT (cross-day continuity):\n"
                    + "\n".join(f"- {p}" for p in vault_parts)
                )

        if not sections:
            return {}

        all_data = "\n\n".join(sections)

        # Detect cross-source correlations
        cross_source_signals = self._detect_cross_source_signals(
            papers, blogs, news, newsletters, github_trending,
        )
        cross_source_note = ""
        if cross_source_signals:
            cross_source_note = (
                "\n\n<cross_source_signals>\n"
                "These topics appear in 2+ sources (PRIORITIZE in summary):\n"
                + "\n".join(f"- {s}" for s in cross_source_signals)
                + "\n</cross_source_signals>"
            )

        prompt = (
            "You are writing a daily AI research + market briefing. "
            "Based on today's data below, write a 3-5 sentence executive summary "
            "highlighting today's key theme, most notable findings, and connections "
            "across papers, news, and blogs. "
            "If emerging themes or multi-day trends are present, mention them. "
            "IMPORTANT: Topics appearing in cross-source signals should be emphasized "
            "as they represent strong multi-source confirmation. "
            "Be specific. Only reference items from the data provided below.\n\n"
            f"<data>\n{all_data}\n</data>"
            f"{cross_source_note}"
        )

        result = self.bedrock.invoke(
            prompt, tier="heavy", max_tokens=1000, system_prompt=SYSTEM_PROMPT,
            name="synthesize_briefing",
        )
        if not result:
            return {}

        logger.info("Briefing synthesis complete")
        return {"editorial_intro": result.strip()}

    def track_trending(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        state: Dict[str, Any],
        newsletters: Optional[List[Dict[str, Any]]] = None,
        github_trending: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Track topics that appear across multiple days and mark trending items.

        Uses Opus to cluster current items against stored trending topics.
        If a topic reappears on Day 2 or Day 3, increment its counter and mark items
        with "🔥 Day N trending" in their summary.

        Args:
            papers: Today's papers.
            blogs: Today's blogs.
            news: Today's news.
            state: Previous state with trending_topics.
            newsletters: Newsletter items from Supabase.
            github_trending: GitHub trending repos from Supabase.

        Returns:
            Tuple of (updated_state, annotated_papers, annotated_blogs, annotated_news).
        """
        if not self.available:
            return state, papers, blogs, news

        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        trending_topics = state.get("trending_topics", {})

        # Build list of current items
        current_items = []
        for p in papers[:10]:
            current_items.append(f"[paper] {p.get('title', '')}")
        for b in blogs[:10]:
            current_items.append(f"[blog] {b.get('title', '')}")
        for n in news[:10]:
            current_items.append(f"[news] {n.get('title', '')}")
        for nl in (newsletters or [])[:8]:
            current_items.append(f"[newsletter] {nl.get('title', '')}")
        for gh in (github_trending or [])[:5]:
            current_items.append(f"[github] {gh.get('title', '')}")

        if not current_items:
            return state, papers, blogs, news

        # Build trending topics summary
        trending_summary = []
        for topic_key, info in trending_topics.items():
            count = info.get("count", 1)
            first_seen = info.get("first_seen", "")
            last_seen = info.get("last_seen", "")
            trending_summary.append(f"- {topic_key}: count={count}, first={first_seen}, last={last_seen}")

        items_block = "\n".join(current_items)
        trending_block = "\n".join(trending_summary) if trending_summary else "NONE"

        prompt = (
            f"Today is {today}. You are tracking trending topics across days.\n\n"
            "<current_items>\n"
            f"{items_block}\n"
            "</current_items>\n\n"
            "<previous_trending_topics>\n"
            f"{trending_block}\n"
            "</previous_trending_topics>\n\n"
            "For each current item, determine if it matches or is closely related to a previous trending topic. "
            "If it matches, output: [item_index] MATCH topic_key\n"
            "If it's a NEW emerging topic appearing 2+ times today, output: [item_index] NEW topic_keyword\n"
            "If it's neither, skip it.\n\n"
            "Example output:\n"
            "[2] MATCH flash-attention-4\n"
            "[5] NEW claude-3.5-haiku\n"
        )

        result = self.bedrock.invoke(
            prompt, tier="light", max_tokens=800, system_prompt=SYSTEM_PROMPT,
            name="track_trending",
        )
        if not result:
            logger.info("Trending tracking skipped (LLM unavailable)")
            return state, papers, blogs, news

        # Parse the result
        matches = {}
        new_topics = {}
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            try:
                bracket_end = line.index("]")
                idx = int(line[1:bracket_end]) - 1
                rest = line[bracket_end + 1:].strip()
                if rest.startswith("MATCH"):
                    topic_key = rest.split("MATCH", 1)[1].strip()
                    matches[idx] = topic_key
                elif rest.startswith("NEW"):
                    topic_key = rest.split("NEW", 1)[1].strip()
                    new_topics[idx] = topic_key
            except (ValueError, IndexError):
                continue

        # Update trending_topics and annotate items
        updated_trending = trending_topics.copy()
        annotated_count = 0

        # Process matches
        for idx, topic_key in matches.items():
            if topic_key in updated_trending:
                updated_trending[topic_key]["count"] += 1
                updated_trending[topic_key]["last_seen"] = today
                day_count = updated_trending[topic_key]["count"]

                # Annotate the item
                if idx < len(current_items):
                    item_type = current_items[idx].split("]")[0][1:]  # Extract type: paper/blog/news
                    if item_type == "paper" and idx < len(papers):
                        # Annotate as trending but DON'T inject "Day N" into summary text
                        # (user found it confusing in final output)
                        papers[idx]["_trending_days"] = day_count
                        annotated_count += 1
                    elif item_type == "blog":
                        blog_idx = idx - len([i for i in current_items[:idx] if "[paper]" in i])
                        if 0 <= blog_idx < len(blogs):
                            blogs[blog_idx]["_trending_days"] = day_count
                            annotated_count += 1
                    elif item_type == "news":
                        news_idx = idx - len([i for i in current_items[:idx] if "[paper]" in i or "[blog]" in i])
                        if 0 <= news_idx < len(news):
                            news[news_idx]["_trending_days"] = day_count
                            annotated_count += 1

        # Process new topics
        for idx, topic_key in new_topics.items():
            if topic_key not in updated_trending:
                updated_trending[topic_key] = {
                    "first_seen": today,
                    "count": 1,
                    "last_seen": today,
                }

        # Clean up old trending topics (older than 3 days)
        from datetime import datetime
        today_date = datetime.strptime(today, "%Y-%m-%d")
        cleaned_trending = {}
        for topic_key, info in updated_trending.items():
            last_seen = datetime.strptime(info["last_seen"], "%Y-%m-%d")
            if (today_date - last_seen).days <= 3:
                cleaned_trending[topic_key] = info

        state["trending_topics"] = cleaned_trending
        logger.info(f"Trending tracking: {annotated_count} items marked, {len(cleaned_trending)} topics tracked")
        return state, papers, blogs, news

    def detect_entity_mentions(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        tracked_entities: List[Dict[str, str]],
        newsletters: Optional[List[Dict[str, Any]]] = None,
        github_trending: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect mentions of tracked entities (companies/people) in content.

        Args:
            papers: Today's papers.
            blogs: Today's blogs.
            news: Today's news.
            tracked_entities: List of entities to track with name and type.
            newsletters: Newsletter items from Supabase.
            github_trending: GitHub trending repos from Supabase.

        Returns:
            List of entity mention dicts with name, type, count, and example_titles.
        """
        if not tracked_entities:
            return []

        entity_mentions = {}

        # Scan all items for entity mentions
        all_items = []
        for p in papers:
            all_items.append({
                "title": p.get("title", ""),
                "summary": p.get("brief_summary", "") or p.get("summary", "")[:200],
                "type": "paper",
            })
        for b in blogs:
            all_items.append({
                "title": b.get("title", ""),
                "summary": b.get("brief_summary", "") or b.get("summary", "")[:200],
                "type": "blog",
            })
        for n in news:
            all_items.append({
                "title": n.get("title", ""),
                "summary": n.get("brief_summary", "") or n.get("description", "")[:200],
                "type": "news",
            })
        for nl in (newsletters or []):
            all_items.append({
                "title": nl.get("title", ""),
                "summary": nl.get("summary", "")[:200],
                "type": "newsletter",
            })
        for gh in (github_trending or []):
            all_items.append({
                "title": gh.get("title", ""),
                "summary": gh.get("summary", "")[:200],
                "type": "github",
            })

        # Case-insensitive substring matching
        for entity in tracked_entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "")
            if not entity_name:
                continue

            entity_name_lower = entity_name.lower()
            matches = []

            for item in all_items:
                title = item.get("title", "").lower()
                summary = item.get("summary", "").lower()

                if entity_name_lower in title or entity_name_lower in summary:
                    matches.append(item.get("title", ""))

            if matches:
                entity_mentions[entity_name] = {
                    "name": entity_name,
                    "type": entity_type,
                    "count": len(matches),
                    "example_titles": matches[:3],  # Keep top 3 examples
                }

        # Convert to list and sort by count (descending)
        result = list(entity_mentions.values())
        result.sort(key=lambda x: x["count"], reverse=True)

        if result:
            logger.info(f"Entity Watch: detected {len(result)} entities with mentions")
        return result

    def generate_weekly_deep_dive(
        self,
        weekly_items: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a 'This Week in AI' deep dive section.

        Uses Opus (heavy tier) to synthesize a narrative from the week's items,
        identifying the 3 biggest themes, explaining why they matter, and predicting
        what to watch next week.

        Args:
            weekly_items: List of items accumulated over the week with date.

        Returns:
            Markdown string for the "This Week in AI" section (500-800 words).
        """
        if not self.available or not weekly_items:
            return ""

        # Group items by date
        items_by_date = {}
        for item in weekly_items:
            date = item.get("date", "unknown")
            if date not in items_by_date:
                items_by_date[date] = []
            items_by_date[date].append(item)

        # Build context from weekly items
        context_parts = []
        for date in sorted(items_by_date.keys()):
            items = items_by_date[date]
            titles = [f"- {i.get('title', '')} ({i.get('type', 'item')})" for i in items]
            context_parts.append(f"{date}:\n" + "\n".join(titles))

        if not context_parts:
            return ""

        context_str = "\n\n".join(context_parts)

        prompt = (
            "You are writing a 'This Week in AI' section for a weekly research briefing. "
            "Based on this week's papers, blogs, and news below, synthesize a narrative that:\n\n"
            "1. Identifies the 3 biggest themes of the week\n"
            "2. Explains why they matter (implications for researchers/engineers)\n"
            "3. Predicts what to watch next week\n\n"
            "Write 500-800 words. Be analytical, opinionated, and forward-looking. "
            "Focus on connections and patterns across the week.\n\n"
            f"<week_items>\n{context_str}\n</week_items>"
        )

        result = self.bedrock.invoke(
            prompt, tier="heavy", max_tokens=1500, system_prompt=SYSTEM_PROMPT,
            name="weekly_deep_dive",
        )

        if result:
            logger.info("Weekly Deep Dive generated successfully")
            return result.strip()

        logger.warning("Weekly Deep Dive generation failed")
        return ""

    def _detect_cross_source_signals(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        newsletters: Optional[List[Dict[str, Any]]] = None,
        github_trending: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Detect topics that appear in 2+ sources (papers, blogs, news, newsletters, github).

        Uses simple keyword matching to find cross-source correlations.

        Args:
            papers: ArXiv papers.
            blogs: Blog articles.
            news: News articles.
            newsletters: Newsletter items from Supabase.
            github_trending: GitHub trending repos from Supabase.

        Returns:
            List of cross-source topics (e.g., "Claude 3.5", "Trainium", etc.)
        """
        # Extract key terms from each source
        def extract_terms(items: List[Dict[str, Any]], key: str = "title") -> set:
            """Extract significant terms (2+ words) from titles."""
            terms = set()
            for item in items[:15]:  # Top 15 items per source
                text = item.get(key, "").lower()
                # Extract multi-word phrases (simple approach)
                words = text.split()
                # Look for capitalized phrases and specific keywords
                for i in range(len(words)):
                    # 2-word phrases
                    if i + 1 < len(words):
                        phrase = f"{words[i]} {words[i+1]}"
                        if len(phrase) > 6:  # Filter short phrases
                            terms.add(phrase)
                    # 3-word phrases
                    if i + 2 < len(words):
                        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                        if len(phrase) > 10:
                            terms.add(phrase)
            return terms

        paper_terms = extract_terms(papers[:15])
        blog_terms = extract_terms(blogs[:10])
        news_terms = extract_terms(news[:10])
        newsletter_terms = extract_terms((newsletters or [])[:8])
        github_terms = extract_terms((github_trending or [])[:5])

        # Find terms appearing in 2+ sources
        cross_source = []
        all_sources = [
            ("papers", paper_terms),
            ("blogs", blog_terms),
            ("news", news_terms),
            ("newsletters", newsletter_terms),
            ("github", github_terms),
        ]

        checked_terms = set()
        for source_name, terms in all_sources:
            for term in terms:
                if term in checked_terms:
                    continue
                checked_terms.add(term)

                # Check if this term appears in other sources
                match_count = 0
                matched_sources = []
                for other_name, other_terms in all_sources:
                    # Use fuzzy matching: check if term is substring of any other term
                    if any(term in other_term or other_term in term for other_term in other_terms):
                        match_count += 1
                        matched_sources.append(other_name)

                if match_count >= 2:
                    cross_source.append(f"{term.title()} ({', '.join(matched_sources)})")

        # Limit to top 5 cross-source signals
        if cross_source:
            logger.info(f"Detected {len(cross_source)} cross-source signals")
        return cross_source[:5]


def _parse_numbered_list(text: str, expected_count: int) -> List[str]:
    """
    Parse a numbered list response from the model.

    Args:
        text: Model response text.
        expected_count: Expected number of items.

    Returns:
        List of parsed items.
    """
    items = []
    current_lines = []
    current_num = -1

    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Check for numbered item start: [1], 1., 1)
        new_num = None
        if stripped.startswith("[") and "]" in stripped:
            try:
                bracket_end = stripped.index("]")
                new_num = int(stripped[1:bracket_end])
                stripped = stripped[bracket_end + 1:].strip()
            except (ValueError, IndexError):
                pass
        elif stripped[0].isdigit():
            for sep in [".", ")", ":"]:
                if sep in stripped[:4]:
                    try:
                        new_num = int(stripped[: stripped.index(sep)])
                        stripped = stripped[stripped.index(sep) + 1:].strip()
                    except (ValueError, IndexError):
                        pass
                    break

        if new_num is not None and new_num != current_num:
            if current_lines:
                items.append(" ".join(current_lines))
            current_num = new_num
            current_lines = [stripped] if stripped else []
        else:
            current_lines.append(stripped)

    if current_lines:
        items.append(" ".join(current_lines))

    return items[:expected_count]
