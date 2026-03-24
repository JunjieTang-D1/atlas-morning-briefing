#!/usr/bin/env python3
# Copyright (c) 2026. MIT License. See LICENSE file for details.
"""
Centralized prompt definitions for the intelligence layer.

All LLM prompts are defined here as string templates so they can be
reviewed, versioned, and modified independently of the pipeline logic.
Use Python format strings ({variable}) for dynamic content injection.
"""

# ---------------------------------------------------------------------------
# System prompt — shared across all Bedrock invocations
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an AI Security research analyst generating a daily morning briefing. "
    "Be concise, insightful, and factual. Use markdown formatting. "
    "Do not invent facts or citations. If information is insufficient, say so. "
    "Provide the Summary in German."
)

# ---------------------------------------------------------------------------
# Stage 1: Relevance filtering
# ---------------------------------------------------------------------------
RELEVANCE_FILTER_PROMPT = (
    "You are filtering papers for a daily AI Security research briefing. "
    "Score each paper 1-10 for relevance to this interest profile:\n\n"
    "<interest_profile>\n{profile_str}\n</interest_profile>\n\n"
    "<papers>\n{papers_block}\n</papers>\n\n"
    "Return ONLY papers scoring >= 7. For each relevant paper, respond with:\n"
    "[number] score reason\n"
    "Example: [5] 9 Directly addresses multi-agent systems with novel evaluation methodology\n\n"
    "Be selective. Only include papers that strongly match the profile."
)

# ---------------------------------------------------------------------------
# Dynamic news query generation
# ---------------------------------------------------------------------------
DYNAMIC_QUERIES_PROMPT = (
    "You are generating follow-up news queries based on yesterday's AI Security research briefing.\n\n"
    "<yesterday_briefing>\n{context_str}\n</yesterday_briefing>\n\n"
    "<static_queries>\n{static_queries_str}\n</static_queries>\n\n"
    "Generate 3 targeted follow-up queries to track developments in yesterday's hot topics. "
    "Return ONLY the new queries, one per line, no numbering or bullets. "
    "Make them specific and actionable for news search.\n\n"
    "Example outputs:\n"
    "- Claude 3.5 Sonnet benchmark results\n"
    "- AWS Trainium chip adoption enterprise\n"
    "- Multi-agent orchestration frameworks release"
)

# ---------------------------------------------------------------------------
# Topic expansion
# ---------------------------------------------------------------------------
TOPIC_EXPANSION_PROMPT = (
    "Given these research topics, suggest 2-3 additional related search "
    "queries that would find relevant papers on arxiv. Return ONLY the "
    "new queries, one per line, no numbering or bullets.\n\n"
    "<topics>\n{topic_list}\n</topics>"
)

# ---------------------------------------------------------------------------
# Paper summarization (batched)
# ---------------------------------------------------------------------------
PAPER_SUMMARIZATION_PROMPT = (
    "For each paper below, write a 1-2 sentence summary that captures "
    "the key contribution. Return as a numbered list matching the input "
    "numbering. Be factual -- do not add information not in the abstract.\n\n"
    "<papers>\n{papers_block}\n</papers>"
)

# ---------------------------------------------------------------------------
# Semantic paper scoring
# ---------------------------------------------------------------------------
SEMANTIC_SCORING_PROMPT = (
    "Rate each paper's relevance to these research interests on a 0-10 scale.\n\n"
    "<interests>{interests_str}</interests>\n\n"
    "<papers>\n{papers_block}\n</papers>\n\n"
    "For each paper, respond with ONLY this format, one per line:\n"
    "[number] score reason\n"
    "Example: [1] 8 Directly addresses agent evaluation methodology"
)

# ---------------------------------------------------------------------------
# Reproduction feasibility assessment
# ---------------------------------------------------------------------------
REPRODUCTION_ASSESSMENT_PROMPT = (
    "You are evaluating papers for PRACTICAL reproduction on this setup:\n"
    "- Single EC2 GPU instance available (g5.xlarge = 1x A10G 24GB, or trn1.2xlarge = AWS Trainium)\n"
    "- Amazon Bedrock API (Claude Sonnet/Opus, Titan Embeddings)\n"
    "- Python + standard ML libraries, Kubernetes OK if single-node\n"
    "- Budget: <$50 per paper, <1 week effort\n\n"
    "Score each paper on 5 dimensions (1-5 each, 25 max):\n"
    "1. code_available: 5=open repo+README, 3=partial code, 1=no code\n"
    "2. data_accessible: 5=open data <50GB, 3=needs request/large, 1=proprietary\n"
    "3. infra_fit: 5=CPU/API only, 4=single GPU(A10G/Trainium), 3=multi-GPU single node, "
    "2=multi-node cluster, 1=datacenter/TPU pod\n"
    "4. bedrock_ready: 5=can swap in Bedrock models directly, 3=needs adapter, 1=incompatible\n"
    "5. effort: 5=weekend(S), 4=1week(M), 3=2weeks(L), 2=month(XL), 1=impossible\n\n"
    "For each paper respond in this EXACT format (one line each):\n"
    "[number] code:X data:X infra:X bedrock:X effort:X | verdict\n\n"
    "Example: [1] code:5 data:4 infra:5 bedrock:5 effort:4 | Open benchmark + Bedrock RAG, easy to reproduce\n"
    "Example: [2] code:1 data:1 infra:1 bedrock:2 effort:1 | No code, needs GPU cluster, skip\n\n"
    "<papers>\n{papers_block}\n</papers>"
)

# ---------------------------------------------------------------------------
# News ranking and summarization
# ---------------------------------------------------------------------------
NEWS_RANKING_PROMPT = (
    "You are curating a daily AI/tech Security briefing. From these news articles, "
    "select the TOP 5 most important for an AI Security researcher/engineer.\n\n"
    "<interests>{interests_str}</interests>\n\n"
    "<articles>\n{articles_block}\n</articles>\n\n"
    "For each of your top 5 picks, respond in this exact format:\n"
    "[original_number] 2-3 sentence summary explaining why this matters.\n\n"
    "Rank by importance. Be factual. Do not invent details."
)

NEWS_RANKING_RETRY_PROMPT = (
    "From these articles, pick the 5 most important for an AI Security researcher. "
    "Format EXACTLY as: [number] summary sentence.\n\n{articles_block}"
)

# ---------------------------------------------------------------------------
# Blog ranking and summarization
# ---------------------------------------------------------------------------
BLOG_RANKING_PROMPT = (
    "You are curating a daily AI/tech Security briefing. From these blog posts, "
    "select the TOP 5 most relevant for an AI Security researcher/engineer.\n\n"
    "<interests>{interests_str}</interests>\n\n"
    "<blogs>\n{blogs_block}\n</blogs>\n\n"
    "For each of your top 5 picks, respond in this exact format:\n"
    "[original_number] SCORE:X/5 1-2 sentence summary of what the post covers.\n\n"
    "SCORE is a combined rating (1-5) of impact, complexity, and innovation. "
    "5 = groundbreaking, 1 = routine.\n"
    "Rank by relevance. Be concise."
)

# ---------------------------------------------------------------------------
# Stock-news correlation
# ---------------------------------------------------------------------------
STOCK_CORRELATION_PROMPT = (
    "These stocks moved today:\n"
    "<stocks>\n{stocks_block}\n</stocks>\n\n"
    "Today's headlines:\n"
    "<headlines>\n{headlines_block}\n</headlines>\n\n"
    "For EVERY stock, write a short driver (max 4 words). "
    "Use the headlines if related, otherwise use general market context "
    "(e.g. 'Broad tech selloff', 'Sector rotation').\n"
    "Respond with one line per stock:\n"
    "SYMBOL | short driver\n"
    "Every stock MUST have a driver. Never leave blank."
)

# ---------------------------------------------------------------------------
# Emerging theme detection
# ---------------------------------------------------------------------------
EMERGING_THEMES_PROMPT = (
    "Given today's papers, blogs, and news, identify 2-3 emerging themes "
    "or trends that are NOT already covered by these configured topics:\n\n"
    "<configured_topics>{topics_str}</configured_topics>\n\n"
    "<content>\n{titles_block}\n</content>\n\n"
    "For each theme, write one line: THEME: brief description\n"
    "Only list genuinely new/emerging themes. If nothing stands out, "
    "respond with NONE."
)

# ---------------------------------------------------------------------------
# Editorial synthesis (executive summary)
# ---------------------------------------------------------------------------
SYNTHESIS_PROMPT = (
    "You are writing a daily AI Security research + market briefing. "
    "Based on today's data below, write a 3-5 sentence executive summary "
    "highlighting today's key theme, most notable findings, and connections "
    "across papers, news, and blogs. "
    "If emerging themes or multi-day trends are present, mention them. "
    "IMPORTANT: Topics appearing in cross-source signals should be emphasized "
    "as they represent strong multi-source confirmation. "
    "Be specific. Only reference items from the data provided below.\n\n"
    "<data>\n{all_data}\n</data>"
    "{cross_source_note}"
)

# ---------------------------------------------------------------------------
# Trending topic tracking
# ---------------------------------------------------------------------------
TRENDING_TRACKING_PROMPT = (
    "Today is {today}. You are tracking trending topics across days.\n\n"
    "<current_items>\n"
    "{items_block}\n"
    "</current_items>\n\n"
    "<previous_trending_topics>\n"
    "{trending_block}\n"
    "</previous_trending_topics>\n\n"
    "For each current item, determine if it matches or is closely related to a previous trending topic. "
    "If it matches, output: [item_index] MATCH topic_key\n"
    "If it's a NEW emerging topic appearing 2+ times today, output: [item_index] NEW topic_keyword\n"
    "If it's neither, skip it.\n\n"
    "Example output:\n"
    "[2] MATCH flash-attention-4\n"
    "[5] NEW claude-3.5-haiku\n"
)

# ---------------------------------------------------------------------------
# Weekly deep dive (Saturday only)
# ---------------------------------------------------------------------------
WEEKLY_DEEP_DIVE_PROMPT = (
    "You are writing a 'This Week in AI Security' section for a weekly research briefing. "
    "Based on this week's papers, blogs, and news below, synthesize a narrative that:\n\n"
    "1. Identifies the 3 biggest themes of the week\n"
    "2. Explains why they matter (implications for researchers/engineers)\n"
    "3. Predicts what to watch next week\n\n"
    "Write 500-800 words. Be analytical, opinionated, and forward-looking. "
    "Focus on connections and patterns across the week.\n\n"
    "<week_items>\n{context_str}\n</week_items>"
)
