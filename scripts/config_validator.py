#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Configuration validator.

Validates config.yaml values at startup to catch errors early.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration loaded from YAML.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
        is_valid is True if no critical errors found.
    """
    errors = []
    warnings = []

    # --- Required fields ---
    if not isinstance(config.get("arxiv_topics"), list):
        errors.append("'arxiv_topics' must be a list of strings")
    elif not config["arxiv_topics"]:
        warnings.append("'arxiv_topics' is empty -- no papers will be scanned")

    # --- Type checks ---
    int_fields = {
        "arxiv_days_back": (1, 365),
        "max_papers": (1, 200),
        "max_blogs": (1, 100),
        "max_news": (1, 100),
        "num_paper_picks": (1, 20),
    }
    for field, (min_val, max_val) in int_fields.items():
        value = config.get(field)
        if value is not None:
            if not isinstance(value, int):
                errors.append(f"'{field}' must be an integer, got {type(value).__name__}")
            elif value < min_val or value > max_val:
                warnings.append(f"'{field}' value {value} is outside recommended range ({min_val}-{max_val})")

    # --- Blog feeds ---
    feeds = config.get("blog_feeds")
    if feeds is not None:
        if not isinstance(feeds, list):
            errors.append("'blog_feeds' must be a list")
        else:
            for i, feed in enumerate(feeds):
                if not isinstance(feed, dict):
                    errors.append(f"blog_feeds[{i}] must be a dict with 'name' and 'url'")
                elif not feed.get("name") or not feed.get("url"):
                    errors.append(f"blog_feeds[{i}] missing 'name' or 'url'")

    # --- Stocks ---
    stocks = config.get("stocks")
    if stocks is not None:
        if not isinstance(stocks, list):
            errors.append("'stocks' must be a list of ticker symbols")
        elif len(stocks) > 30:
            warnings.append(
                f"'stocks' has {len(stocks)} tickers. "
                "Finnhub free tier allows 60 calls/min; consider reducing."
            )

    # --- News queries ---
    queries = config.get("news_queries")
    if queries is not None:
        if not isinstance(queries, list):
            errors.append("'news_queries' must be a list of strings")

    # --- Kindle/email ---
    kindle_email = config.get("kindle_email", "")
    if kindle_email and "kindle" not in kindle_email.lower() and kindle_email != "YOUR_NAME@kindle.com":
        warnings.append(
            f"'kindle_email' ({kindle_email}) does not contain 'kindle' -- "
            "verify this is correct"
        )

    # --- Paper scoring ---
    scoring = config.get("paper_scoring")
    if scoring is not None:
        if not isinstance(scoring, dict):
            errors.append("'paper_scoring' must be a dictionary")
        else:
            for key in ["has_code", "topic_match", "recency", "citation_count"]:
                val = scoring.get(key)
                if val is not None and not isinstance(val, (int, float)):
                    errors.append(f"paper_scoring.{key} must be a number")

    # --- PDF settings ---
    pdf = config.get("pdf")
    if pdf is not None:
        if not isinstance(pdf, dict):
            errors.append("'pdf' must be a dictionary")
        else:
            font_size = pdf.get("font_size")
            if font_size is not None and not isinstance(font_size, (int, float)):
                errors.append("pdf.font_size must be a number")
            line_spacing = pdf.get("line_spacing")
            if line_spacing is not None and not isinstance(line_spacing, (int, float)):
                errors.append("pdf.line_spacing must be a number")

    # --- Output format ---
    output_format = config.get("output_format")
    if output_format and output_format not in ("kindle", "a4", "letter", "html"):
        errors.append(f"'output_format' must be 'kindle', 'a4', or 'letter', got '{output_format}'")

    # --- LLM config ---
    llm = config.get("llm")
    if llm is not None:
        if not isinstance(llm, dict):
            errors.append("'llm' must be a dictionary")
        else:
            fallback = llm.get("fallback", {})
            if isinstance(fallback, dict):
                models = fallback.get("models")
                if models is not None:
                    if not isinstance(models, dict):
                        errors.append("llm.fallback.models must be a dictionary")
                    else:
                        for tier in models:
                            if tier not in ("heavy", "medium", "light"):
                                warnings.append(
                                    f"llm.fallback.models.{tier} is not a recognized tier "
                                    "(expected: heavy, medium, light)"
                                )

    # --- Obsidian config ---
    obsidian = config.get("obsidian")
    if obsidian is not None:
        if not isinstance(obsidian, dict):
            errors.append("'obsidian' must be a dictionary")
        elif obsidian.get("enabled", False):
            if not obsidian.get("api_url"):
                warnings.append(
                    "obsidian.api_url not set, defaulting to http://localhost:27123"
                )
            threshold = obsidian.get("trending_threshold")
            if threshold is not None and (
                not isinstance(threshold, int) or threshold < 1
            ):
                errors.append(
                    "obsidian.trending_threshold must be a positive integer"
                )

    # --- Podcast config ---
    podcast = config.get("podcast")
    if podcast is not None:
        if not isinstance(podcast, dict):
            errors.append("'podcast' must be a dictionary")
        else:
            valid_formats = {"brief", "deep-dive", "deep_dive", "critique", "debate"}
            fmt = podcast.get("audio_format", "brief")
            if fmt.lower().replace("-", "_") not in {f.replace("-", "_") for f in valid_formats}:
                errors.append(
                    f"podcast.audio_format must be one of: brief, deep-dive, critique, debate; got '{fmt}'"
                )
            valid_lengths = {"short", "default", "long"}
            length = podcast.get("audio_length", "short")
            if length.lower() not in valid_lengths:
                errors.append(
                    f"podcast.audio_length must be one of: short, default, long; got '{length}'"
                )
            max_wait = podcast.get("max_wait_seconds")
            if max_wait is not None and (
                not isinstance(max_wait, (int, float)) or max_wait < 60
            ):
                errors.append("podcast.max_wait_seconds must be a number >= 60")

    # --- Log results ---
    for w in warnings:
        logger.warning(f"Config warning: {w}")
    for e in errors:
        logger.error(f"Config error: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors + warnings


def check_environment(config: Dict[str, Any], dry_run: bool = False) -> List[str]:
    """
    Check required environment variables based on config.

    Args:
        config: Configuration dictionary.
        dry_run: If True, skip email credential checks.

    Returns:
        List of warning messages for missing variables.
    """
    warnings = []

    # Stocks require Finnhub key
    if config.get("stocks"):
        if not os.environ.get("FINNHUB_API_KEY"):
            warnings.append(
                "FINNHUB_API_KEY not set -- stock data will be skipped"
            )

    # News requires Brave key
    if config.get("news_queries"):
        if not os.environ.get("BRAVE_API_KEY"):
            warnings.append(
                "BRAVE_API_KEY not set -- news aggregation will be skipped"
            )

    # Email requires Gmail credentials (unless dry run)
    if not dry_run:
        if not os.environ.get("GMAIL_USER"):
            warnings.append("GMAIL_USER not set -- Kindle delivery will be skipped")
        if not os.environ.get("GMAIL_APP_PASSWORD"):
            warnings.append(
                "GMAIL_APP_PASSWORD not set -- Kindle delivery will be skipped"
            )

    # Obsidian requires API key
    obsidian = config.get("obsidian", {})
    if obsidian.get("enabled", False):
        if not os.environ.get("OBSIDIAN_API_KEY"):
            warnings.append(
                "OBSIDIAN_API_KEY not set -- Obsidian publish will be skipped"
            )

    # Podcast requires NotebookLM storage state
    podcast = config.get("podcast", {})
    if podcast.get("enabled", False):
        has_path = (
            podcast.get("storage_state_path")
            or os.environ.get("NOTEBOOKLM_STORAGE_STATE_PATH")
            or os.environ.get("NOTEBOOKLM_STORAGE_STATE_B64")
        )
        if not has_path:
            warnings.append(
                "podcast.enabled is true but no NotebookLM auth configured. "
                "Run `notebooklm login` or set NOTEBOOKLM_STORAGE_STATE_B64."
            )

    for w in warnings:
        logger.warning(w)

    return warnings
