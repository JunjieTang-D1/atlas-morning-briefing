#!/usr/bin/env python3
"""
Batch Gmail Digest — processes a backlog of unread newsletter emails.

Two-pass pipeline:
  Pass 1: Fetch all unread emails in batches, extract URLs, score via LLM
  Pass 2: Filter, deduplicate, cluster by topic, generate per-cluster digests
  Pass 3: Render markdown + generate NotebookLM podcast per cluster

Usage:
    python scripts/batch_gmail_digest.py --config config-local.yaml [--batch-size 50] [--dry-run]
"""

import argparse
import imaplib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gmail_scanner import GmailScanner
from scripts.intelligence import BriefingIntelligence
from scripts.llm_client import LLMClient
from scripts.obsidian_writer import ObsidianWriter
from scripts.podcast_generator import PodcastGenerator
from scripts.briefing_runner import BriefingCoordinator
from scripts.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STATE_FILE = ".batch-gmail-state.json"


def _fetch_all_batch(
    source_label: str,
    batch_size: int,
    already_processed: set,
) -> tuple:
    """Fetch a batch of ALL emails (not just unread), skipping already-processed ones.

    Returns (items_list, set_of_fetched_imap_nums).
    """
    gmail_user = os.environ.get("GMAIL_USER", "")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not gmail_user or not gmail_password:
        logger.warning("Gmail credentials not configured")
        return [], set()

    import email as email_mod
    import email.message
    from scripts.gmail_scanner import _decode_header_str, _extract_body, _SNIPPET_MAX

    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        imap.login(gmail_user, gmail_password)
    except Exception as e:
        logger.error(f"Gmail IMAP login failed: {e}")
        return [], set()

    try:
        status, _ = imap.select(source_label, readonly=True)
        if status != "OK":
            logger.error(f"Cannot select mailbox '{source_label}'")
            return [], set()

        _, data = imap.search(None, "ALL")
        all_nums = data[0].split() if data[0] else []

        # Filter out already-processed and take newest first
        remaining = [n for n in all_nums if n not in already_processed]
        remaining.reverse()  # newest first
        batch_nums = remaining[:batch_size]

        if not batch_nums:
            return [], set()

        items = []
        fetched = set()
        for num in batch_nums:
            try:
                _, msg_data = imap.fetch(num, "(RFC822)")
                raw = msg_data[0][1]
                msg = email_mod.message_from_bytes(raw)

                subject = _decode_header_str(msg.get("Subject"))
                from_full = _decode_header_str(msg.get("From"))
                date_str = msg.get("Date", "")
                from_name = from_full.split("<")[0].strip().strip('"') or from_full

                body = _extract_body(msg)
                snippet = body[:_SNIPPET_MAX].strip()

                fetched.add(num)
                items.append({
                    "source": from_name,
                    "title": subject,
                    "link": "",
                    "summary": "",
                    "snippet": snippet,
                    "published": date_str,
                    "category": "",
                })
            except Exception as e:
                logger.warning(f"Failed to parse email {num}: {e}")
                fetched.add(num)  # Skip on retry too

        logger.info(f"Found {len(items)} emails in '{source_label}' (ALL mode)")
        return items, fetched

    finally:
        try:
            imap.close()
            imap.logout()
        except Exception:
            pass


# ------------------------------------------------------------------
# Pass 1: Fetch + Triage
# ------------------------------------------------------------------

def fetch_and_triage(
    config: Dict[str, Any],
    intelligence: BriefingIntelligence,
    batch_size: int,
    dry_run: bool,
    max_batches: int = 20,
    search_all: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch emails in batches, score URLs for relevance.

    Args:
        search_all: If True, fetch ALL emails (not just unread). For backlog processing.

    Returns list of scored URL items across all batches.
    In dry-run mode, only fetches ONE batch.
    """
    ns_config = config.get("newsletter_source", {})
    source_label = ns_config.get("source_label", "INBOX")
    interest_topics = config.get("interest_profile", [])

    all_scored: List[Dict[str, Any]] = []
    all_newsletters: List[Dict[str, Any]] = []
    batch_num = 0
    processed_nums: set = set()  # Track processed IMAP message numbers

    while batch_num < max_batches:
        batch_num += 1
        logger.info(f"=== Batch {batch_num}: Fetching up to {batch_size} emails ===")

        if search_all:
            emails, fetched_nums = _fetch_all_batch(
                source_label, batch_size, processed_nums
            )
            processed_nums.update(fetched_nums)
        else:
            scanner = GmailScanner(source_label=source_label, max_items=batch_size)
            emails = scanner.scan()
            fetched_nums = set()

        if not emails:
            logger.info(f"No more unread emails. Total batches: {batch_num - 1}")
            break

        logger.info(f"Batch {batch_num}: {len(emails)} emails fetched")
        all_newsletters.extend(emails)

        # Extract URLs from snippets
        picks = []
        for nl in emails:
            text = (nl.get("snippet") or "") + " " + (nl.get("summary") or "")
            urls = BriefingCoordinator._extract_urls_from_text(text)
            for url in urls:
                picks.append({
                    "url": url,
                    "title": nl.get("title", "Newsletter article"),
                    "source": nl.get("source", "Newsletter"),
                    "description": nl.get("snippet", "")[:250],
                    "source_type": "newsletter",
                    "published": nl.get("published", ""),
                })

        # Score in sub-batches of 15
        if picks and intelligence.available:
            for i in range(0, len(picks), 15):
                chunk = picks[i:i + 15]
                scored = intelligence.rank_source_links(chunk, interest_topics)
                all_scored.extend(scored)
                logger.info(f"  Sub-batch {i // 15 + 1}: {len(scored)} items scored")
        else:
            # No LLM — keep all picks unscored
            all_scored.extend(picks)

        # Mark as read (only relevant for UNSEEN mode)
        if not search_all and not dry_run:
            scanner.mark_digested()
            logger.info(f"Batch {batch_num}: marked {len(emails)} emails as read")
        elif dry_run and not search_all:
            logger.info(f"Batch {batch_num}: dry-run, skipping mark-as-read")
            logger.info("Dry-run: stopping after first batch (emails not marked as read)")
            break

    logger.info(
        f"Pass 1 complete: {len(all_newsletters)} emails, {len(all_scored)} scored URLs"
    )

    # Save intermediate state
    _save_state(all_scored, all_newsletters)
    return all_scored


def _save_state(scored: List[Dict[str, Any]], newsletters: List[Dict[str, Any]]):
    """Save intermediate triage results for resumability."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "scored_count": len(scored),
        "newsletter_count": len(newsletters),
        "scored_items": scored,
        "newsletters": newsletters,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved state to {STATE_FILE}")


def _load_state() -> Optional[Dict[str, Any]]:
    """Load previously saved triage state."""
    path = Path(STATE_FILE)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------
# Pass 2: Filter + Cluster
# ------------------------------------------------------------------

def filter_and_cluster(
    scored_items: List[Dict[str, Any]],
    llm: LLMClient,
    min_score: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Filter relevant items, deduplicate, and cluster by topic.

    Returns dict mapping cluster_name -> list of items.
    """
    # Filter by score
    relevant = [
        item for item in scored_items
        if item.get("score_combined", 0) >= min_score
    ]
    logger.info(f"Filter: {len(scored_items)} → {len(relevant)} (score >= {min_score})")

    if not relevant:
        return {}

    # Deduplicate by normalized URL
    seen_urls = set()
    deduped = []
    for item in relevant:
        url = item.get("url", "")
        parsed = urlparse(url)
        norm = f"{parsed.netloc}{parsed.path}".rstrip("/").lower()
        if norm not in seen_urls:
            seen_urls.add(norm)
            deduped.append(item)
    logger.info(f"Dedup: {len(relevant)} → {len(deduped)} unique URLs")

    if not deduped:
        return {}

    # Sort by score descending
    deduped.sort(key=lambda x: x.get("score_combined", 0), reverse=True)

    # Cluster via LLM
    clusters = _cluster_items(deduped, llm)
    return clusters


def _cluster_items(
    items: List[Dict[str, Any]], llm: LLMClient
) -> Dict[str, List[Dict[str, Any]]]:
    """Use LLM to assign items to topic clusters."""
    if not llm.available or len(items) <= 5:
        return {"General AI & Tech": items}

    # Build compact item list for LLM
    lines = []
    for i, item in enumerate(items[:60], 1):
        title = (item.get("title") or "")[:100]
        summary = (item.get("brief_summary") or "")[:150]
        lines.append(f"[{i}] {title}: {summary}")

    items_block = "\n".join(lines)

    prompt = (
        "Group these AI/tech newsletter items into 3-7 topical clusters. "
        "Choose descriptive cluster names (e.g., 'AI Agent Frameworks', "
        "'LLM Releases & Benchmarks', 'Developer Tools', 'AI Industry News').\n\n"
        f"<items>\n{items_block}\n</items>\n\n"
        "For each item, respond in this exact format:\n"
        "[number] CLUSTER: cluster name\n\n"
        "Respond ONLY with the numbered assignments, no preamble."
    )

    result = llm.invoke(
        prompt, tier="medium", max_tokens=1500,
        system_prompt="You are a content clustering assistant.",
        name="cluster_items",
    )

    if not result:
        return {"General AI & Tech": items}

    # Parse cluster assignments
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unassigned = []

    for line in result.strip().splitlines():
        match = re.match(r"\[(\d+)\]\s*CLUSTER:\s*(.+)", line.strip())
        if match:
            idx = int(match.group(1)) - 1
            cluster_name = match.group(2).strip()
            if 0 <= idx < len(items):
                items[idx]["cluster"] = cluster_name
                clusters[cluster_name].append(items[idx])

    # Assign unclustered items
    for i, item in enumerate(items):
        if "cluster" not in item:
            unassigned.append(item)

    if unassigned:
        # Add to largest cluster or create "Other"
        if clusters:
            largest = max(clusters, key=lambda k: len(clusters[k]))
            clusters[largest].extend(unassigned)
        else:
            clusters["General AI & Tech"] = unassigned

    # Sort items within each cluster by score
    for name in clusters:
        clusters[name].sort(
            key=lambda x: x.get("score_combined", 0), reverse=True
        )

    logger.info(
        f"Clustering: {len(clusters)} clusters — "
        + ", ".join(f"{k} ({len(v)})" for k, v in clusters.items())
    )
    return dict(clusters)


# ------------------------------------------------------------------
# Pass 3: Render + Publish
# ------------------------------------------------------------------

def render_and_publish(
    clusters: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
    total_emails: int,
    dry_run: bool,
):
    """Render per-cluster markdown, generate podcasts, publish to Obsidian."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    podcast_gen = PodcastGenerator(config.get("podcast", {}))

    obsidian_config = config.get("obsidian", {})
    api_url = obsidian_config.get("api_url", "http://localhost:27123")
    api_key = os.environ.get("OBSIDIAN_API_KEY", "")

    results = []

    for cluster_name, items in clusters.items():
        if len(items) < 2:
            logger.info(f"Skipping cluster '{cluster_name}' with only {len(items)} items")
            continue

        # Render markdown
        md = _render_cluster_markdown(cluster_name, items, total_emails, date_str)

        # Slug for filename
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", cluster_name).strip("-")[:40]
        filename = f"Newsletter-Digest-{slug}-{date_str}"

        # Save locally
        local_path = f"{filename}.md"
        with open(local_path, "w") as f:
            f.write(md)
        logger.info(f"Saved: {local_path}")

        # Podcast
        podcast_url = None
        if podcast_gen.enabled and not dry_run:
            source_urls = [item.get("url", "") for item in items[:10] if item.get("url")]
            podcast_url = podcast_gen.generate(md, now, source_urls)
            if podcast_url:
                logger.info(f"Podcast for '{cluster_name}': {podcast_url}")
                # Inject podcast URL into markdown
                podcast_section = (
                    f"\n## Podcast\n\n"
                    f"Listen to the audio overview: [{cluster_name} Digest]({podcast_url})\n"
                )
                md = md + podcast_section
                # Re-save with podcast URL
                with open(local_path, "w") as f:
                    f.write(md)

        # Obsidian publish
        if not dry_run and obsidian_config.get("enabled") and api_key:
            try:
                writer = ObsidianWriter(api_url, api_key, obsidian_config)
                folder = obsidian_config.get("briefing_folder", "Sources/Briefings")
                vault_path = (
                    f"{folder}/{now.strftime('%Y')}/{now.strftime('%m')}/{filename}.md"
                )
                frontmatter = (
                    f"---\n"
                    f"type: source/newsletter-digest\n"
                    f"source: batch-gmail-digest\n"
                    f"created: '{date_str}'\n"
                    f"cluster: '{cluster_name}'\n"
                    f"items-count: {len(items)}\n"
                    f"total-emails-processed: {total_emails}\n"
                )
                if podcast_url:
                    frontmatter += f"podcast-url: '{podcast_url}'\n"
                frontmatter += "---\n\n"

                writer._put_note(vault_path, frontmatter + md)
                logger.info(f"Published to Obsidian: {vault_path}")
            except Exception as e:
                logger.error(f"Obsidian publish failed for '{cluster_name}': {e}")

        results.append({
            "cluster": cluster_name,
            "items": len(items),
            "file": local_path,
            "podcast_url": podcast_url,
        })

    return results


def _render_cluster_markdown(
    cluster_name: str,
    items: List[Dict[str, Any]],
    total_emails: int,
    date_str: str,
) -> str:
    """Render a single cluster's digest as markdown."""
    lines = [
        f"# Newsletter Digest: {cluster_name}",
        f"*{date_str} — Curated from {total_emails} newsletter emails*\n",
    ]

    # Top highlights (up to 10)
    top_items = items[:10]
    lines.append("## Key Highlights\n")

    for i, item in enumerate(top_items, 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        score = item.get("score_combined", 0)
        summary = item.get("brief_summary", "")
        source = item.get("source", "Unknown")

        stars = "★" * int(score) + "☆" * (5 - int(score))

        lines.append(f"### {i}. [{title}]({url}) {stars}")
        if summary:
            lines.append(f"{summary}\n")
        lines.append(f"*via {source}*\n")

    # Remaining items (compact)
    remaining = items[10:]
    if remaining:
        lines.append("## More Items\n")
        for item in remaining:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            source = item.get("source", "")
            score = item.get("score_combined", 0)
            lines.append(f"- [{title}]({url}) ({source}, {score}/5)")
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Gmail Newsletter Digest")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--batch-size", type=int, default=50, help="Emails per IMAP batch")
    parser.add_argument("--dry-run", action="store_true", help="Skip mark-as-read and publishing")
    parser.add_argument("--all", action="store_true", help="Process ALL emails (not just unread)")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state (skip Pass 1)")
    parser.add_argument("--min-score", type=int, default=3, help="Minimum relevance score (1-5)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    config = load_config(args.config)

    # Setup LLM + intelligence
    llm_config = config.get("llm", {})
    llm = LLMClient(llm_config)
    intelligence = BriefingIntelligence(llm, config)

    start_time = time.time()

    # Pass 1: Fetch + Triage
    if args.resume:
        state = _load_state()
        if state:
            scored_items = state["scored_items"]
            total_emails = state["newsletter_count"]
            logger.info(
                f"Resumed from state: {total_emails} emails, {len(scored_items)} scored URLs"
            )
        else:
            logger.error("No saved state found. Run without --resume first.")
            return 2
    else:
        scored_items = fetch_and_triage(
            config, intelligence, args.batch_size, args.dry_run,
            search_all=args.all,
        )
        state = _load_state()
        total_emails = state["newsletter_count"] if state else 0

    if not scored_items:
        logger.warning("No scored items found. Nothing to process.")
        return 0

    # Pass 2: Filter + Cluster
    logger.info("=== Pass 2: Filter + Cluster ===")
    clusters = filter_and_cluster(scored_items, llm, min_score=args.min_score)

    if not clusters:
        logger.warning("No relevant content found after filtering.")
        return 0

    # Pass 3: Render + Publish
    logger.info("=== Pass 3: Render + Publish ===")
    results = render_and_publish(clusters, config, total_emails, args.dry_run)

    elapsed = time.time() - start_time
    logger.info(f"\n=== Batch Gmail Digest Complete in {elapsed:.1f}s ===")
    logger.info(f"Emails processed: {total_emails}")
    logger.info(f"Clusters generated: {len(results)}")
    for r in results:
        podcast = f" | Podcast: {r['podcast_url']}" if r.get("podcast_url") else ""
        logger.info(f"  {r['cluster']}: {r['items']} items → {r['file']}{podcast}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
