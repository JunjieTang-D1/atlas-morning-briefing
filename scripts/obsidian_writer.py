#!/usr/bin/env python3
"""
Obsidian vault writer.

Publishes briefing artifacts to an Obsidian vault via the Local REST API plugin,
turning daily ephemeral briefings into a compounding LLM Wiki.
"""

import logging
import re
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import yaml

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15


class ObsidianWriter:
    """Writes briefing artifacts to Obsidian vault via REST API."""

    def __init__(self, api_url: str, api_key: str, config: Dict[str, Any]):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.briefing_folder = config.get("briefing_folder", "Sources/Briefings")
        self.trending_threshold = config.get("trending_threshold", 3)

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------

    def _encode_path(self, path: str) -> str:
        """URL-encode each path segment individually."""
        return "/".join(
            urllib.parse.quote(segment, safe="")
            for segment in path.split("/")
        )

    def _get_note(self, path: str) -> Optional[str]:
        """GET /vault/{path}. Returns content or None on 404/error."""
        url = f"{self.api_url}/vault/{self._encode_path(path)}"
        try:
            resp = requests.get(
                url,
                headers={
                    "Accept": "text/markdown",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Obsidian GET {path} failed: {e}")
            return None

    def _put_note(self, path: str, content: str) -> bool:
        """PUT /vault/{path}. Returns True on success."""
        url = f"{self.api_url}/vault/{self._encode_path(path)}"
        try:
            resp = requests.put(
                url,
                data=content.encode("utf-8"),
                headers={
                    "Content-Type": "text/markdown",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning(f"Obsidian PUT {path} failed: {e}")
            return False

    def _note_exists(self, path: str) -> bool:
        """Check if a note exists via GET (returns bool)."""
        return self._get_note(path) is not None

    # ------------------------------------------------------------------
    # Frontmatter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_frontmatter(fields: Dict[str, Any]) -> str:
        """Render YAML frontmatter block."""
        return "---\n" + yaml.dump(fields, default_flow_style=False, allow_unicode=True, sort_keys=False) + "---\n"

    @staticmethod
    def _extract_frontmatter(content: str) -> Dict[str, Any]:
        """Parse YAML frontmatter from note content."""
        if not content.startswith("---"):
            return {}
        end = content.find("---", 3)
        if end == -1:
            return {}
        try:
            return yaml.safe_load(content[3:end]) or {}
        except yaml.YAMLError:
            return {}

    @staticmethod
    def _extract_body(content: str) -> str:
        """Strip frontmatter, return body only."""
        if not content.startswith("---"):
            return content
        end = content.find("---", 3)
        if end == -1:
            return content
        return content[end + 3:].lstrip("\n")

    # ------------------------------------------------------------------
    # Name normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vault_name(name: str) -> str:
        """Convert a display name to vault filename format.

        'Agent Memory Architectures' → 'Agent-Memory-Architectures'
        'claude-3.5' → 'Claude-3.5'
        """
        # Replace non-alphanumeric (except dots and hyphens) with hyphens
        cleaned = re.sub(r"[^a-zA-Z0-9.\-]+", "-", name.strip())
        # Title-case each hyphen-separated segment
        parts = cleaned.split("-")
        titled = "-".join(p.capitalize() if p else p for p in parts)
        # Strip leading/trailing hyphens
        return titled.strip("-")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_daily_briefing(
        self,
        markdown_content: str,
        date: datetime,
        status: Dict[str, Any],
        emerging_themes: List[str],
        top_papers: List[Dict[str, Any]],
        entity_mentions: List[Dict[str, Any]],
        podcast_url: Optional[str] = None,
    ) -> bool:
        """Write briefing to Sources/Briefings/YYYY/MM/Personal-Briefing-YYYY-MM-DD.md"""
        date_str = date.strftime("%Y-%m-%d")
        year = date.strftime("%Y")
        month = date.strftime("%m")
        filename = f"Personal-Briefing-{date_str}"
        path = f"{self.briefing_folder}/{year}/{month}/{filename}.md"

        # Build wikilinks to mentioned entities
        entity_names = [m["name"] for m in entity_mentions if m.get("name")]
        wiki_links = [f"[[{self._to_vault_name(n)}]]" for n in entity_names]

        fm_fields: Dict[str, Any] = {
            "type": "source/briefing",
            "source": "personal-morning-briefing",
            "created": date_str,
            "updated": date_str,
            "papers-found": status.get("papers_found", 0),
            "blogs-found": status.get("blogs_found", 0),
            "news-found": status.get("news_found", 0),
            "newsletters-found": status.get("newsletters_found", 0),
            "github-trending-found": status.get("github_trending_found", 0),
            "intelligence-enabled": status.get("intelligence_enabled", False),
            "emerging-themes": emerging_themes[:5] if emerging_themes else [],
            "top-papers": [p.get("title", "") for p in top_papers[:5]],
            "entity-mentions": entity_names,
            "wiki-links": wiki_links,
            "tags": ["source", "briefing", "morning-briefing"],
        }
        if podcast_url:
            fm_fields["podcast-url"] = podcast_url

        frontmatter = self._build_frontmatter(fm_fields)

        content = frontmatter + "\n" + markdown_content
        return self._put_note(path, content)

    def update_entity_timelines(
        self,
        entity_mentions: List[Dict[str, Any]],
        date: datetime,
        briefing_name: str,
    ) -> Dict[str, bool]:
        """Append dated timeline entries to existing entity pages."""
        results = {}
        date_str = date.strftime("%Y-%m-%d")
        heading = f"### {date_str}"

        for mention in entity_mentions:
            name = mention.get("name", "")
            if not name:
                continue

            vault_name = self._to_vault_name(name)
            path = f"Wiki/Entities/{vault_name}.md"

            try:
                existing = self._get_note(path)
                if existing is None:
                    # Entity page doesn't exist — skip, don't auto-create
                    logger.debug(f"Entity page not found, skipping: {path}")
                    continue

                # Duplicate-day guard
                if heading in existing:
                    logger.debug(f"Entity {vault_name} already has entry for {date_str}")
                    results[name] = True
                    continue

                # Build timeline entry
                count = mention.get("count", 0)
                titles = mention.get("example_titles", [])
                titles_str = ", ".join(f'"{t}"' for t in titles[:3])
                entry_lines = [
                    "",
                    heading,
                    f"- Mentioned {count} times: {titles_str}" if titles_str else f"- Mentioned {count} times",
                    f"- Source: [[{briefing_name}]]",
                    "",
                ]
                entry = "\n".join(entry_lines)

                # Append to existing content
                updated = existing.rstrip("\n") + "\n" + entry
                results[name] = self._put_note(path, updated)

            except Exception as e:
                logger.warning(f"Entity timeline update failed for {name}: {e}")
                results[name] = False

        return results

    def promote_concepts(
        self,
        trending_topics: Dict[str, Dict[str, Any]],
        emerging_themes: List[str],
        date: datetime,
        briefing_name: str,
    ) -> Dict[str, bool]:
        """Auto-promote trending topics to Wiki/Concepts/ when they hit threshold."""
        results = {}
        date_str = date.strftime("%Y-%m-%d")

        for topic_key, topic_data in trending_topics.items():
            count = topic_data.get("count", 0)
            if count < self.trending_threshold:
                continue

            vault_name = self._to_vault_name(topic_key)
            path = f"Wiki/Concepts/{vault_name}.md"

            try:
                existing = self._get_note(path)

                if existing is None:
                    # Create new concept page
                    first_seen = topic_data.get("first_seen", date_str)

                    # Find matching theme description for seeding
                    seed_description = ""
                    topic_words = set(topic_key.lower().replace("-", " ").split())
                    for theme in emerging_themes:
                        theme_words = set(theme.lower().split()[:6])
                        if topic_words & theme_words:
                            seed_description = theme
                            break

                    frontmatter = self._build_frontmatter({
                        "type": "wiki/concept",
                        "created": date_str,
                        "updated": date_str,
                        "first-detected": first_seen,
                        "detection-count": count,
                        "tags": ["wiki", "concept", "auto-promoted"],
                        "sources": [f"[[{briefing_name}]]"],
                    })

                    body = f"# {vault_name.replace('-', ' ')}\n\n"
                    if seed_description:
                        body += f"{seed_description}\n\n"
                    body += f"*Auto-promoted on {date_str} after appearing in {count} briefings.*\n"

                    results[topic_key] = self._put_note(path, frontmatter + "\n" + body)

                else:
                    # Append new occurrence to existing concept
                    heading = f"### {date_str}"
                    if heading in existing:
                        results[topic_key] = True
                        continue

                    # Update frontmatter detection-count and sources
                    fm = self._extract_frontmatter(existing)
                    body = self._extract_body(existing)

                    fm["updated"] = date_str
                    fm["detection-count"] = count
                    sources = fm.get("sources", [])
                    briefing_link = f"[[{briefing_name}]]"
                    if briefing_link not in sources:
                        sources.append(briefing_link)
                    fm["sources"] = sources

                    entry = f"\n{heading}\n- Trending for {count} days\n- Source: [[{briefing_name}]]\n"
                    updated_body = body.rstrip("\n") + "\n" + entry + "\n"

                    results[topic_key] = self._put_note(
                        path, self._build_frontmatter(fm) + "\n" + updated_body
                    )

            except Exception as e:
                logger.warning(f"Concept promotion failed for {topic_key}: {e}")
                results[topic_key] = False

        return results

    def write_weekly_synthesis(
        self,
        weekly_deep_dive: str,
        date: datetime,
        briefing_names: List[str],
    ) -> bool:
        """Write Wiki/Syntheses/Weekly-Digest-YYYY-WNN.md (Saturday only)."""
        if not weekly_deep_dive:
            return False

        iso_year, iso_week, _ = date.isocalendar()
        date_str = date.strftime("%Y-%m-%d")
        filename = f"Weekly-Digest-{iso_year}-W{iso_week:02d}"
        path = f"Wiki/Syntheses/{filename}.md"

        source_links = [f"[[{name}]]" for name in briefing_names]

        frontmatter = self._build_frontmatter({
            "type": "wiki/synthesis",
            "created": date_str,
            "updated": date_str,
            "week": f"{iso_year}-W{iso_week:02d}",
            "tags": ["wiki", "synthesis", "weekly-digest"],
            "sources": source_links,
        })

        content = frontmatter + "\n" + weekly_deep_dive + "\n"
        return self._put_note(path, content)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def publish(
        self,
        markdown_content: str,
        date: datetime,
        status: Dict[str, Any],
        emerging_themes: List[str],
        top_papers: List[Dict[str, Any]],
        entity_mentions: List[Dict[str, Any]],
        trending_topics: Dict[str, Dict[str, Any]],
        weekly_deep_dive: str,
        briefing_name: str,
        weekly_briefing_names: List[str],
        dry_run: bool = False,
        podcast_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish all briefing artifacts to Obsidian. Each operation isolated."""
        if dry_run:
            logger.info("Dry run: Skipping Obsidian publish")
            return {"dry_run": True}

        results: Dict[str, Any] = {}

        # 1. Daily briefing (always)
        try:
            results["briefing"] = self.write_daily_briefing(
                markdown_content, date, status, emerging_themes,
                top_papers, entity_mentions, podcast_url=podcast_url,
            )
        except Exception as e:
            logger.error(f"Obsidian briefing write failed: {e}")
            results["briefing"] = False

        # 2. Entity timelines (if mentions available)
        try:
            if entity_mentions:
                results["entities"] = self.update_entity_timelines(
                    entity_mentions, date, briefing_name,
                )
            else:
                results["entities"] = {}
        except Exception as e:
            logger.error(f"Obsidian entity update failed: {e}")
            results["entities"] = {}

        # 3. Concept promotion (if trending data available)
        try:
            if trending_topics:
                results["concepts"] = self.promote_concepts(
                    trending_topics, emerging_themes, date, briefing_name,
                )
            else:
                results["concepts"] = {}
        except Exception as e:
            logger.error(f"Obsidian concept promotion failed: {e}")
            results["concepts"] = {}

        # 4. Weekly synthesis (Saturday only, caller decides)
        try:
            if weekly_deep_dive:
                results["weekly_synthesis"] = self.write_weekly_synthesis(
                    weekly_deep_dive, date, weekly_briefing_names,
                )
            else:
                results["weekly_synthesis"] = None
        except Exception as e:
            logger.error(f"Obsidian weekly synthesis failed: {e}")
            results["weekly_synthesis"] = False

        return results
