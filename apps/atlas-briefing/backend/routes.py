"""Atlas Briefing app backend — read-only dashboard data.

This app does NOT own the briefing pipeline or its cron. This backend only READS:
  - $ATLAS_HOME/status.json    (last-run counts/errors)
  - $ATLAS_HOME/config.yaml    (current topics)
  - artifacts under the gateway's artifact store (archived briefings, one per
    day, dirs named atlas-briefing-YYYY-MM-DD)
  - $ATLAS_RESEARCH_HOME       (optional research deep-dive inventory)

Paths come from environment variables so the app works on any machine; the
defaults match a checkout in the user's home directory. Set them in the
gateway's environment (e.g. ~/.kiro/crew/.env) and restart the gateway.

No network calls, no cron permission needed -- hence permissions.network=false
and permissions.cron=false in app.json.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kiro_crew.apps.route_registry import AppRoute


def _env_path(name: str, default: Path) -> Path:
    """Path from environment variable ``name`` (``~`` expanded), else ``default``."""
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


ATLAS_HOME = _env_path("ATLAS_HOME", Path.home() / "atlas-morning-briefing")
STATUS_FILE = ATLAS_HOME / "status.json"
CONFIG_FILE = ATLAS_HOME / "config.yaml"

# Research-papers repo (deep-dive / validated topic inventory). Read-only.
# The panel is simply empty when the directory does not exist.
RESEARCH_HOME = _env_path("ATLAS_RESEARCH_HOME", Path.home() / "workplace" / "research-papers")

# Artifact store on disk. Each archived briefing is a dir
# atlas-briefing-YYYY-MM-DD/ with meta.json + current.html (current.html holds
# the raw markdown despite the extension). Reading these directly avoids a
# wrong MCP import path -- the earlier bug that left the artifact-backed panels
# empty while the pipeline/research panels (no artifact dependency) worked.
_CREW_HOME = _env_path("KIROCREW_HOME", Path.home() / ".kiro" / "crew")
ARTIFACTS_HOME = _env_path("ATLAS_ARTIFACTS_HOME", _CREW_HOME / "artifacts")

# LLM-written weekly summary cache, refreshed by the atlas_weekly_summary.py
# script cron (top-tier model). Served in preference to the inline regex
# synthesis when present and fresh -- the LLM call is far too slow to run
# inside this 30s-polled request path.
WEEKLY_CACHE = Path(__file__).resolve().parent.parent / "data" / "weekly_summary.json"

# A project dir counts as "validated" if it carries one of these artifacts
# (a paper draft, recorded findings/results, or a camera-ready). Dirs that are
# plumbing/content rather than research deep-dives are excluded.
_VALIDATION_GLOBS = [
    "*paper*.md", "KEY-FINDINGS.md", "RESULTS.md", "*results*.json",
    "*camera-ready*.pdf", "VERDICT.md",
]
# Files searched (in order) for a real "# Title" heading, so the panel shows
# the actual paper/project name instead of a directory slug.
_TITLE_FILES = [
    "README.md", "paper.md", "paper_draft.md", "KEY-FINDINGS.md",
    "RESULTS.md", "VERDICT.md",
]
_NON_RESEARCH_DIRS = {
    "_archive", "content", "skills", "tools", "tasks", "briefings", "blog",
    "references", "inspiration", "patents", "proposals", "reading-notes",
    "morning-briefing", "morning-briefing-bot", "paperforge",
    "agentcorp-cli", "agentcorp-cli-commands", "kiro-forge", "paper-forge-engine",
    "accessclaw", "atlas-notes", "beyondswe", "caid-agentcore",
    "flashmemory-trn2", "kdd2026", "mini-d4rt",
}
# paper-forge/ is a parent holding MANY distinct paper reproductions -- walk
# its children as individual topics instead of collapsing to one "paper-forge"
# entry (that collapse was the main source of the "too strict / unreal" list).
_EXPAND_CHILDREN_OF = {"paper-forge"}
# Children of an expanded parent that are skill/reference scaffolding, not a
# paper deep-dive, even though they contain a markdown heading somewhere.
_EXCLUDE_EXPANDED_CHILDREN = {"agent-security"}


def _require_user(request):
    return request.get("user") is not None


def _load_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_topics() -> list[str]:
    try:
        import yaml

        cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
        return cfg.get("arxiv_topics", [])
    except Exception:
        return []


def _list_briefing_artifacts() -> list[dict]:
    """Archived daily briefings, newest first, read DIRECTLY from the artifact
    store on disk. Dirs named atlas-briefing-YYYY-MM-DD; the date comes from the
    slug. No MCP/auth needed -- plain filesystem."""
    out = []
    if not ARTIFACTS_HOME.exists():
        return out
    for d in ARTIFACTS_HOME.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"atlas-briefing-(\d{4})-(\d{2})-(\d{2})$", d.name)
        if not m:
            continue
        out.append({"slug": d.name, "date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}"})
    out.sort(key=lambda x: x["date"], reverse=True)
    return out


def _read_briefing_md(slug: str) -> str:
    """Raw markdown of an archived briefing (current.html holds markdown)."""
    f = ARTIFACTS_HOME / slug / "current.html"
    try:
        return f.read_text(encoding="utf-8")
    except Exception:
        return ""


def _extract_section(md: str, header: str) -> str:
    marker = f"## {header}"
    if marker not in md:
        return ""
    body = md.split(marker, 1)[1]
    if "\n## " in body:
        body = body.split("\n## ", 1)[0]
    return body.strip()


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
    "is", "are", "this", "that", "as", "by", "at", "from", "its", "be",
    "ai", "llm", "llms", "paper", "papers",
    "can", "new", "system", "systems", "using", "based", "via", "into",
    "towards", "toward", "how", "what", "why", "when", "does", "do", "did",
    "we", "our", "their", "it", "which", "has", "have", "had", "will",
    "about", "across", "study", "studies", "approach", "framework",
    "frameworks", "method", "methods", "analysis", "work", "research",
    "introduce", "introduces", "present", "presents", "propose", "proposes",
    "one", "two", "three", "first", "large", "small", "real", "world",
    "real-world", "key", "than", "best", "just", "not", "making", "makes",
    "make", "your", "you", "out", "up", "down", "more", "most", "who",
    "was", "were", "them", "then", "now", "next", "get", "got", "per",
    "use", "used", "uses", "after", "before", "over", "under", "between",
    "year", "years", "record", "move", "made", "mount", "going", "report",
    "reports", "latest", "amid", "could", "would", "should", "may", "might",
    "say", "says", "said", "tech", "bytes",
    # too generic for an AI-research feed to be a useful "hot topic"
    "agent", "agents", "model", "models", "data", "learning", "training",
}

# Canonicalize a handful of near-duplicates so e.g. "multi-agent" and
# "multiagent" or "gpt-" fragments don't fragment the counts.
_CANON = {
    "multiagent": "multi-agent",
    "gpt-": "gpt",
    "openai's": "openai",
}


def _hot_topics_from_titles(titles: list[str], top_n: int = 12) -> list[dict]:
    """Meaningful single-term frequency over recent paper/news titles.

    An earlier cut used bigrams, but two-word phrases almost never recur
    across different titles in this feed (max observed count was 2), so every
    'hot topic' tied at 2 and the ranking was noise. Single salient words
    (openai, autonomous, safety, anthropic, ...) DO recur with real spread,
    so they are the honest frequency signal. A broad stopword set drops the
    generic filler ('system', 'new', 'agent', 'model') that was the original
    complaint."""
    counts: Counter = Counter()
    for t in titles:
        seen_in_title = set()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9\-']{2,}", t.lower()):
            w = _CANON.get(w, w).strip("-'")
            if len(w) < 3 or w in _STOPWORDS:
                continue
            # Count each term at most once per title so a single wordy title
            # cannot dominate the ranking.
            if w in seen_in_title:
                continue
            seen_in_title.add(w)
            counts[w] += 1
    return [{"term": w, "count": c} for w, c in counts.most_common(top_n) if c >= 2] \
        or [{"term": w, "count": c} for w, c in counts.most_common(top_n)]


def _read_weekly_cache() -> dict | None:
    """The LLM-written weekly summary cached by the atlas_weekly_summary cron.
    Ignored when missing or older than ~2 days (a stale LLM blurb is worse
    than the always-current regex synthesis)."""
    try:
        data = json.loads(WEEKLY_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None
    gen = data.get("generated_at")
    if gen:
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(gen)
            if age > timedelta(days=2):
                return None
        except Exception:
            pass
    return data


async def handle_overview(request, ctx):
    if not _require_user(request):
        return _json(401, {"error": "unauthorized"})
    status = _load_status()
    topics = _load_topics()
    # Last 3 archived briefings, each with a deep link into the artifact viewer
    # so the header can offer "jump to the recent editions" shortcuts.
    recent = _list_briefing_artifacts()[:3]
    recent_briefings = [
        {"date": b["date"], "slug": b["slug"], "url": f"/artifacts/{b['slug']}"}
        for b in recent
    ]
    return _json(200, {
        "status": status,
        "topics": topics,
        "topic_count": len(topics),
        "recent_briefings": recent_briefings,
        "description": (
            "A read-only dashboard over the Atlas Morning Briefing pipeline: "
            "daily arXiv paper scans, AI/tech news, and market data, distilled "
            "into pipeline health, hot topics, deep-dive research tracking, a "
            "topic knowledge graph, and an LLM-written weekly executive summary."
        ),
    })


def _star_count(stars: str) -> int:
    return stars.count("★")


def _lead_theme_sentence(exec_summary: str) -> str:
    """The opening theme claim of a day's Executive Summary. The pipeline's
    own bolding style varies day to day (**Theme: text.**, **Key theme:**
    text, or no bold marker at all with the theme as the plain opening
    sentence) -- so rather than matching one bold pattern (which silently
    grabbed the wrong thing on the days that use a different style), strip
    all markdown emphasis first and then take the first sentence. That is
    the one structure every day's section actually shares."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", exec_summary.strip())
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Drop a leading "Theme:" / "Key theme:" label now that bolding is gone.
    text = re.sub(r"^(Key\s+)?Theme:\s*", "", text, flags=re.IGNORECASE)
    if not text:
        return ""
    # Split on sentence boundaries, but don't split on an abbreviation-like
    # single capital letter before the period (good enough for this prose).
    m = re.match(r"(.+?[.!?])(?:\s|$)", text)
    return m.group(1) if m else text[:200]


def _market_driver_lines(market_section: str) -> list[tuple[str, float, str]]:
    """(ticker, pct_change, one-line driver) rows from a day's market table."""
    rows = re.findall(
        r"\|\s*\*\*(\w+)\*\*\s*\|[^|]*\|\s*([+-]?[\d.]+)%\s*\|\s*([^|]+?)\s*\|",
        market_section,
    )
    return [(t, float(p), d.strip()) for t, p, d in rows]


async def handle_weekly(request, ctx):
    """7-day EXECUTIVE SUMMARY: synthesizes each day's own Executive Summary
    theme sentence, the highest-rated papers of the week, and the dominant
    market driver -- rather than a word-frequency guess at a 'theme'. PLUS
    hot-topics computed over a wider 30-day window for the Hot Topics panel."""
    if not _require_user(request):
        return _json(401, {"error": "unauthorized"})

    briefings = _list_briefing_artifacts()
    today = datetime.now(timezone.utc).date()
    cutoff_7 = (today - timedelta(days=7)).isoformat()
    cutoff_30 = (today - timedelta(days=30)).isoformat()
    recent = [b for b in briefings if b["date"] >= cutoff_7]          # summary window
    recent_30 = [b for b in briefings if b["date"] >= cutoff_30]      # hot-topics window

    daily_themes: list[str] = []
    top_papers: list[tuple[int, str, str]] = []  # (stars, title, date)
    market_rows: list[tuple[str, float, str, str]] = []  # (ticker, pct, driver, date)
    total_papers = 0
    total_news = 0
    summary_paper_titles: list[str] = []
    summary_news_titles: list[str] = []

    for b in recent:
        md = _read_briefing_md(b["slug"])
        exec_section = _extract_section(md, "Executive Summary")
        papers_section = _extract_section(md, "Top Papers")
        news_section = _extract_section(md, "AI & Tech News")
        market_section = _extract_section(md, "Financial Market Overview")

        theme = _lead_theme_sentence(exec_section)
        if theme:
            daily_themes.append(theme)

        for m in re.finditer(r"^### \d+\.\s*\[([^\]]+)\].*?(★+☆*)", papers_section, re.MULTILINE):
            title, stars = m.group(1), m.group(2)
            top_papers.append((_star_count(stars), title, b["date"]))

        for ticker, pct, driver in _market_driver_lines(market_section):
            market_rows.append((ticker, pct, driver, b["date"]))

        paper_titles = re.findall(r"^### \d+\.\s*\[([^\]]+)\]", papers_section, re.MULTILINE)
        news_titles = re.findall(r"^\*\*\[([^\]]+)\]", news_section, re.MULTILINE)
        summary_paper_titles.extend(paper_titles)
        summary_news_titles.extend(news_titles)
        total_papers += len(paper_titles)
        total_news += len(news_titles)

    # --- 30-day hot topics (wider window = more data points) ---
    hot_titles: list[str] = []
    hot_term_dates: dict[str, set] = {}
    hot_term_titles: dict[str, list[str]] = {}
    for b in recent_30:
        md = _read_briefing_md(b["slug"])
        papers_section = _extract_section(md, "Top Papers")
        news_section = _extract_section(md, "AI & Tech News")
        day_titles = (
            re.findall(r"^### \d+\.\s*\[([^\]]+)\]", papers_section, re.MULTILINE)
            + re.findall(r"^\*\*\[([^\]]+)\]", news_section, re.MULTILINE)
        )
        hot_titles.extend(day_titles)
        # Per-term provenance for the click-to-inspect detail panel.
        day_terms = {t["term"] for t in _hot_topics_from_titles(day_titles, top_n=50)}
        for term in day_terms:
            hot_term_dates.setdefault(term, set()).add(b["date"])
            for title in day_titles:
                if term.lower() in title.lower():
                    lst = hot_term_titles.setdefault(term, [])
                    if title not in lst and len(lst) < 6:
                        lst.append(title)
    hot_topics = _hot_topics_from_titles(hot_titles)
    for t in hot_topics:
        t["days"] = sorted(hot_term_dates.get(t["term"], []), reverse=True)
        t["titles"] = hot_term_titles.get(t["term"], [])

    if not recent:
        summary = "No archived briefings in the last 7 days yet."
        summary_meta = ""
    else:
        n_days = len(recent)
        date_range = f"{recent[-1]['date']} to {recent[0]['date']}"

        # Research narrative: the pipeline's own day-by-day theme sentences
        # are highly repetitive when one topic dominates the whole week (as
        # "agentic AI" does here) -- joining the first 3 verbatim just repeats
        # the same claim three ways. Instead: state the week's dominant theme
        # once, then add ONE distinct day's theme if a different angle showed
        # up (e.g. a day the dominant topic was something else).
        # Group by dominant subject rather than fuzzy text similarity: the
        # pipeline rephrases "agentic AI dominates" a different way nearly
        # every day, and plain word-overlap between those rephrasings is too
        # low (observed max ~0.29) to collapse them reliably. Instead, bucket
        # each day's theme by its leading subject phrase (first 2-3
        # significant words) and keep one representative per bucket -- this
        # reliably merges every "agentic ai ..." day into one line while
        # still surfacing a day about something else (e.g. "openai's safety
        # retreat") as its own distinct thread.
        def _subject_key(sentence: str) -> str:
            words = re.findall(r"[a-z]{2,}", sentence.lower())
            return " ".join(words[:2])

        buckets: dict[str, str] = {}
        order: list[str] = []
        for t in daily_themes:
            key = _subject_key(t)
            if key not in buckets:
                buckets[key] = t
                order.append(key)
        theme_lines = [buckets[k] for k in order]
        if not theme_lines:
            research_narrative = "No distinct research theme was called out this week."
        elif len(theme_lines) == 1 or len(daily_themes) <= 1:
            research_narrative = theme_lines[0]
        else:
            dominant = theme_lines[0]
            other = next((t for t in theme_lines[1:] if t != dominant), None)
            research_narrative = f"{dominant}" + (f" A distinct thread also surfaced: {other}" if other else "")

        # Top papers of the week by star rating.
        top_papers.sort(key=lambda x: x[0], reverse=True)
        best = top_papers[:3]
        papers_line = (
            "Highest-rated reads this week: " +
            "; ".join(f"\u201c{title}\u201d ({stars}\u2605)" for stars, title, _d in best) + "."
        ) if best else ""

        # Market narrative: the single biggest mover's own driver text, plus
        # the overall bias, instead of a bare up/down tally.
        if market_rows:
            biggest = max(market_rows, key=lambda r: abs(r[1]))
            ticker, pct, driver, date = biggest
            direction = "gained" if pct >= 0 else "fell"
            up = sum(1 for _t, p, _d, _dt in market_rows if p >= 0)
            down = len(market_rows) - up
            bias = "a broadly positive tape" if up >= down else "a broadly negative tape"
            market_narrative = (
                f"Markets saw {bias} across AMZN/GOOGL/TSLA/SPY this week; the sharpest move was "
                f"{ticker} ({direction} {abs(pct):.1f}% on {date}) on {driver.lower()}."
            )
        else:
            market_narrative = "No market data captured this week."

        summary = (
            f"{research_narrative}\n\n"
            f"{papers_line}\n\n"
            f"{market_narrative}"
        ).replace("\n\n\n", "\n\n").strip()
        summary_meta = (
            f"{date_range} · {n_days} editions · {total_papers} papers · {total_news} news items"
        )

    # Prefer the LLM-written summary (top-tier model, refreshed by the
    # atlas_weekly_summary.py cron) when it exists and is recent; the regex
    # synthesis above is the fallback for before the first cron run or if the
    # cache goes stale.
    summary_source = "synthesized"
    summary_model = None
    cached = _read_weekly_cache()
    if cached and cached.get("summary"):
        summary = cached["summary"].strip()
        summary_meta = (
            f"{cached.get('date_range', '')} · {len(recent)} editions · "
            f"{total_papers} papers · {total_news} news items"
        )
        summary_source = "llm"
        summary_model = cached.get("model")

    return _json(200, {
        "summary": summary,
        "summary_meta": summary_meta,
        "summary_source": summary_source,
        "summary_model": summary_model,
        "hot_topics": hot_topics,
        "hot_topics_window_days": 30,
        "window_days": 7,
        "briefings_found": len(recent),
        "briefings_found_30d": len(recent_30),
        "total_papers": total_papers,
        "total_news": total_news,
    })


async def handle_knowledge_graph(request, ctx):
    """Topic co-occurrence graph styled after KiroCrew's own native Knowledge
    Graph view: nodes colored by category (org/technology/concept/service)
    with a legend, sized by importance (# days mentioned), edges = two topics
    co-occurring in the SAME day's Top Papers + News. Uses the wider 30-day
    window so the graph has a comparable node count/density to the native
    view rather than the sparse 7-day cut."""
    if not _require_user(request):
        return _json(401, {"error": "unauthorized"})

    briefings = _list_briefing_artifacts()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
    recent = [b for b in briefings if b["date"] >= cutoff]

    per_day_terms: list[list[str]] = []
    term_weight: Counter = Counter()
    term_dates: dict[str, set] = {}
    term_titles: dict[str, list[str]] = {}
    for b in recent:
        md = _read_briefing_md(b["slug"])
        papers_section = _extract_section(md, "Top Papers")
        news_section = _extract_section(md, "AI & Tech News")
        titles = (
            re.findall(r"^### \d+\.\s*\[([^\]]+)\]", papers_section, re.MULTILINE)
            + re.findall(r"^\*\*\[([^\]]+)\]", news_section, re.MULTILINE)
        )
        day_topics = _hot_topics_from_titles(titles, top_n=6)
        terms = [t["term"] for t in day_topics]
        per_day_terms.append(terms)
        for t in terms:
            term_weight[t] += 1
            term_dates.setdefault(t, set()).add(b["date"])
            # Up to a few source titles that actually contain the term, for
            # the click-to-inspect detail panel.
            for title in titles:
                if t.lower() in title.lower() and title not in term_titles.setdefault(t, []):
                    if len(term_titles[t]) < 5:
                        term_titles[t].append(title)

    nodes = [
        {
            "id": f"term:{term}", "label": term, "weight": w,
            "category": _term_category(term),
            "days": sorted(term_dates.get(term, []), reverse=True),
            "titles": term_titles.get(term, []),
        }
        for term, w in term_weight.most_common(24)
    ]
    node_ids = {n["id"] for n in nodes}

    edge_weight: Counter = Counter()
    for terms in per_day_terms:
        present = [f"term:{t}" for t in terms if f"term:{t}" in node_ids]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                key = tuple(sorted((present[i], present[j])))
                edge_weight[key] += 1

    edges = [{"source": a, "target": b, "weight": w} for (a, b), w in edge_weight.items()]

    return _json(200, {
        "nodes": nodes,
        "edges": edges,
        "window_days": 30,
        "categories": sorted({n["category"] for n in nodes}),
    })


# Category legend matching KiroCrew's native Knowledge Graph view
# (org / technology / concept / service), applied heuristically to the
# hot-topic terms this app extracts from briefing titles.
_ORGS = {
    "openai", "anthropic", "google", "amazon", "aws", "deepmind", "meta",
    "microsoft", "nvidia", "tesla",
}
_TECH = {
    "claude", "gpt", "gemini", "bedrock", "trainium", "transformer",
    "transformers", "diffusion", "pytorch", "cuda", "kernel", "llm", "gpu",
}


def _term_category(term: str) -> str:
    """Best-effort bucket for a hot-topic term into the native KG's own
    category vocabulary, so this graph's legend/colors read the same way."""
    t = term.lower()
    if t in _ORGS:
        return "org"
    if t in _TECH:
        return "technology"
    if t.endswith(("ing", "tion", "ety", "ance", "ence")):
        return "concept"
    return "concept"


def _json(status: int, body: dict):
    from aiohttp import web
    return web.json_response(body, status=status)


def _real_title(child: Path) -> str:
    """The actual paper/project name from a '# Title' heading in one of the
    known title-bearing files, falling back to any markdown file's heading,
    then to the directory slug only when nothing titled exists."""
    for fname in _TITLE_FILES:
        f = child / fname
        if f.exists():
            try:
                m = re.search(r"^#\s+(.+)$", f.read_text(encoding="utf-8"), re.MULTILINE)
                if m:
                    title = re.sub(r"^(KEY[\s-]?FINDINGS:?\s*)", "", m.group(1), flags=re.IGNORECASE)
                    return title.strip()
            except Exception:
                continue
    for md in sorted(child.glob("*.md")):
        try:
            m = re.search(r"^#\s+(.+)$", md.read_text(encoding="utf-8"), re.MULTILINE)
            if m:
                return m.group(1).strip()
        except Exception:
            continue
    return child.name


def _has_real_content(child: Path) -> bool:
    """Excludes dirs with nothing a human would recognize as a deep-dive:
    skill/scaffolding-only dirs (YAML frontmatter, no real '# Title' heading)
    or empty placeholders. Requires either a known title file OR a markdown
    file with an actual '# ' heading -- a bare SKILL.md with no heading
    (e.g. paper-forge/agent-security) does not count."""
    if any((child / f).exists() for f in _TITLE_FILES):
        return True
    for md in child.glob("*.md"):
        try:
            if re.search(r"^#\s+\S", md.read_text(encoding="utf-8"), re.MULTILINE):
                return True
        except Exception:
            continue
    return False


def _dir_recency(child: Path) -> float:
    """Most-recent modification time anywhere in the dir -- a proxy for 'when
    did we last work on this'. Walks a bounded set of files so a big repo dir
    does not stall the request."""
    newest = child.stat().st_mtime
    try:
        for p in list(child.rglob("*"))[:400]:
            try:
                m = p.stat().st_mtime
                if m > newest:
                    newest = m
            except Exception:
                continue
    except Exception:
        pass
    return newest


def _activity(child: Path) -> int:
    """How much work the dir holds -- a frequency proxy (count of md/py/ipynb
    files). More files => a deeper dive."""
    try:
        return sum(1 for p in child.rglob("*") if p.suffix in (".md", ".py", ".ipynb"))
    except Exception:
        return 0


async def handle_research_topics(request, ctx):
    """The 5 deep-dives the user most recently/heavily worked on, shown by
    their REAL paper/project title (not the directory slug). Sorted by
    recency (dir mtime) then by activity (file count) so 'what we did last
    time' surfaces first. paper-forge/'s children are walked individually
    (each a distinct paper); empty scaffolding dirs are dropped."""
    if not _require_user(request):
        return _json(401, {"error": "unauthorized"})

    candidates: list[Path] = []
    if RESEARCH_HOME.exists():
        for child in sorted(RESEARCH_HOME.iterdir()):
            if not child.is_dir() or child.name.startswith(".") or child.name in _NON_RESEARCH_DIRS:
                continue
            if child.name in _EXPAND_CHILDREN_OF:
                for sub in sorted(child.iterdir()):
                    if not sub.is_dir() or sub.name in _EXCLUDE_EXPANDED_CHILDREN:
                        continue
                    if _has_real_content(sub):
                        candidates.append(sub)
                continue
            if _has_real_content(child):
                candidates.append(child)

    scored = [(c, _dir_recency(c), _activity(c)) for c in candidates]
    # Newest worked-on first; ties broken by activity (frequency).
    scored.sort(key=lambda s: (s[1], s[2]), reverse=True)
    top5 = scored[:5]

    topics = []
    for child, mtime, activity in top5:
        t = _classify(child)
        t["last_worked"] = datetime.fromtimestamp(mtime, timezone.utc).date().isoformat()
        t["activity"] = activity
        topics.append(t)

    return _json(200, {
        "topics": topics,
        "total": len(topics),
        "total_available": len(candidates),
        "validated": sum(1 for t in topics if t["validated"]),
        "repo": "JunjieTang-D1/research-papers",
    })


def _extract_summary(child: Path) -> str:
    """A short prose summary of the deep-dive: the first substantive paragraph
    (after the '# Title' heading, skipping metadata/badge lines) from the best
    available title-bearing file. Shown in the Details expansion."""
    search = _TITLE_FILES + ["DRAFT.md", "paper/DRAFT.md", "ABSTRACT.md"]
    for fname in search:
        f = child / fname
        if not f.exists():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        # Drop the leading H1 and anything before it, then find the first
        # real paragraph: a line of prose, not a heading, list, badge/table,
        # or key:value metadata.
        after_h1 = re.split(r"^#\s+.+$", text, maxsplit=1, flags=re.MULTILINE)
        body = after_h1[1] if len(after_h1) > 1 else text
        for para in re.split(r"\n\s*\n", body):
            line = para.strip()
            if not line:
                continue
            first = line.splitlines()[0].strip()
            if first.startswith(("#", "-", "*", "|", ">", "`", "![", "[!")):
                continue
            if re.match(r"^\*{0,2}[\w \-]{1,30}\*{0,2}\s*:", first):  # metadata key:
                continue
            # strip markdown emphasis/links for a clean sentence
            clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
            clean = re.sub(r"[*_`#]", "", clean).strip()
            clean = re.sub(r"\s+", " ", clean)
            if len(clean) >= 40:
                return clean[:320] + ("…" if len(clean) > 320 else "")
    return ""


def _classify(child: Path) -> dict:
    validated = False
    signal = ""
    for pat in _VALIDATION_GLOBS:
        hits = list(child.rglob(pat))
        if hits:
            validated = True
            signal = hits[0].relative_to(RESEARCH_HOME).as_posix()
            break
    return {
        "name": _real_title(child),
        "dir": child.relative_to(RESEARCH_HOME).as_posix(),
        "summary": _extract_summary(child),
        "validated": validated,
        "signal": signal,
    }


def register_routes(ctx) -> list:
    return [
        AppRoute("GET", "/overview", handle_overview),
        AppRoute("GET", "/weekly", handle_weekly),
        AppRoute("GET", "/knowledge-graph", handle_knowledge_graph),
        AppRoute("GET", "/research-topics", handle_research_topics),
    ]
