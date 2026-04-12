"""Tests for BriefingCoordinator (v2 runner)."""

from datetime import datetime

from scripts.briefing_runner import BriefingCoordinator


MINIMAL_CONFIG = {
    "arxiv_topics": ["AI"],
    "blog_feeds": [],
    "stocks": [],
    "news_queries": [],
    "llm": {"enabled": False},
    "podcast": {"enabled": False},
    "obsidian": {"enabled": False},
    "newsletter_source": {"enabled": False},
    "github_trending": {"enabled": False},
}


class TestCoordinatorInit:
    def test_basic_init(self):
        coord = BriefingCoordinator(MINIMAL_CONFIG, dry_run=True)
        assert coord.dry_run is True
        assert coord.run_date is None
        assert coord.status["intelligence_enabled"] is False

    def test_init_with_date(self):
        d = datetime(2026, 4, 7)
        coord = BriefingCoordinator(MINIMAL_CONFIG, run_date=d)
        assert coord.run_date == d


class TestExtractItems:
    def test_extracts_all_worker_types(self):
        coord = BriefingCoordinator(MINIMAL_CONFIG)
        findings = [
            {"worker": "papers_worker", "items": [{"title": "Paper A"}], "status": "success",
             "metadata": {"processing_time": 1, "token_count": 0, "items_found": 1, "items_kept": 1},
             "synthesis": "", "error": ""},
            {"worker": "blogs_worker", "items": [{"title": "Blog A"}], "status": "success",
             "metadata": {"processing_time": 1, "token_count": 0, "items_found": 1, "items_kept": 1},
             "synthesis": "", "error": ""},
            {"worker": "news_market_worker", "items": {"news": [{"title": "News A"}], "stocks": []},
             "status": "success",
             "metadata": {"processing_time": 1, "token_count": 0, "items_found": 1, "items_kept": 1},
             "synthesis": "", "error": ""},
            {"worker": "gmail_worker", "items": [{"title": "NL A"}], "status": "success",
             "metadata": {"processing_time": 1, "token_count": 0, "items_found": 1, "items_kept": 1},
             "synthesis": "", "error": ""},
            {"worker": "github_trending_worker", "items": [{"title": "owner/repo"}], "status": "success",
             "metadata": {"processing_time": 1, "token_count": 0, "items_found": 1, "items_kept": 1},
             "synthesis": "", "error": ""},
        ]
        papers, blogs, news, stocks, newsletters, gh = coord._extract_items(findings)
        assert len(papers) == 1
        assert len(blogs) == 1
        assert len(news) == 1
        assert len(newsletters) == 1
        assert len(gh) == 1


class TestDedup:
    def test_dedup_news_and_blogs(self):
        news = [
            {"title": "Same Title", "url": "https://blog.example.com/post"},
            {"title": "Unique News", "url": "https://news.example.com/article"},
        ]
        blogs = [
            {"title": "Same Title", "link": "https://blog.example.com/other"},
        ]
        deduped_news, _ = BriefingCoordinator.deduplicate_news_and_blogs(news, blogs)
        # "Same Title" removed (title match), blog domain also matches
        assert len(deduped_news) == 1
        assert deduped_news[0]["title"] == "Unique News"

    def test_dedup_similar_papers(self):
        papers = [
            {"title": "Attention Is All You Need For Language Understanding"},
            {"title": "Attention Is All You Need for Language Understanding"},
            {"title": "Totally Different Paper"},
        ]
        deduped = BriefingCoordinator.deduplicate_similar_papers(papers)
        assert len(deduped) == 2

    def test_dedup_against_previous(self):
        papers = [{"title": "Old Paper"}, {"title": "New Paper"}]
        previous = {"top_paper_titles": ["Old Paper"]}
        p, _, _ = BriefingCoordinator._dedup_against_previous(papers, [], [], previous)
        assert len(p) == 1
        assert p[0]["title"] == "New Paper"


class TestCommunityPicks:
    def test_build_community_picks(self):
        newsletters = [{"snippet": "Check https://example.com/article for more", "summary": ""}]
        github = [{"link": "https://github.com/owner/repo", "title": "owner/repo", "stars": "100", "language": "Python"}]
        picks = BriefingCoordinator._build_community_picks(newsletters, github)
        assert len(picks) == 2
        assert picks[0]["source_type"] == "newsletter"
        assert picks[1]["source_type"] == "github"


class TestFilename:
    def test_format_filename(self):
        coord = BriefingCoordinator(MINIMAL_CONFIG)
        coord.config["file_naming"] = "Personal-Briefing-{yyyy}.{mm}.{dd}"
        result = coord._format_filename(datetime(2026, 4, 8))
        assert result == "Personal-Briefing-2026.04.08"
