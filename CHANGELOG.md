# Changelog

All notable changes to Atlas Morning Briefing will be documented in this file.

## [0.2.0] - 2026-04-09

### Added
- **DeepXiv SDK integration** — ArXiv paper search now uses [DeepXiv](https://github.com/DeepXiv/deepxiv_sdk) semantic hybrid search instead of the raw ArXiv API
  - 200M+ papers indexed with daily sync
  - Semantic + keyword hybrid search for better relevance
  - Auto-enrichment: top 10 papers get TLDR summaries, keywords, and GitHub URLs from DeepXiv
  - Citation counts included in paper metadata
- Graceful fallback: if `deepxiv-sdk` is not installed, automatically falls back to legacy ArXiv API
- `CHANGELOG.md` added

### Changed
- `scripts/arxiv_scanner.py` rewritten with `DeepXivScanner` class and `ArxivScanner` legacy fallback
- `.gitignore` updated to exclude generated PDFs, logs, and venv

### Removed
- Test PDF (`Atlas-Briefing-2026.04.03.pdf`) removed from repo
- Benchmark run log removed from repo
- Cleaned leaked error logs from benchmark JSON files

### Fixed
- Benchmark JSON no longer contains raw error output with internal infrastructure details

## [0.1.0] - 2026-03-24

### Added
- Initial open-source release
- v0.1 single-pass briefing runner (`briefing_runner.py`)
- v0.2 parallel coordinator + workers architecture (`briefing_runner_v2.py`)
- ArXiv paper scanning with configurable topics
- Blog/RSS feed monitoring (20+ feeds)
- Stock watchlist via Finnhub API
- News aggregation via Brave Search API
- Amazon Bedrock LLM intelligence layer (relevance filtering, summarization, scoring)
- PDF generation with ReportLab
- Kindle email delivery
- Benchmark suite (`benchmark_v1_v2.py`)
- OpenClaw skill integration (`SKILL.md`)
- Example configuration (`config.yaml.example`)
