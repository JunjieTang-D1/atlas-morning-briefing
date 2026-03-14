---
name: morning-briefing
description: Generate a daily AI research + market + news briefing. Use when setting up automated morning briefings, research digests, or daily knowledge feeds. Covers arxiv papers, tech blogs, stock watchlist, industry news, and paper recommendations. Outputs Kindle PDF + channel message. Configurable topics, sources, stocks, and delivery schedule. Optionally uses Amazon Bedrock for AI-powered synthesis and summarization.
inputs:
  config_path:
    type: string
    required: true
    description: Path to configuration YAML file
  dry_run:
    type: boolean
    default: false
    description: Generate briefing without sending email
outputs:
  pdf_path:
    type: string
    description: Path to generated PDF file
  markdown_path:
    type: string
    description: Path to generated markdown file
  status:
    type: object
    description: Run status (papers_found, blogs_found, stocks_fetched, news_found, errors, elapsed_seconds)
triggers:
  schedule: "50 6 * * *"
  manual: "generate my morning briefing"
requires:
  python: ">=3.10"
  env:
    required:
      - FINNHUB_API_KEY
      - BRAVE_API_KEY
    optional:
      - GMAIL_USER
      - GMAIL_APP_PASSWORD
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_REGION
---

# Morning Briefing Skill

Generates a comprehensive morning briefing covering AI/ML papers (arxiv), tech blogs, stock watchlist, industry news, and paper recommendations for reproduction. Outputs as Kindle-optimized PDF with optional email delivery.

Enhanced with **Amazon Bedrock** for intelligent summarization, cross-section synthesis, and semantic paper scoring. Falls back gracefully to deterministic mode when Bedrock is unavailable.

## Prerequisites

### Required
- **Python 3.10+** (`python3 --version`)
- **pip** (`pip3 --version`)

### API Keys (all free tier)
| Service | Purpose | Sign Up | Free Tier |
|---------|---------|---------|-----------|
| **Finnhub** | Stock market data | [finnhub.io](https://finnhub.io/) | 60 calls/min |
| **Brave Search** | News aggregation | [brave.com/search/api](https://brave.com/search/api/) | 2000 queries/mo |
| **Gmail App Password** | Kindle email delivery | [myaccount.google.com](https://myaccount.google.com/apppasswords) | Free |

### Optional
- **AWS credentials** -- For Amazon Bedrock intelligence features (~$2.45/month)
- **Kindle Scribe/device** -- For PDF delivery via email. See `references/kindle_setup.md`
- **CJK fonts** -- For Chinese/Japanese/Korean support in PDFs:
  ```bash
  # Ubuntu/Debian
  sudo apt install fonts-noto-cjk
  # macOS
  brew install font-noto-sans-cjk
  ```

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt install python3-venv python3-pip

# macOS
brew install python3
```

## Features

### Deterministic (no LLM required)
- **ArXiv Paper Scanning**: Tracks new papers on configured topics
- **Blog Feed Monitoring**: Aggregates updates from RSS feeds
- **Stock Watchlist**: Fetches market data for configured tickers (Finnhub API)
- **News Aggregation**: Collects top AI/tech headlines (Brave Search API)
- **Paper Scoring**: Ranks papers by reproduction value (code availability, topic match, recency)
- **Cross-Section Deduplication**: Removes duplicate content between news and blogs
- **Kindle-Optimized PDF**: 6x8 inch format with CJK support
- **Email Delivery**: Send directly to Kindle via SMTP
- **Config Validation**: Catches configuration errors at startup
- **Status Reporting**: Generates status.json for monitoring

### Intelligence Layer (Amazon Bedrock, optional)
- **Topic Expansion**: Suggests related search queries using Nova Lite
- **Paper Summarization**: 1-2 sentence takeaways for each paper using Nova Pro
- **Semantic Scoring**: Relevance scoring using LLM understanding (beyond TF-IDF)
- **Stock-News Correlation**: Links stock movements to news drivers
- **Reproduction Assessment**: Evaluates compute, data, and feasibility for top papers
- **Cross-Section Synthesis**: Finds themes across papers, news, and blogs
- **Editorial Intro**: Opens briefing with today's key insight
- **Market Trend Summary**: 2-sentence market analysis with key drivers

## Setup

### 1. Install Dependencies

It's recommended to use a virtual environment:

```bash
cd atlas-morning-briefing
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or install as a package (enables `morning-briefing` CLI command):

```bash
pip install -e .
```

### 2. Configure API Keys

Set environment variables:

```bash
export FINNHUB_API_KEY="your_finnhub_key"
export BRAVE_API_KEY="your_brave_search_key"
export GMAIL_USER="your_email@gmail.com"
export GMAIL_APP_PASSWORD="your_app_password"
```

For Bedrock features, configure AWS credentials:

```bash
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_REGION="us-east-1"
```

Or use an IAM role / AWS profile (recommended for EC2/Lambda).

### 3. Configure Topics and Sources

Edit `config.yaml`:

```yaml
arxiv_topics:
  - "Agent Evaluation"
  - "Multi-Agent Systems"

blog_feeds:
  - name: "Anthropic"
    url: "https://www.anthropic.com/rss.xml"

stocks:
  - AMZN
  - GOOGL

news_queries:
  - "AI artificial intelligence"

kindle_email: "YOUR_NAME@kindle.com"
sender_email: "YOUR_EMAIL@gmail.com"

bedrock:
  enabled: true
  models:
    heavy: "us.anthropic.claude-sonnet-4-20250514-v1:0"
    medium: "amazon.nova-pro-v1:0"
    light: "amazon.nova-lite-v1:0"
```

See `references/config_guide.md` for full configuration options.

### 4. Set Up Kindle Email Delivery

See `references/kindle_setup.md` for instructions on configuring your Kindle email address.

## Usage

### Generate Briefing (Dry Run)

```bash
python3 scripts/briefing_runner.py --config config.yaml --dry-run
```

### Generate and Send to Kindle

```bash
python3 scripts/briefing_runner.py --config config.yaml
```

### Run Individual Scanners

Each scanner can be run independently:

```bash
python3 scripts/arxiv_scanner.py --config config.yaml --output papers.json
python3 scripts/blog_scanner.py --config config.yaml --output blogs.json
python3 scripts/stock_fetcher.py --config config.yaml --output stocks.json
python3 scripts/news_aggregator.py --config config.yaml --output news.json
```

### Score Papers for Reproduction

```bash
python3 scripts/paper_scorer.py --input papers.json --config config.yaml --output scored_papers.json
```

### Generate PDF Only

```bash
python3 scripts/pdf_generator.py --input briefing.md --output Atlas-Briefing.pdf --format kindle
```

## Run Status

After each run, a `status.json` file is generated:

```json
{
  "timestamp": "2026-03-06T06:50:12",
  "papers_found": 14,
  "blogs_found": 3,
  "stocks_fetched": 5,
  "news_found": 8,
  "intelligence_enabled": true,
  "errors": [],
  "pdf_generated": true,
  "email_sent": true,
  "elapsed_seconds": 7.2
}
```

## Scheduling

Set up a daily cron job:

```bash
crontab -e
```

Add:

```
0 7 * * * /path/to/atlas-morning-briefing/run_briefing.sh >> /path/to/atlas-morning-briefing/logs/briefing.log 2>&1
```

Create wrapper script (`run_briefing.sh`):

```bash
#!/bin/bash
cd /path/to/atlas-morning-briefing
source venv/bin/activate
source .env  # Load API keys
python3 scripts/briefing_runner.py --config config.yaml
```

## Cost Estimate

### Without Bedrock: $0.00/month

All external APIs (ArXiv, Finnhub, Brave, Gmail) have free tiers sufficient for daily use.

### With Bedrock (default: Claude Opus 4.6): ~$12-24/month

Default configuration uses Claude Opus 4.6 for all tiers for maximum quality.

**Estimated cost per run: ~$0.40-0.80 | Monthly (30 daily runs): ~$12-24**

For lower cost (~$0.08/run, ~$2.45/month), switch to tiered models in `config.yaml`:

```yaml
bedrock:
  models:
    heavy: "us.anthropic.claude-sonnet-4-20250514-v1:0"
    medium: "amazon.nova-pro-v1:0"
    light: "amazon.nova-lite-v1:0"
```

## File Naming

Configure in `config.yaml`:

```yaml
file_naming: "Atlas-Briefing-{yyyy}.{mm}.{dd}"
```

Available variables:
- `{yyyy}`: Year (4 digits)
- `{mm}`: Month (2 digits)
- `{dd}`: Day (2 digits)
- `{type}`: Briefing type (defaults to "Daily")

Example output: `Atlas-Briefing-2026.03.06.pdf`

## Paper Scoring Criteria

Papers are scored based on:

- **has_code** (weight: 5): Links to open source code repository
- **topic_match** (weight: 3): Cosine similarity to configured topics (TF-IDF)
- **semantic_score** (Bedrock): LLM-assessed relevance with explanation
- **recency** (weight: 2): Days since publication
- **citation_count** (weight: 1): Number of citations (if available)

Reproduction difficulty is estimated as S/M/L/XL based on:
- Dependencies complexity
- Dataset size
- Compute requirements

When Bedrock is enabled, reproduction assessment includes specific compute estimates and blocker identification.

## Troubleshooting

### No papers found
- Check arxiv_topics in config.yaml match arxiv categories
- Verify date range is not too narrow

### PDF generation fails
- Ensure fonts are installed for CJK support
- Check markdown formatting is valid

### Email delivery fails
- Verify GMAIL_USER and GMAIL_APP_PASSWORD are set
- Check sender_email matches GMAIL_USER
- Ensure Kindle email is whitelisted in Amazon account

### API rate limits
- Finnhub: Free tier allows 60 calls/minute
- Brave Search: Check your plan limits
- Rate limiting is built-in (0.5s delay between Finnhub calls, 1.0s between Brave calls)

### LLM call budget exhausted
- The intelligence layer has a per-run call budget (default: 20 calls)
- With all features enabled and many papers, this can be exceeded
- Increase via `bedrock.max_calls_per_run` in config.yaml:
  ```yaml
  bedrock:
    max_calls_per_run: 30
  ```
- Typical usage: ~10-15 calls with default settings

### Bedrock errors
- Verify AWS credentials are configured
- Check the model ID is available in your region
- Set `bedrock.enabled: false` to disable and run deterministically

### Config validation errors
- The runner validates config at startup and reports specific errors
- Check the error message for the invalid field and expected type

## Architecture

```
briefing_runner.py (orchestrator)
├── [Intelligence] Topic expansion (Bedrock Light)
├── arxiv_scanner.py → papers
├── blog_scanner.py → blogs
├── stock_fetcher.py → stocks
├── news_aggregator.py → news
├── [Dedup] Cross-section deduplication
├── [Intelligence] Paper summarization (Bedrock Medium)
├── [Intelligence] Semantic scoring (Bedrock Medium)
├── [Intelligence] Stock-news correlation (Bedrock Heavy)
├── paper_scorer.py → scored papers
├── [Intelligence] Reproduction assessment (Bedrock Medium)
├── [Intelligence] Cross-section synthesis (Bedrock Heavy)
├── [Generate markdown briefing with editorial content]
├── pdf_generator.py → Atlas-Briefing.pdf
├── kindle_sender.py → Email to Kindle
└── [Save status.json]
```

## Exit Codes

- **0**: Success
- **1**: Partial failure (some scanners failed but briefing generated)
- **2**: Total failure (unable to generate briefing or invalid config)

## References

- `references/config_guide.md`: Full configuration reference
- `references/kindle_setup.md`: Kindle email setup instructions
- `examples/sample-briefing.md`: Example generated briefing
