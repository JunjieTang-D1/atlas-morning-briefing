# Configuration Guide

Complete reference for `config.yaml` settings.

## ArXiv Topics

List of topics to search on arxiv.org. Use natural language queries that match paper titles, abstracts, and categories.

```yaml
arxiv_topics:
  - "Agent Evaluation"
  - "Multi-Agent Systems"
  - "Tool Use LLM"
  - "Reinforcement Learning from Human Feedback"
  - "Chain of Thought"
  - "Prompt Engineering"
```

**Tips**:
- Use specific terms for better matches
- ArXiv searches across title, abstract, and metadata
- Papers are filtered by publication date (see `arxiv_days_back`)

## Blog Feeds

RSS/Atom feeds to monitor for new articles.

```yaml
blog_feeds:
  - name: "Anthropic"
    url: "https://www.anthropic.com/rss.xml"
  - name: "OpenAI"
    url: "https://openai.com/blog/rss.xml"
  - name: "DeepMind"
    url: "https://deepmind.google/blog/rss.xml"
```

**Fields**:
- `name`: Display name for the blog
- `url`: RSS/Atom feed URL

**Tips**:
- Validate feed URLs before adding
- Most blogs have `/feed`, `/rss`, or `/atom.xml` endpoints
- Check blog footer or source HTML for feed links

## Stock Watchlist

Stock symbols to track (uses Finnhub API).

```yaml
stocks:
  - AMZN
  - GOOGL
  - TSLA
  - NVDA
  - MSFT
  - META
```

**Format**: Use standard ticker symbols (e.g., NASDAQ, NYSE)

**API Key**: Requires `FINNHUB_API_KEY` environment variable
- Get free key at: https://finnhub.io/register
- Free tier: 60 calls/minute

## News Queries

Search queries for news aggregation (uses Brave Search API).

```yaml
news_queries:
  - "AI artificial intelligence"
  - "AWS Amazon cloud"
  - "machine learning breakthrough"
  - "LLM large language model"
  - "GPT transformer"
```

**API Key**: Requires `BRAVE_API_KEY` environment variable
- Get key at: https://api.search.brave.com/
- Check your plan limits

**Tips**:
- Use specific queries for targeted results
- Combine related terms in one query
- Results are filtered to past 24 hours

## Paper Scoring

Weights for scoring papers based on reproduction value.

```yaml
paper_scoring:
  has_code: 5          # Paper links to code repository
  topic_match: 3       # Matches configured topics
  recency: 2           # Recently published
  citation_count: 1    # Number of citations (if available)
```

**Criteria**:

1. **has_code** (boolean → 0 or weight)
   - Detects GitHub, GitLab, Hugging Face links
   - Searches title and abstract
   - Weight applied if code is available

2. **topic_match** (0-1 score)
   - Cosine similarity between paper and topics
   - Uses TF-IDF vectorization
   - Higher score = better match

3. **recency** (0-1 score)
   - Exponential decay: `score = e^(-days/30)`
   - Today = 1.0, 30 days ago ≈ 0.37
   - Favors recent papers

4. **citation_count** (currently unused)
   - Future enhancement
   - Would require citation API

**Reproduction Difficulty**: Automatically estimated as S/M/L/XL based on:
- Dataset size indicators
- Compute requirements
- Complexity keywords

## Delivery Settings

```yaml
kindle_email: "username@kindle.com"
sender_email: "agent@gmail.com"
output_format: "kindle"  # kindle | a4 | letter
```

**Fields**:
- `kindle_email`: Your Kindle email address (find in Amazon account)
- `sender_email`: Gmail address (must match `GMAIL_USER`)
- `output_format`: PDF page size
  - `kindle`: 6x8 inch (Kindle Scribe optimized)
  - `a4`: 8.27x11.69 inch
  - `letter`: 8.5x11 inch

**Environment Variables Required**:
- `GMAIL_USER`: Your Gmail address
- `GMAIL_APP_PASSWORD`: Gmail app password (not regular password)

See `kindle_setup.md` for detailed Kindle configuration.

## File Naming

Pattern for output filenames.

```yaml
file_naming: "Personal-Briefing-{type}-{yyyy}.{mm}.{dd}"
```

**Variables**:
- `{type}`: Briefing type (e.g., "Daily", "Weekly")
- `{yyyy}`: Year (4 digits)
- `{mm}`: Month (2 digits, zero-padded)
- `{dd}`: Day (2 digits, zero-padded)

**Examples**:
- `Personal-Briefing-{type}-{yyyy}.{mm}.{dd}` → `Personal-Briefing-Daily-2026.03.06`
- `Briefing-{yyyy}-{mm}-{dd}` → `Briefing-2026-03-06`
- `Daily-Digest-{mm}{dd}` → `Daily-Digest-0306`

## Paper Recommendations

```yaml
num_paper_picks: 3
```

Number of top-scored papers to highlight in "Papers for Reproduction" section.

## Date Ranges

```yaml
arxiv_days_back: 7
```

Number of days to look back when searching ArXiv. Papers older than this are filtered out.

**Recommendations**:
- Daily briefings: 1-2 days
- Weekly briefings: 7 days
- Monthly digests: 30 days

## Content Limits

```yaml
max_papers: 20
max_blogs: 10
max_news: 15
```

Maximum items to fetch per section.

**Tips**:
- Higher limits = longer generation time
- ArXiv API has rate limits
- Adjust based on your reading capacity

## PDF Settings

```yaml
pdf:
  font_size: 10
  line_spacing: 1.5
  include_toc: true
  include_emoji: false
```

**Fields**:
- `font_size`: Base font size in points (8-14 recommended)
- `line_spacing`: Line spacing multiplier (1.0-2.0)
- `include_toc`: Table of contents (not yet implemented)
- `include_emoji`: Strip emoji from output (recommended: false)

**Kindle Optimization**:
- Smaller font sizes work better on Kindle
- 1.5 line spacing improves readability
- 6x8 inch page size fits Kindle Scribe perfectly

## Logging

```yaml
log_level: "INFO"  # DEBUG | INFO | WARNING | ERROR
```

**Levels**:
- `DEBUG`: Verbose output (API calls, parsing details)
- `INFO`: Standard output (progress, summary)
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors

## Complete Example

```yaml
# Morning Briefing Configuration

arxiv_topics:
  - "Agent Evaluation"
  - "Multi-Agent Systems"
  - "Tool Use LLM"

blog_feeds:
  - name: "Anthropic"
    url: "https://www.anthropic.com/rss.xml"
  - name: "OpenAI"
    url: "https://openai.com/blog/rss.xml"

stocks:
  - AMZN
  - GOOGL
  - NVDA

news_queries:
  - "AI artificial intelligence"
  - "machine learning breakthrough"

paper_scoring:
  has_code: 5
  topic_match: 3
  recency: 2
  citation_count: 1

kindle_email: "username@kindle.com"
sender_email: "agent@gmail.com"
output_format: "kindle"

file_naming: "Personal-Briefing-{type}-{yyyy}.{mm}.{dd}"

num_paper_picks: 3
arxiv_days_back: 7

max_papers: 20
max_blogs: 10
max_news: 15

pdf:
  font_size: 10
  line_spacing: 1.5
  include_toc: true
  include_emoji: false

log_level: "INFO"
```

## Amazon Bedrock Configuration

```yaml
bedrock:
  enabled: true
  region: "us-east-1"
  temperature: 0.3
  max_tokens: 2048
  models:
    heavy: "us.anthropic.claude-sonnet-4-20250514-v1:0"
    medium: "amazon.nova-pro-v1:0"
    light: "amazon.nova-lite-v1:0"
```

**Fields**:
- `enabled`: Set to `false` to disable all LLM features (zero cost, deterministic)
- `region`: AWS region for Bedrock API calls
- `temperature`: Sampling temperature (0.0-1.0, lower = more deterministic)
- `max_tokens`: Maximum tokens per model response
- `models`: Model IDs for each tier

**Tiers**:
- `heavy`: Used for cross-section synthesis, editorial content, stock-news correlation. Needs strong reasoning.
- `medium`: Used for paper summaries, semantic scoring, reproduction assessment. Needs accuracy.
- `light`: Used for topic expansion. Simple brainstorming task.

**Cost per run** (default: Claude Opus 4.6 on all tiers):
- **Estimated: ~$0.40-0.80/run, ~$12-24/month**

For lower cost, switch to tiered models:
- Heavy (Claude Sonnet 4): ~$0.06
- Medium (Nova Pro): ~$0.03
- Light (Nova Lite): ~$0.001
- **Tiered total: ~$0.08/run, ~$2.45/month**

**Supported models** (any Bedrock model can be used):

| Model | ID | Recommended Tier |
|---|---|---|
| Claude Sonnet 4 | `us.anthropic.claude-sonnet-4-20250514-v1:0` | heavy |
| Nova Pro | `amazon.nova-pro-v1:0` | medium |
| Nova Lite | `amazon.nova-lite-v1:0` | light |
| Kimi K2.5 | `moonshotai.kimi-k2.5` | heavy/medium |
| GLM 4.7 | `zai.glm-4.7` | medium |
| DeepSeek V3.2 | `deepseek.v3.2` | heavy/medium |

## Environment Variables

Required environment variables for API access:

```bash
# Finnhub API (stock data)
export FINNHUB_API_KEY="your_finnhub_key"

# Brave Search API (news)
export BRAVE_API_KEY="your_brave_key"

# Gmail SMTP (Kindle delivery)
export GMAIL_USER="your_email@gmail.com"
export GMAIL_APP_PASSWORD="your_app_password"

# AWS credentials (for Bedrock -- optional)
# Or use IAM roles / AWS profiles instead
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_REGION="us-east-1"
```

**Security**:
- Never commit `.env` files with credentials
- Use app-specific passwords for Gmail
- Keep API keys private
- Rotate keys periodically
- Prefer IAM roles over access keys when running on AWS infrastructure
