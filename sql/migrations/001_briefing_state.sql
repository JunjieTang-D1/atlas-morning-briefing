-- Migration: 001_briefing_state
-- Stores cross-day deduplication state for the atlas morning briefing.
-- Replaces the PVC-backed .personal-state.json file.
-- Apply via Supabase dashboard or MCP execute_sql.

create table if not exists briefing_state (
  date              date        primary key,
  top_paper_titles  text[]      not null default '{}',
  top_blog_titles   text[]      not null default '{}',
  top_news_titles   text[]      not null default '{}',
  top_github_titles text[]      not null default '{}',
  stock_closes      jsonb       not null default '{}',
  emerging_themes   text[]      not null default '{}',
  trending_topics   jsonb       not null default '{}',
  weekly_items      jsonb       not null default '[]',
  email_sent_date   date,
  created_at        timestamptz not null default now()
);

-- Only the service role (used by the atlas pod) can read/write rows.
alter table briefing_state enable row level security;
