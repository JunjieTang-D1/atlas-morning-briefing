FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Install cron and curl (for poetry installer)
RUN apt-get update && \
    apt-get install -y --no-install-recommends cron curl && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Install dependencies first (layer caching)
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

# Copy application code and install the project itself
COPY scripts/ scripts/
COPY README.md .
RUN poetry install --only main

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create logs directory
RUN mkdir -p /app/logs

ENTRYPOINT ["/entrypoint.sh"]
