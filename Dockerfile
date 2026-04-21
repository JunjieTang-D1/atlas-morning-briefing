FROM python:3.12-slim

WORKDIR /app

# Install n8t-scraper (private shared package, checked out by CI before docker build)
COPY n8t-scraper/ /tmp/n8t-scraper/
RUN pip install --no-cache-dir /tmp/n8t-scraper && rm -rf /tmp/n8t-scraper

# Install remaining dependencies (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ scripts/
COPY config.yaml.example config.yaml.example

# Config and state injected at runtime via volume mounts
ENTRYPOINT ["python3", "/app/scripts/briefing_runner.py", "--config", "/config/config.yaml"]
