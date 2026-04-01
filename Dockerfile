FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ scripts/
COPY config.yaml.example config.yaml.example

# Config and state injected at runtime via volume mounts
ENTRYPOINT ["python3", "/app/scripts/briefing_runner.py", "--config", "/config/config.yaml"]
