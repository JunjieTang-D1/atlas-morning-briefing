# Contributing to Personal Morning Briefing

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork and create a branch:
   ```bash
   git clone https://github.com/your-username/atlas-morning-briefing.git
   cd atlas-morning-briefing
   git checkout -b feature/your-feature
   ```
3. Set up the development environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings for public functions and classes

### Project Structure

- `scripts/` -- Pipeline modules (each scanner is self-contained)
- `tests/` -- Unit tests (mirror the scripts/ structure)
- `references/` -- User-facing documentation

## Submitting Changes

1. Write tests for new functionality
2. Run the full test suite: `pytest tests/ -v`
3. Test a dry-run: `python3 scripts/briefing_runner.py --config config.yaml --dry-run`
4. Commit with a clear message describing the change
5. Open a pull request against `main`

## Adding a New Scanner

1. Create `scripts/your_scanner.py` following the pattern of existing scanners
2. Add a `YourScanner` class with `__init__` and a main scan method
3. Add standalone CLI support with `argparse`
4. Wire it into `briefing_runner.py`
5. Add tests in `tests/test_your_scanner.py`
6. Update `config.yaml` with any new config keys
7. Update `references/config_guide.md`

## Adding a New Intelligence Feature

1. Add the method to `scripts/intelligence.py`
2. Choose the appropriate Bedrock tier (light/medium/heavy)
3. Handle the `not self.available` case (graceful fallback)
4. Wire it into `briefing_runner.py` in the intelligence section
5. Add the enriched data to `generate_markdown_briefing`
6. Add tests (mock the Bedrock client)

## Reporting Issues

Please include:
- Python version (`python3 --version`)
- OS and version
- The error message or unexpected behavior
- Your `config.yaml` (with API keys redacted)
- Contents of `status.json` if available

## Security

- Never commit API keys or credentials
- Use environment variables for all secrets
- Report security vulnerabilities privately (do not open a public issue)
