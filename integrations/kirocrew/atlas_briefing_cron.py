# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Kiro Crew script cron for the Atlas Morning Briefing (macOS, Linux, Windows).

Runs the pipeline from the repo's own virtualenv and reports the result to the
Kiro Crew dashboard. Kiro Crew refuses ``command=`` crons on Windows (they need a
POSIX ``/bin/sh``) and ``run_briefing.sh`` is bash-only, so this script is the
portable way to schedule the briefing.

Install: copy this file into ``~/.kiro/crew/crons/`` and register it, see
``references/kirocrew_setup.md``.

Cron message:
  * ``deliver`` -- full run (email / Kindle delivery)
  * anything else, including empty -- dry run (PDF only, nothing sent)

Repo location: ``ATLAS_HOME`` environment variable, else ``DEFAULT_ATLAS_HOME``.
API keys are read from ``$ATLAS_HOME/.env``, the same file ``run_briefing.sh`` uses.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from kiro_crew.cron_script import Report

# Edit this if you cloned the repo elsewhere and cannot set ATLAS_HOME.
DEFAULT_ATLAS_HOME = Path.home() / "atlas-morning-briefing"
RUN_TIMEOUT_S = 900
_STATUS_KEYS = ("papers_found", "blogs_found", "news_found", "stocks_fetched")


def atlas_home() -> Path:
    """Return the repo directory: ``ATLAS_HOME`` if set, else the default."""
    value = os.environ.get("ATLAS_HOME", "").strip()
    return Path(value).expanduser() if value else DEFAULT_ATLAS_HOME


def load_env(path: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``base`` (default: os.environ) overlaid with KEY=VALUE lines from ``path``.

    Blank lines and ``#`` comments are skipped, an unquoted value drops a trailing
    `` # comment``, and one pair of surrounding quotes is removed. A missing file
    leaves the environment unchanged.
    """
    env = dict(os.environ if base is None else base)
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()
        env[key.strip()] = value
    return env


def venv_python(home: Path) -> Path:
    """Path of the repo virtualenv's Python for the current OS."""
    if os.name == "nt":
        return home / "venv" / "Scripts" / "python.exe"
    return home / "venv" / "bin" / "python"


def build_command(home: Path, message: str | None) -> list[str]:
    """Pipeline command line; a dry run unless ``message`` is ``deliver``."""
    cmd = [str(venv_python(home)), "scripts/briefing_runner.py", "--config", "config.yaml"]
    if (message or "").strip().lower() != "deliver":
        cmd.append("--dry-run")
    return cmd


def summarize(home: Path) -> str:
    """One-line report: newest PDF in the repo root plus counts from status.json."""
    pdfs = sorted(home.glob("*.pdf"), key=lambda p: p.stat().st_mtime)
    status: dict = {}
    try:
        status = json.loads((home / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    counts = ", ".join(f"{key}={status.get(key)}" for key in _STATUS_KEYS)
    return f"Atlas briefing ready: {pdfs[-1] if pdfs else '(no PDF found)'}\n{counts}"


def run(ctx) -> None:
    """Kiro Crew entry point (``atlas_briefing_cron.py:run``)."""
    home = atlas_home()
    if not venv_python(home).is_file():
        raise Report(f"Atlas briefing FAILED: no virtualenv Python at {venv_python(home)}")
    proc = subprocess.run(
        build_command(home, ctx.message),
        cwd=home,
        env=load_env(home / ".env"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=RUN_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise Report("Atlas briefing FAILED:\n" + (proc.stderr or proc.stdout)[-1500:])
    raise Report(summarize(home))
