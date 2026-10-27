# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for the Kiro Crew script cron (integrations/kirocrew/atlas_briefing_cron.py)."""

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

CRON_FILE = Path(__file__).resolve().parent.parent / "integrations" / "kirocrew" / "atlas_briefing_cron.py"


class _Report(Exception):
    """Stand-in for kiro_crew.cron_script.Report (kiro_crew is not a test dependency)."""


@pytest.fixture
def cron(monkeypatch):
    stub = types.ModuleType("kiro_crew.cron_script")
    stub.Report = _Report
    monkeypatch.setitem(sys.modules, "kiro_crew", types.ModuleType("kiro_crew"))
    monkeypatch.setitem(sys.modules, "kiro_crew.cron_script", stub)
    spec = importlib.util.spec_from_file_location("atlas_briefing_cron", CRON_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_repo(tmp_path: Path, cron) -> Path:
    py = cron.venv_python(tmp_path)
    py.parent.mkdir(parents=True)
    py.write_text("")
    return tmp_path


class TestLoadEnv:
    def test_parses_comments_quotes_and_exports(self, cron, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\n"
            "\n"
            "PLAIN=value\n"
            "TRAILING=value   # note\n"
            'DOUBLE="has # hash"\n'
            "SINGLE='single'\n"
            "export EXPORTED=yes\n"
            "EMPTY=\n"
            "EQUALS=a=b\n"
            "not a pair\n",
            encoding="utf-8",
        )
        env = cron.load_env(env_file, base={"KEEP": "1"})
        assert env == {
            "KEEP": "1",
            "PLAIN": "value",
            "TRAILING": "value",
            "DOUBLE": "has # hash",
            "SINGLE": "single",
            "EXPORTED": "yes",
            "EMPTY": "",
            "EQUALS": "a=b",
        }

    def test_missing_file_keeps_base(self, cron, tmp_path):
        assert cron.load_env(tmp_path / "missing.env", base={"A": "1"}) == {"A": "1"}


class TestCommand:
    def test_dry_run_by_default(self, cron, tmp_path):
        cmd = cron.build_command(tmp_path, "")
        assert cmd[1:] == ["scripts/briefing_runner.py", "--config", "config.yaml", "--dry-run"]

    def test_deliver_message_runs_full_pipeline(self, cron, tmp_path):
        assert "--dry-run" not in cron.build_command(tmp_path, " Deliver ")

    def test_none_message_is_dry_run(self, cron, tmp_path):
        assert cron.build_command(tmp_path, None)[-1] == "--dry-run"

    def test_venv_python_matches_os(self, cron, tmp_path, monkeypatch):
        monkeypatch.setattr(cron.os, "name", "nt")
        assert cron.venv_python(tmp_path) == tmp_path / "venv" / "Scripts" / "python.exe"
        monkeypatch.setattr(cron.os, "name", "posix")
        assert cron.venv_python(tmp_path) == tmp_path / "venv" / "bin" / "python"

    def test_atlas_home_env_override(self, cron, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_HOME", str(tmp_path))
        assert cron.atlas_home() == tmp_path
        monkeypatch.delenv("ATLAS_HOME")
        assert cron.atlas_home() == cron.DEFAULT_ATLAS_HOME


class TestRun:
    def test_reports_newest_pdf_and_counts(self, cron, tmp_path, monkeypatch):
        home = _fake_repo(tmp_path, cron)
        (home / ".env").write_text("BRAVE_API_KEY=abc\n", encoding="utf-8")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs, cmd=cmd)
            (home / "My-Briefing-2026.10.27.pdf").write_bytes(b"%PDF")
            (home / "status.json").write_text(json.dumps({"papers_found": 3, "blogs_found": 2}))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setenv("ATLAS_HOME", str(home))
        monkeypatch.setattr(cron.subprocess, "run", fake_run)
        with pytest.raises(_Report) as exc:
            cron.run(types.SimpleNamespace(message=""))
        assert "My-Briefing-2026.10.27.pdf" in str(exc.value)
        assert "papers_found=3" in str(exc.value)
        assert seen["cwd"] == home
        assert seen["env"]["BRAVE_API_KEY"] == "abc"
        assert seen["cmd"][-1] == "--dry-run"

    def test_reports_failure_output(self, cron, tmp_path, monkeypatch):
        home = _fake_repo(tmp_path, cron)
        monkeypatch.setenv("ATLAS_HOME", str(home))
        monkeypatch.setattr(
            cron.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"),
        )
        with pytest.raises(_Report, match="FAILED:\nboom"):
            cron.run(types.SimpleNamespace(message="deliver"))

    def test_missing_venv_is_reported(self, cron, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_HOME", str(tmp_path))
        with pytest.raises(_Report, match="no virtualenv Python"):
            cron.run(types.SimpleNamespace(message=""))
