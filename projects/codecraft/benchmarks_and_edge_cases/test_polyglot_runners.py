"""Tests for Polyglot Multi-Language Test Runner Detection in Codecraft."""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from codecraft.test_runner import detect_project_runner, build_test_runner


def test_detect_python_project_default():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        cmd, runner_name = detect_project_runner(root)
        assert runner_name == "pytest"
        assert "pytest" in cmd


def test_detect_nodejs_project():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "package.json").write_text('{"scripts": {"test": "jest"}}\n')
        cmd, runner_name = detect_project_runner(root)
        assert runner_name == "npm test"
        assert "npm" in cmd[0]
        assert "test" in cmd[1]


def test_detect_rust_project():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "Cargo.toml").write_text('[package]\nname = "demo"\n')
        cmd, runner_name = detect_project_runner(root)
        assert runner_name == "cargo test"
        assert cmd == ["cargo", "test"]


def test_detect_go_project():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "go.mod").write_text("module demo\n")
        cmd, runner_name = detect_project_runner(root)
        assert runner_name == "go test"
        assert cmd == ["go", "test", "./..."]


def test_detect_custom_test_command():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cmd, runner_name = detect_project_runner(root, custom_cmd="vitest run --coverage")
        assert "vitest" in runner_name
        assert cmd == ["vitest", "run", "--coverage"]

