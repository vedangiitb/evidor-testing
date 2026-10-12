"""Sandboxed polyglot test runner tool for Codecraft supporting Python, JS/TS, Go, Rust, Java, and custom commands."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from evidor import Tool, tool


def detect_project_runner(root: Path, custom_cmd: str | None = None) -> tuple[list[str], str]:
    """Detect the appropriate test command based on project manifest files or custom input.

    Returns:
        tuple of (command_list, runner_name)
    """
    if custom_cmd and custom_cmd.strip():
        parts = custom_cmd.strip().split()
        return parts, f"custom ({parts[0]})"

    # 1. JavaScript / TypeScript (Node.js)
    if (root / "package.json").exists():
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        return [npm_cmd, "test"], "npm test"

    # 2. Rust (Cargo)
    if (root / "Cargo.toml").exists():
        return ["cargo", "test"], "cargo test"

    # 3. Go
    if (root / "go.mod").exists():
        return ["go", "test", "./..."], "go test"

    # 4. Java (Maven or Gradle)
    if (root / "pom.xml").exists():
        mvn_cmd = "mvn.cmd" if os.name == "nt" else "mvn"
        return [mvn_cmd, "test"], "mvn test"
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        gradlew = "./gradlew.bat" if os.name == "nt" else "./gradlew"
        if (root / ("gradlew.bat" if os.name == "nt" else "gradlew")).exists():
            return [str(root / gradlew), "test"], "gradlew test"
        gradle_cmd = "gradle.cmd" if os.name == "nt" else "gradle"
        return [gradle_cmd, "test"], "gradle test"

    # 5. Make / C / C++
    if (root / "Makefile").exists() or (root / "makefile").exists():
        return ["make", "test"], "make test"

    # 6. Default to Python (pytest)
    return [sys.executable, "-m", "pytest", "-q", "--tb=short"], "pytest"


def execute_test_runner_direct(
    project_root: Path,
    test_target: str = "",
    extra_args: str = "",
    custom_cmd: str | None = None,
    timeout: float = 30.0,
    max_output_chars: int = 4000,
) -> dict[str, Any]:
    """Execute the project's test suite directly within project_root and return structured results."""
    root = Path(project_root).resolve(strict=True)
    base_cmd, runner_name = detect_project_runner(root, custom_cmd)

    cmd = list(base_cmd)

    if test_target.strip():
        # Prevent escaping project root via test_target
        target_path = (root / test_target.strip()).resolve(strict=False)
        try:
            target_path.relative_to(root)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid test target outside project root: {test_target}",
                "output": "",
                "exit_code": -1,
                "runner": runner_name,
            }
        cmd.append(str(target_path.relative_to(root)))

    if extra_args.strip():
        for arg in extra_args.split():
            # Allow common safe flags
            if arg.startswith("-") or not any(c in arg for c in ";|&$><`"):
                cmd.append(arg)

    env = os.environ.copy()
    # Python-specific path enhancement when running Python tests
    if "pytest" in runner_name.lower() or sys.executable in cmd:
        src_dir = root / "src"
        pythonpaths = [str(root)]
        if src_dir.exists():
            pythonpaths.insert(0, str(src_dir))
        env["PYTHONPATH"] = os.pathsep.join(pythonpaths + [env.get("PYTHONPATH", "")])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        combined_output = (proc.stdout + "\n" + proc.stderr).strip()
        if len(combined_output) > max_output_chars:
            half = max_output_chars // 2
            combined_output = (
                combined_output[:half]
                + f"\n... [Truncated {len(combined_output) - max_output_chars} characters] ...\n"
                + combined_output[-half:]
            )

        success = proc.returncode == 0
        summary_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        return {
            "success": success,
            "exit_code": proc.returncode,
            "runner": runner_name,
            "summary": summary_line or ("Tests passed" if success else "Tests failed"),
            "output": combined_output,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "exit_code": -1,
            "runner": runner_name,
            "summary": f"Timed out after {timeout} seconds",
            "output": f"Execution timed out after {timeout} seconds",
        }
    except Exception as exc:
        return {
            "success": False,
            "exit_code": -1,
            "runner": runner_name,
            "summary": f"Execution error: {exc}",
            "output": str(exc),
        }


# Backward-compatible alias
execute_pytest_direct = execute_test_runner_direct


def build_test_runner(
    project_root: Path,
    custom_cmd: str | None = None,
    default_timeout: float = 30.0,
) -> Tool:
    """Create an Evidor tool for running project test suites safely inside project_root."""
    root = Path(project_root).resolve(strict=True)
    _, runner_name = detect_project_runner(root, custom_cmd)

    @tool(
        name="run_tests",
        description=(
            f"Run project automated test suite ({runner_name}) inside the repository sandbox. "
            "Use this before refactoring to establish a baseline, and after refactoring "
            "to ensure no tests or functionality broke."
        ),
        timeout=default_timeout + 5.0,
    )
    def run_tests(test_target: str = "", extra_args: str = "") -> dict[str, Any]:
        """Run project tests in the sandbox.

        Args:
            test_target: Optional specific test file or directory path relative to project root.
            extra_args: Optional extra safe arguments for the test runner.

        Returns:
            Dictionary containing 'success' (bool), 'summary' (str), 'output' (str), and 'exit_code' (int).
        """
        return execute_test_runner_direct(
            project_root=root,
            test_target=test_target,
            extra_args=extra_args,
            custom_cmd=custom_cmd,
            timeout=default_timeout,
        )

    return run_tests
