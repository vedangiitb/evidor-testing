"""High-level Codecraft orchestration engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Sequence

from evidor import (
    Agent,
    InMemorySink,
    ModelProvider,
    RetryConfig,
    TelemetrySink,
)
from evidor.telemetry import (
    ErrorEvent,
    LLMCallEndEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

from .agent import create_codecraft_agent
from .cloner import clone_or_load_repo
from .test_runner import execute_pytest_direct

logger = logging.getLogger("codecraft.engine")


@dataclass
class RefactorReport:
    """Structured report returned after a Codecraft refactoring run."""

    project_root: Path
    baseline_tests: dict[str, Any]
    verification_tests: dict[str, Any]
    agent_summary: str
    modified_files: list[str] = field(default_factory=list)
    total_tool_calls: int = 0
    tool_call_breakdown: dict[str, int] = field(default_factory=dict)
    total_llm_calls: int = 0
    errors_encountered: int = 0
    success: bool = False
    phoenix_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "baseline_passed": self.baseline_tests.get("success", False),
            "baseline_summary": self.baseline_tests.get("summary", ""),
            "verification_passed": self.verification_tests.get("success", False),
            "verification_summary": self.verification_tests.get("summary", ""),
            "agent_summary": self.agent_summary,
            "modified_files": self.modified_files,
            "total_tool_calls": self.total_tool_calls,
            "tool_call_breakdown": self.tool_call_breakdown,
            "total_llm_calls": self.total_llm_calls,
            "errors_encountered": self.errors_encountered,
            "success": self.success,
            "phoenix_url": self.phoenix_url,
        }


class Codecraft:
    """End-to-end Codecraft refactoring orchestrator with Arize Phoenix telemetry."""

    def __init__(
        self,
        source: str | Path,
        *,
        sandbox_dir: str | Path | None = None,
        provider: ModelProvider | None = None,
        model_name: str = "gemini-2.5-flash",
        custom_test_cmd: str | None = None,
        enable_phoenix: bool = False,
        phoenix_endpoint: str | None = None,
        phoenix_project_name: str = "codecraft",
        max_file_bytes: int = 100_000,
        max_tool_iterations: int = 25,
        tool_timeout: float = 35.0,
        telemetry_sinks: Sequence[TelemetrySink] | None = None,
    ) -> None:
        self.source = source
        self.sandbox_dir = sandbox_dir
        self.provider = provider
        self.model_name = model_name
        self.custom_test_cmd = custom_test_cmd
        self.max_file_bytes = max_file_bytes
        self.max_tool_iterations = max_tool_iterations
        self.tool_timeout = tool_timeout

        self._in_memory_sink = InMemorySink()
        sinks: list[TelemetrySink] = [self._in_memory_sink]
        if telemetry_sinks:
            sinks.extend(telemetry_sinks)

        # Arize Phoenix telemetry integration
        self.enable_phoenix = enable_phoenix or (
            os.getenv("PHOENIX_ENABLE", "").lower() in ("1", "true", "yes")
        )
        self.phoenix_endpoint = phoenix_endpoint or os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
        )
        self.phoenix_project_name = phoenix_project_name or os.getenv(
            "PHOENIX_PROJECT_NAME", "codecraft"
        )
        self.phoenix_ui_url: str | None = None

        if self.enable_phoenix:
            try:
                from evidor import PhoenixSink

                self._phoenix_sink = PhoenixSink(
                    endpoint=self.phoenix_endpoint,
                    project_name=self.phoenix_project_name,
                )
                sinks.append(self._phoenix_sink)
                # Compute base dashboard URL (strip /v1/traces)
                base = self.phoenix_endpoint.replace("/v1/traces", "").rstrip("/")
                self.phoenix_ui_url = f"{base}/projects/{self.phoenix_project_name}" if "localhost" in base else base
                logger.info(f"Arize Phoenix telemetry enabled. Dashboard: {self.phoenix_ui_url}")
            except Exception as exc:
                logger.warning(f"Failed to initialize Arize Phoenix telemetry: {exc}")
                self.enable_phoenix = False

        self._sinks = sinks
        self.project_root: Path | None = None
        self.agent: Agent | None = None
        self._initial_snapshots: dict[str, bytes] = {}

    def setup(self) -> Path:
        """Clone or copy the source repository into the sandbox and instantiate the agent."""
        if self.project_root is None:
            self.project_root = clone_or_load_repo(self.source, self.sandbox_dir)
            self._take_snapshot()
            self.agent = create_codecraft_agent(
                project_root=self.project_root,
                custom_test_cmd=self.custom_test_cmd,
                provider=self.provider,
                model_name=self.model_name,
                max_file_bytes=self.max_file_bytes,
                max_tool_iterations=self.max_tool_iterations,
                tool_timeout=self.tool_timeout,
                telemetry=self._sinks,
            )
        return self.project_root

    def _take_snapshot(self) -> None:
        """Snapshot file contents to accurately detect modified files post-refactor."""
        if self.project_root is None:
            return
        self._initial_snapshots = {}
        for path in self.project_root.rglob("*"):
            if path.is_file() and not any(part.startswith((".", "__")) for part in path.parts):
                try:
                    rel = path.relative_to(self.project_root).as_posix()
                    self._initial_snapshots[rel] = path.read_bytes()
                except Exception:
                    continue

    def _get_modified_files(self) -> list[str]:
        """Compare current files with snapshot."""
        if self.project_root is None:
            return []
        modified = []
        current_files = set()
        for path in self.project_root.rglob("*"):
            if path.is_file() and not any(part.startswith((".", "__")) for part in path.parts):
                try:
                    rel = path.relative_to(self.project_root).as_posix()
                    current_files.add(rel)
                    content = path.read_bytes()
                    if rel not in self._initial_snapshots or self._initial_snapshots[rel] != content:
                        modified.append(rel)
                except Exception:
                    continue
        for rel in self._initial_snapshots:
            if rel not in current_files:
                modified.append(f"{rel} (deleted)")
        return sorted(modified)

    def run_baseline_tests(self) -> dict[str, Any]:
        """Run tests directly before starting the refactor."""
        if self.project_root is None:
            self.setup()
        assert self.project_root is not None
        return execute_pytest_direct(
            self.project_root,
            custom_cmd=self.custom_test_cmd,
            timeout=self.tool_timeout,
        )

    def run_verification_tests(self) -> dict[str, Any]:
        """Run tests directly after refactoring."""
        assert self.project_root is not None
        return execute_pytest_direct(
            self.project_root,
            custom_cmd=self.custom_test_cmd,
            timeout=self.tool_timeout,
        )

    def refactor(self, instructions: str | None = None) -> RefactorReport:
        """Execute the full refactoring loop."""
        root = self.setup()
        assert self.agent is not None

        logger.info(f"Running baseline tests on {root}...")
        baseline = self.run_baseline_tests()

        prompt = instructions or (
            "Explore the project structure, run the automated tests to establish a baseline, "
            "inspect messy source code for code smells, antipatterns, and poor typing, "
            "and refactor the code to improve quality, structure, and readability while "
            "ensuring all tests pass with zero regressions. Remember that write_file replaces "
            "the entire file content so supply complete files. Report your changes when complete."
        )

        logger.info("Dispatching refactoring prompt to Codecraft agent...")
        response = self.agent.send(prompt)

        logger.info("Running post-refactoring verification tests...")
        verification = self.run_verification_tests()
        modified_files = self._get_modified_files()

        # Telemetry metrics
        tool_ends = self._in_memory_sink.filter(ToolCallEndEvent)
        breakdown: dict[str, int] = {}
        for te in tool_ends:
            breakdown[te.tool_name] = breakdown.get(te.tool_name, 0) + 1

        llm_ends = self._in_memory_sink.filter(LLMCallEndEvent)
        errors = self._in_memory_sink.filter(ErrorEvent)

        success = verification.get("success", False)

        return RefactorReport(
            project_root=root,
            baseline_tests=baseline,
            verification_tests=verification,
            agent_summary=response.text,
            modified_files=modified_files,
            total_tool_calls=len(tool_ends),
            tool_call_breakdown=breakdown,
            total_llm_calls=len(llm_ends),
            errors_encountered=len(errors),
            success=success,
            phoenix_url=self.phoenix_ui_url,
        )

    def close(self) -> None:
        """Close agent and telemetry resources."""
        if self.agent is not None:
            self.agent.close()

    def __enter__(self) -> "Codecraft":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

