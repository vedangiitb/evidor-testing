"""End-to-End Integration Tests for Codecraft on Messy Codebase.

Includes:
1. Deterministic zero-cost offline test using a Mock Provider:
   Simulates full agent flow: list_files -> run_tests (baseline) -> read_file ->
   write_file (clean code) -> run_tests (verification) -> summary report.
2. Full live end-to-end test against messy_project with GeminiProvider.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pytest

from evidor import (
    GenerationRequest,
    GenerationResponse,
    InMemorySink,
    ModelProvider,
    ToolCall,
)
from evidor.telemetry import LLMCallEndEvent, ToolCallEndEvent

from codecraft.cloner import clone_or_load_repo
from codecraft.engine import Codecraft
from codecraft.test_runner import execute_pytest_direct


class ScriptedRefactoringProvider:
    """Deterministic mock provider that simulates a skilled refactoring agent loop."""

    def __init__(self, refactored_code: str) -> None:
        self.refactored_code = refactored_code
        self.turns = 0
        self.model = "mock-refactorer"

    def with_model(self, model: str) -> "ScriptedRefactoringProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.turns += 1

        # Turn 1: Discover directory tree
        if self.turns == 1:
            return GenerationResponse(
                text="Exploring project files.",
                model=self.model,
                tool_calls=(
                    ToolCall(id="call_list", name="list_files", arguments={"path": ".", "recursive": True}),
                ),
            )

        # Turn 2: Run baseline tests
        if self.turns == 2:
            return GenerationResponse(
                text="Running baseline tests to verify current state.",
                model=self.model,
                tool_calls=(
                    ToolCall(id="call_test_1", name="run_tests", arguments={}),
                ),
            )

        # Turn 3: Read messy calculator service
        if self.turns == 3:
            return GenerationResponse(
                text="Reading calculator_service.py to inspect code smells.",
                model=self.model,
                tool_calls=(
                    ToolCall(id="call_read", name="read_file", arguments={"path": "src/calculator_service.py"}),
                ),
            )

        # Turn 4: Write clean refactored code
        if self.turns == 4:
            return GenerationResponse(
                text="Refactoring calculator_service.py with proper types, docstrings, and clean design.",
                model=self.model,
                tool_calls=(
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "src/calculator_service.py", "content": self.refactored_code},
                    ),
                ),
            )

        # Turn 5: Run tests again to verify refactored code
        if self.turns == 5:
            return GenerationResponse(
                text="Re-running tests to confirm no regressions.",
                model=self.model,
                tool_calls=(
                    ToolCall(id="call_test_2", name="run_tests", arguments={}),
                ),
            )

        # Turn 6: Conclude with report
        return GenerationResponse(
            text=(
                "Successfully refactored calculator_service.py! Added comprehensive type hints, "
                "eliminated mutable default argument bug, added PEP 257 docstrings, and verified "
                "that all 8 tests pass cleanly with zero regressions."
            ),
            model=self.model,
        )


CLEAN_REFACTORED_CODE = """\"\"\"Clean, PEP 8 compliant, type-annotated calculator service.\"\"\"

from __future__ import annotations

from typing import Any, Sequence


def do_math(a: float, b: float, op: str) -> float:
    \"\"\"Perform a binary arithmetic calculation.

    Args:
        a: Left operand.
        b: Right operand.
        op: Arithmetic operator string (add, sub, mul, div, pow or symbol equivalents).

    Returns:
        The calculated result as a float.

    Raises:
        ValueError: If division by zero is attempted or operator is unknown.
    \"\"\"
    normalized_op = op.strip().lower()
    if normalized_op in ("add", "+"):
        return a + b
    if normalized_op in ("sub", "-"):
        return a - b
    if normalized_op in ("mul", "*"):
        return a * b
    if normalized_op in ("div", "/"):
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a / b
    if normalized_op in ("pow", "^", "**"):
        return a ** b
    raise ValueError(f"Unknown operator: {op}")


def calc_stats(numbers: Sequence[float]) -> dict[str, Any]:
    \"\"\"Compute summary statistics (count, sum, average, min, max) for numbers.\"\"\"
    if not numbers:
        return {"count": 0, "sum": 0, "avg": 0, "min": None, "max": None}
    
    total = sum(numbers)
    count = len(numbers)
    return {
        "count": count,
        "sum": total,
        "avg": total / count,
        "min": min(numbers),
        "max": max(numbers),
    }


def record_history(entry: str, history: list[str] | None = None) -> list[str]:
    \"\"\"Safely record an operation history entry without mutable default argument bug.\"\"\"
    active_history = list(history) if history is not None else []
    active_history.append(entry)
    return active_history
"""


def test_deterministic_codecraft_refactor_pipeline():
    """Verify Codecraft end-to-end orchestration deterministically with mock provider."""
    messy_root = Path(__file__).parent.parent / "messy_project"
    mock_provider = ScriptedRefactoringProvider(CLEAN_REFACTORED_CODE)

    with tempfile.TemporaryDirectory() as tmp_sandbox:
        orchestrator = Codecraft(
            source=messy_root,
            sandbox_dir=tmp_sandbox,
            provider=mock_provider,
        )

        report = orchestrator.refactor()
        orchestrator.close()

        assert report.success is True
        assert report.baseline_tests["success"] is True
        assert report.verification_tests["success"] is True
        assert "src/calculator_service.py" in report.modified_files
        assert report.total_tool_calls == 5
        assert report.tool_call_breakdown["list_files"] == 1
        assert report.tool_call_breakdown["run_tests"] == 2
        assert report.tool_call_breakdown["read_file"] == 1
        assert report.tool_call_breakdown["write_file"] == 1
        assert "Successfully refactored" in report.agent_summary


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY required for live test")
def test_live_codecraft_refactor_gemini():
    """Live integration test: Run Codecraft with Gemini on messy_project."""
    messy_root = Path(__file__).parent.parent / "messy_project"

    with tempfile.TemporaryDirectory() as tmp_sandbox:
        orchestrator = Codecraft(
            source=messy_root,
            sandbox_dir=tmp_sandbox,
            model_name="gemini-2.5-flash",
            max_tool_iterations=15,
        )

        report = orchestrator.refactor(
            instructions=(
                "You are Codecraft. 1) Discover files with list_files. "
                "2) Run tests with run_tests to verify baseline. "
                "3) Read src/calculator_service.py. "
                "4) Refactor src/calculator_service.py with write_file to add type hints and docstrings. "
                "5) Run tests with run_tests to ensure all tests pass. "
                "6) Provide a concise report of your actions."
            )
        )
        orchestrator.close()

        # Baseline must have passed
        assert report.baseline_tests["success"] is True
        # Verification must pass
        assert report.verification_tests["success"] is True
        assert report.success is True
        assert report.total_tool_calls >= 3

