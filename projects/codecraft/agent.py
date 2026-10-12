"""Agent factory for Codecraft built on Evidor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv()

from evidor import (
    Agent,
    GeminiProvider,
    ModelProvider,
    RetryConfig,
    TelemetryRuntime,
    TelemetrySink,
)
from .tools import build_codecraft_tools

SYSTEM_PROMPT = """You are Codecraft, an elite autonomous code refactoring agent powered by Evidor.
Your mission is to take an existing codebase in any programming language (Python, TypeScript, JavaScript, Go, Rust, Java, C++, etc.), analyze its structure, read messy code, run automated tests, and produce clean, production-grade refactored code while ensuring all tests pass with zero regressions.

### Your Professional Workflow:
1. **Explore the Workspace**: Use `list_files` to discover the project tree and directory structure.
2. **Run Baseline Tests**: ALWAYS invoke `run_tests` first to determine the baseline passing/failing state before modifying any files.
3. **Inspect the Code**: Use `read_file` and `search_files` to inspect messy files, locate code smells (such as mutable defaults, antipatterns, poor naming, lack of types/interfaces, duplicate logic, and unhandled exceptions), and understand the test expectations.
4. **Refactor Incrementally**:
   - Write clean, idiomatic code adhering to the project's language conventions (e.g. PEP 8 & typing for Python, ESLint/Prettier & TypeScript for JS/TS, idiomatic Go for Go, Rustfmt for Rust).
   - Add comprehensive type annotations/interfaces and informative docstrings/comments.
   - Break large monolithic functions into modular, focused helper functions.
   - Fix latent bugs and anti-patterns.
   - CRITICAL OVERWRITE RULE: `write_file` replaces the ENTIRE file contents. Never output partial diffs or snippets into `write_file`. Always write complete, runnable files.
5. **Verify with Tests**: Immediately call `run_tests` after modifying files. If any test fails, read the test output, fix the regression, and re-test until tests pass cleanly.
6. **Report**: Conclude with a clear summary of all refactored files, improvements made, and test status.

### Sandboxing & Operational Rules:
- All paths MUST be relative to the project root (e.g., 'src/service.py', NOT '../service.py' or absolute paths).
- Never attempt to traverse outside the project root; Evidor enforces a strict sandbox.
- If `read_file` reports that a file exceeds the byte limit, use `read_file_chunk` to inspect it in slices, and consider refactoring it into smaller modules.
- Ensure files are encoded cleanly as UTF-8.
"""


def create_codecraft_agent(
    project_root: Path,
    *,
    custom_test_cmd: str | None = None,
    provider: ModelProvider | None = None,
    model_name: str = "gemini-2.5-flash",
    max_tool_iterations: int = 25,
    tool_timeout: float = 35.0,
    context_window: int = 32_000,
    max_messages: int = 50,
    max_file_bytes: int = 100_000,
    telemetry: TelemetryRuntime | TelemetrySink | Sequence[TelemetrySink] | None = None,
    retry_config: RetryConfig | bool | None = None,
    include_resilience_tools: bool = True,
) -> Agent:
    """Create a Codecraft agent scoped to the given project root.

    Args:
        project_root: Root directory of the project to refactor.
        custom_test_cmd: Optional explicit test command (e.g. 'npm test', 'cargo test').
        provider: Custom Evidor ModelProvider. If None, GeminiProvider is instantiated.
        model_name: Model name if default GeminiProvider is used.
        max_tool_iterations: Maximum loop iterations per turn.
        tool_timeout: Timeout for tool execution.
        context_window: Max token window before compaction.
        max_messages: Max messages before compaction.
        max_file_bytes: Max file size for read/write.
        telemetry: Optional telemetry sinks (InMemorySink, ConsoleSink, etc.).
        retry_config: Optional retry configuration.
        include_resilience_tools: Whether to include chunked reader & file info tools.

    Returns:
        Configured Evidor Agent instance.
    """
    root = Path(project_root).resolve(strict=True)

    if provider is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set in environment or .env. "
                "Provide an explicit ModelProvider or set GEMINI_API_KEY."
            )
        provider = GeminiProvider(model=model_name)

    tools = build_codecraft_tools(
        root,
        custom_test_cmd=custom_test_cmd,
        max_file_bytes=max_file_bytes,
        test_timeout=tool_timeout,
        include_resilience_tools=include_resilience_tools,
    )

    return Agent(
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        max_tool_iterations=max_tool_iterations,
        tool_timeout=tool_timeout,
        context_window=context_window,
        max_messages=max_messages,
        telemetry=telemetry,
        retry_config=retry_config or RetryConfig(max_retries=3, initial_delay=1.0),
    )
