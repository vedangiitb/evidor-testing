"""Tool bundle builder for Codecraft."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evidor import Tool, filesystem_tools, tool
from .test_runner import build_test_runner


def build_chunked_reader(root: Path, max_bytes_per_chunk: int = 50_000) -> Tool:
    """Create a chunk-based reader tool for handling files that exceed max_file_bytes."""
    resolved_root = Path(root).resolve(strict=True)

    @tool(
        name="read_file_chunk",
        description=(
            "Read a specific line range from a text file under the project root. "
            "Use this when a file is too large for read_file or when you only need to inspect a section."
        ),
    )
    def read_file_chunk(path: str, start_line: int = 1, line_count: int = 100) -> str:
        """Read a slice of lines from a file.

        Args:
            path: File path relative to project root.
            start_line: 1-indexed starting line number.
            line_count: Number of lines to read (clamped to max 500).
        """
        if start_line < 1:
            raise ValueError("start_line must be at least 1")
        count = min(max(1, line_count), 500)

        target = (resolved_root / path).resolve(strict=True)
        try:
            target.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("Path is outside the configured filesystem root") from exc

        if not target.is_file():
            raise ValueError(f"Not a file: {path}")

        try:
            # Read with utf-8, fallback with replace to avoid crash on dirty encoding
            with target.open("r", encoding="utf-8", errors="replace") as f:
                lines = []
                for idx, line in enumerate(f, start=1):
                    if idx < start_line:
                        continue
                    if idx >= start_line + count:
                        break
                    lines.append(f"{idx}: {line}")
                return "".join(lines) if lines else f"[No lines in range {start_line}-{start_line + count - 1}]"
        except Exception as exc:
            raise ValueError(f"Failed to read chunk from {path}: {exc}") from exc

    return read_file_chunk


def build_file_info_tool(root: Path) -> Tool:
    """Create a diagnostic tool for inspecting file metadata, size, line endings, and encoding."""
    resolved_root = Path(root).resolve(strict=True)

    @tool(
        name="get_file_info",
        description=(
            "Inspect file metadata including size in bytes, total line count, detected line endings (CRLF vs LF), "
            "and encoding health."
        ),
    )
    def get_file_info(path: str) -> dict[str, Any]:
        """Inspect file metadata.

        Args:
            path: File path relative to project root.
        """
        target = (resolved_root / path).resolve(strict=True)
        try:
            target.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("Path is outside the configured filesystem root") from exc

        if not target.is_file():
            raise ValueError(f"Not a file: {path}")

        raw_bytes = target.read_bytes()
        size_bytes = len(raw_bytes)
        has_crlf = b"\r\n" in raw_bytes
        has_bom = raw_bytes.startswith(b"\xef\xbb\xbf")

        try:
            text = raw_bytes.decode("utf-8")
            is_utf8 = True
            line_count = len(text.splitlines())
        except UnicodeDecodeError:
            is_utf8 = False
            line_count = len(raw_bytes.splitlines())

        return {
            "path": target.relative_to(resolved_root).as_posix(),
            "size_bytes": size_bytes,
            "line_count": line_count,
            "is_utf8": is_utf8,
            "has_bom": has_bom,
            "newline": "CRLF (Windows)" if has_crlf else "LF (Unix)",
        }

    return get_file_info


def build_codecraft_tools(
    project_root: Path,
    *,
    custom_test_cmd: str | None = None,
    max_file_bytes: int = 100_000,
    max_results: int = 100,
    max_files_scanned: int = 1000,
    test_timeout: float = 30.0,
    include_resilience_tools: bool = True,
) -> tuple[Tool, ...]:
    """Assemble all tools for Codecraft: Evidor filesystem tools + test runner + diagnostics."""
    root = Path(project_root).resolve(strict=True)

    fs_tools = filesystem_tools(
        root_dir=root,
        max_file_bytes=max_file_bytes,
        max_results=max_results,
        max_files_scanned=max_files_scanned,
    )
    test_tool = build_test_runner(root, custom_cmd=custom_test_cmd, default_timeout=test_timeout)

    tools = list(fs_tools) + [test_tool]
    if include_resilience_tools:
        tools.append(build_chunked_reader(root))
        tools.append(build_file_info_tool(root))

    return tuple(tools)

