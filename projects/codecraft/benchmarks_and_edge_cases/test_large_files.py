"""Edge Case Probes: Large File Reads, Writes, and Buffer Limits.

This test suite investigates:
1. What happens when files exceed `max_file_bytes` in Evidor's filesystem tools?
   - `read_file` raises `ValueError("File exceeds the X-byte read limit")`.
   - `search_files` silently skips large files (a major hidden trap!).
   - `write_file` raises `ValueError("File exceeds the X-byte write limit")`.
2. Exact boundary behavior:
   - Size == max_file_bytes (allowed).
   - Size == max_file_bytes + 1 (rejected).
3. Multi-byte UTF-8 string length vs byte length (e.g., emojis / 4-byte codepoints).
4. Codecraft's resilience strategy:
   - Reading slices via `read_file_chunk`.
5. Live Agent Error Recovery:
   - The agent encounters a file exceeding `max_file_bytes`, receives the tool error,
     and recovers cleanly without crashing.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from evidor import filesystem_tools
from evidor.tools.core.tool_executor import ToolExecutor
from evidor.models import ToolCall
from codecraft.tools import build_codecraft_tools
from codecraft.agent import create_codecraft_agent


@pytest.fixture
def large_files_env():
    """Create a sandbox with various file sizes and test limits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sandbox = Path(tmp_dir).resolve()

        # 50-byte file
        (sandbox / "small.txt").write_text("A" * 50, encoding="utf-8")

        # Exactly 100-byte file
        (sandbox / "exact_100.txt").write_text("B" * 100, encoding="utf-8")

        # 101-byte file
        (sandbox / "oversized_101.txt").write_text("C" * 101, encoding="utf-8")

        # 5,000-byte file (50 lines of 100 chars)
        lines = [f"Line {i:03d}: " + ("X" * 90) + "\n" for i in range(1, 51)]
        (sandbox / "large_monolith.py").write_text("".join(lines), encoding="utf-8")

        yield {
            "sandbox": sandbox,
            "small": sandbox / "small.txt",
            "exact_100": sandbox / "exact_100.txt",
            "oversized_101": sandbox / "oversized_101.txt",
            "large_monolith": sandbox / "large_monolith.py",
        }


def test_read_file_boundary_limit(large_files_env):
    """Verify read_file boundary at max_file_bytes."""
    sandbox = large_files_env["sandbox"]
    # Limit: 100 bytes
    tools = {t.name: t for t in filesystem_tools(sandbox, max_file_bytes=100)}

    # 1. 50 bytes should succeed
    content_small = tools["read_file"].func("small.txt")
    assert len(content_small) == 50

    # 2. Exactly 100 bytes should succeed
    content_exact = tools["read_file"].func("exact_100.txt")
    assert len(content_exact) == 100

    # 3. 101 bytes must raise ValueError
    with pytest.raises(ValueError) as exc:
        tools["read_file"].func("oversized_101.txt")
    assert "File exceeds the 100-byte read limit" in str(exc.value)


def test_search_files_silently_skips_large_files(large_files_env):
    """UNCOVERED TRAP:
    search_files silently ignores files larger than max_file_bytes without returning any warning.
    """
    sandbox = large_files_env["sandbox"]
    tools = {t.name: t for t in filesystem_tools(sandbox, max_file_bytes=100)}

    # 'Line 001' is in large_monolith.py, but large_monolith.py is 5000 bytes > 100 bytes
    matches = tools["search_files"].func(query="Line 001")
    # Result is empty! The file was silently skipped by search_files
    assert len(matches) == 0

    # Whereas searching in small.txt (<= 100 bytes) finds the match
    small_matches = tools["search_files"].func(query="AAA")
    assert len(small_matches) == 1
    assert small_matches[0]["path"] == "small.txt"


def test_write_file_and_create_file_limit(large_files_env):
    """Verify write_file and create_file reject contents exceeding max_file_bytes."""
    sandbox = large_files_env["sandbox"]
    tools = {t.name: t for t in filesystem_tools(sandbox, max_file_bytes=100)}

    # write_file with 105 bytes
    with pytest.raises(ValueError) as exc:
        tools["write_file"].func("small.txt", "Z" * 105)
    assert "File exceeds the 100-byte write limit" in str(exc.value)

    # create_file with 105 bytes
    with pytest.raises(ValueError) as exc:
        tools["create_file"].func("new_oversized.txt", "Z" * 105)
    assert "File exceeds the 100-byte write limit" in str(exc.value)


def test_multibyte_utf8_byte_vs_string_length(large_files_env):
    """Verify limits count encoded UTF-8 bytes, not Python string characters."""
    sandbox = large_files_env["sandbox"]
    tools = {t.name: t for t in filesystem_tools(sandbox, max_file_bytes=100)}

    # Each fire emoji is 4 UTF-8 bytes. 30 emojis = 120 bytes (string length is only 30!)
    thirty_emojis = "🔥" * 30
    assert len(thirty_emojis) == 30
    assert len(thirty_emojis.encode("utf-8")) == 120

    with pytest.raises(ValueError) as exc:
        tools["write_file"].func("small.txt", thirty_emojis)
    assert "File exceeds the 100-byte write limit" in str(exc.value)


def test_codecraft_chunked_reader_resilience(large_files_env):
    """Verify Codecraft's read_file_chunk tool successfully reads slices of oversized files."""
    sandbox = large_files_env["sandbox"]
    tools = {t.name: t for t in build_codecraft_tools(sandbox, max_file_bytes=100)}

    # read_file fails on large_monolith.py (5000 bytes)
    with pytest.raises(ValueError):
        tools["read_file"].func("large_monolith.py")

    # But read_file_chunk reads lines 1 to 5 cleanly!
    chunk = tools["read_file_chunk"].func("large_monolith.py", start_line=1, line_count=5)
    assert "Line 001" in chunk
    assert "Line 005" in chunk
    assert "Line 006" not in chunk


def test_agent_large_file_error_recovery(large_files_env):
    """Live LLM test: Verify agent recovers when read_file reports byte limit exceeded."""
    sandbox = large_files_env["sandbox"]
    # Provide a 2000-byte limit; large_monolith is 5000 bytes
    agent = create_codecraft_agent(
        sandbox,
        max_file_bytes=2000,
        max_tool_iterations=4,
    )

    prompt = (
        "Please read 'large_monolith.py'. If read_file fails because the file exceeds the byte limit, "
        "use read_file_chunk to read the first 3 lines of 'large_monolith.py'."
    )
    response = agent.send(prompt)
    agent.close()

    # Verify agent attempted read_file or read_file_chunk
    tool_names = [m.name for m in agent.messages if m.role == "tool"]
    assert "read_file" in tool_names or "read_file_chunk" in tool_names
    # Verify agent recognized the size constraint
    res_lower = response.text.lower()
    assert any(term in res_lower for term in ["too large", "exceed", "limit", "chunk", "line"])
