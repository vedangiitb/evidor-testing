"""Edge Case Probes: Multi-Turn File Editing, Newline Handling, and UTF-8 Encoding.

This test suite investigates:
1. Windows CRLF (`\\r\\n`) vs Linux LF (`\\n`) handling:
   - How Evidor preserves CRLF during read, but converts to LF when an LLM writes standard newlines.
2. Non-UTF-8 encoding pitfalls:
   - Files encoded with Windows-1252 / ISO-8859-1 (smart quotes, accented characters)
     cause read_file to raise `ValueError("File is not valid UTF-8 text")`.
3. UTF-8 with BOM (`\\xef\\xbb\\xbf`):
   - Decoded text contains `\\ufeff` (Zero-Width No-Break Space), impacting parsers and string comparisons.
4. Multi-turn full-file replacement pitfall:
   - Demonstrating that `write_file` completely replaces the file contents,
     and showing how Codecraft's `run_tests` catches partial file overwrites and forces self-correction.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from evidor import filesystem_tools
from codecraft.tools import build_codecraft_tools
from codecraft.test_runner import execute_pytest_direct


@pytest.fixture
def encoding_env():
    """Create sandbox with various encoding and newline fixtures."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sandbox = Path(tmp_dir).resolve()

        # 1. Windows CRLF file
        crlf_file = sandbox / "windows_crlf.py"
        crlf_content = "def hello():\r\n    msg = 'Windows line endings'\r\n    return msg\r\n"
        crlf_file.write_bytes(crlf_content.encode("utf-8"))

        # 2. Windows-1252 encoded file (e.g. old Windows codebase with Latin characters)
        win1252_file = sandbox / "legacy_win1252.py"
        # Byte \xe9 is 'é' in Windows-1252/Latin-1, invalid in UTF-8
        win1252_bytes = b"# Author: Ren\xe9 Descartes\r\nx = 100\r\n"
        win1252_file.write_bytes(win1252_bytes)

        # 3. UTF-8 with BOM file (common in Windows Notepad)
        bom_file = sandbox / "with_bom.py"
        bom_bytes = b"\xef\xbb\xbfdef get_val():\r\n    return 42\r\n"
        bom_file.write_bytes(bom_bytes)

        # 4. Multi-function module with unit tests for multi-turn editing test
        service_file = sandbox / "service.py"
        service_file.write_text(
            "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
            encoding="utf-8",
        )
        test_file = sandbox / "test_service.py"
        test_file.write_text(
            "from service import add, multiply\n\n"
            "def test_service():\n"
            "    assert add(2, 3) == 5\n"
            "    assert multiply(2, 3) == 6\n",
            encoding="utf-8",
        )

        yield {
            "sandbox": sandbox,
            "crlf_file": crlf_file,
            "win1252_file": win1252_file,
            "bom_file": bom_file,
            "service_file": service_file,
            "test_file": test_file,
        }


def test_windows_crlf_preservation_and_lf_replacement(encoding_env):
    """Test CRLF detection during read and conversion to LF upon LLM write_file."""
    sandbox = encoding_env["sandbox"]
    tools = {t.name: t for t in filesystem_tools(sandbox)}

    # 1. read_file reads raw binary and decodes utf-8, preserving \r\n in python string
    text = tools["read_file"].func("windows_crlf.py")
    assert "\r\n" in text
    assert text.count("\r\n") == 3

    # 2. When an LLM generates code with \n and calls write_file
    llm_updated_code = "def hello():\n    msg = 'Updated with Unix line endings'\n    return msg\n"
    tools["write_file"].func("windows_crlf.py", llm_updated_code)

    # 3. Inspect raw bytes on disk
    raw_on_disk = encoding_env["crlf_file"].read_bytes()
    assert b"\r\n" not in raw_on_disk
    assert b"\n" in raw_on_disk
    assert b"Updated with Unix line endings" in raw_on_disk


def test_non_utf8_windows1252_rejection(encoding_env):
    """Verify that non-UTF-8 files (Windows-1252) cause read_file to raise ValueError."""
    sandbox = encoding_env["sandbox"]
    tools = {t.name: t for t in filesystem_tools(sandbox)}

    # read_file attempts strict UTF-8 decode
    with pytest.raises(ValueError) as exc:
        tools["read_file"].func("legacy_win1252.py")
    assert "File is not valid UTF-8 text" in str(exc.value)


def test_file_info_detects_non_utf8_and_crlf(encoding_env):
    """Verify Codecraft's get_file_info tool identifies non-UTF-8 and CRLF line endings."""
    sandbox = encoding_env["sandbox"]
    tools = {t.name: t for t in build_codecraft_tools(sandbox)}

    # 1. Check CRLF file
    crlf_info = tools["get_file_info"].func("windows_crlf.py")
    assert crlf_info["newline"] == "CRLF (Windows)"
    assert crlf_info["is_utf8"] is True

    # 2. Check legacy Windows-1252 file
    win1252_info = tools["get_file_info"].func("legacy_win1252.py")
    assert win1252_info["is_utf8"] is False
    assert win1252_info["size_bytes"] > 0


def test_utf8_with_bom_handling(encoding_env):
    """Verify that UTF-8 BOM is decoded into \\ufeff character."""
    sandbox = encoding_env["sandbox"]
    tools = {t.name: t for t in filesystem_tools(sandbox)}

    content = tools["read_file"].func("with_bom.py")
    # Standard UTF-8 decoding keeps BOM as character U+FEFF
    assert content.startswith("\ufeff")
    assert "def get_val():" in content


def test_multiturn_file_replacement_pitfall_and_test_recovery(encoding_env):
    """Test the classic multi-turn file overwrite trap:
    If an agent only outputs the modified function instead of the entire file,
    the remaining functions are erased, and tests immediately fail.
    
    Verifies that running tests detects this regression immediately!
    """
    sandbox = encoding_env["sandbox"]
    tools = {t.name: t for t in build_codecraft_tools(sandbox)}

    # Baseline: both add() and multiply() exist; tests pass
    baseline = execute_pytest_direct(sandbox, test_target="test_service.py")
    assert baseline["success"] is True

    # Turn 1: Bad agent action — writes ONLY the refactored add() function, omitting multiply()!
    partial_code = "def add(a: int, b: int) -> int:\n    \"\"\"Add two integers.\"\"\"\n    return a + b\n"
    tools["write_file"].func("service.py", partial_code)

    # Verification: tests immediately FAIL because multiply() was truncated!
    broken_test = execute_pytest_direct(sandbox, test_target="test_service.py")
    assert broken_test["success"] is False
    assert "ImportError" in broken_test["output"] or "cannot import name 'multiply'" in broken_test["output"]

    # Turn 2: Self-correction — agent inspects test failure and writes complete module
    complete_code = (
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Add two numbers cleanly.\"\"\"\n"
        "    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n"
        "    \"\"\"Multiply two numbers cleanly.\"\"\"\n"
        "    return a * b\n"
    )
    tools["write_file"].func("service.py", complete_code)

    # Re-testing: now all tests pass cleanly!
    recovered_test = execute_pytest_direct(sandbox, test_target="test_service.py")
    assert recovered_test["success"] is True

