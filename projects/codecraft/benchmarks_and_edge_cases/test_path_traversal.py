"""Edge Case & Sandboxing Probes: Path Traversal and Filesystem Escapes.

This test suite investigates:
1. Does Evidor cleanly reject path traversal attempts with `../`?
2. What happens when an outside file exists vs when it does NOT exist?
   - Uncovering the subtle leak where non-existent paths raise `FileNotFoundError`
     containing the host system's full absolute path!
3. Absolute path behavior (Windows drive letters `C:\\...` and Unix root `/...`).
4. Rejection across all filesystem tools: read_file, write_file, create_file, delete_file, list_files, search_files.
5. In-sandbox relative path normalization (e.g., `sub/../sub/file.txt`).
6. Agent recovery: How the LLM agent handles rejection when an out-of-bounds path is encountered.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pytest

from evidor import filesystem_tools
from evidor.tools.core.tool_executor import ToolExecutor
from evidor.models import ToolCall
from codecraft.agent import create_codecraft_agent
from codecraft.test_runner import execute_pytest_direct


@pytest.fixture
def sandbox_env():
    """Create a temporary sandbox root alongside a sibling outside directory with sensitive data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir).resolve()
        
        # Outside directory (simulates host filesystem / private data)
        outside_dir = base / "host_system"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.env"
        secret_file.write_text("DB_PASSWORD=super_secret_12345", encoding="utf-8")

        # Sandboxed project root
        sandbox_root = base / "sandbox_project"
        sandbox_root.mkdir()
        (sandbox_root / "src").mkdir()
        (sandbox_root / "src" / "app.py").write_text("print('inside sandbox')", encoding="utf-8")

        yield {
            "base": base,
            "outside_dir": outside_dir,
            "secret_file": secret_file,
            "sandbox_root": sandbox_root,
        }


def test_path_traversal_rejection_existing_file(sandbox_env):
    """Verify Evidor rejects path traversal to an EXISTING file outside sandbox."""
    sandbox = sandbox_env["sandbox_root"]
    tools = {t.name: t for t in filesystem_tools(sandbox)}
    read_tool = tools["read_file"]

    # Attempting to read ../host_system/secret.env
    with pytest.raises(ValueError) as excinfo:
        read_tool.func("../host_system/secret.env")
    
    assert "Path is outside the configured filesystem root" in str(excinfo.value)


def test_path_traversal_nonexistent_file_leak_edge_case(sandbox_env):
    """UNCOVERED BUG/EDGE CASE:
    When an outside file DOES NOT EXIST, resolve(strict=True) raises FileNotFoundError
    BEFORE relative_to(root) check!
    
    This leaks the host's full absolute path instead of saying 'Path is outside the configured filesystem root'.
    """
    sandbox = sandbox_env["sandbox_root"]
    tools = {t.name: t for t in filesystem_tools(sandbox)}
    read_tool = tools["read_file"]

    # Attempt to read a non-existent outside file
    with pytest.raises(FileNotFoundError) as excinfo:
        read_tool.func("../host_system/does_not_exist.txt")

    # The exception message contains the full resolved host path!
    error_msg = str(excinfo.value)
    assert "host_system" in error_msg
    assert "does_not_exist.txt" in error_msg


def test_create_file_destination_path_traversal(sandbox_env):
    """Verify create_file uses strict=False and always raises ValueError on outside destinations."""
    sandbox = sandbox_env["sandbox_root"]
    tools = {t.name: t for t in filesystem_tools(sandbox)}
    create_tool = tools["create_file"]

    # create_file resolves with strict=False, so it properly catches outside paths
    # even when the file does not exist!
    with pytest.raises(ValueError) as excinfo:
        create_tool.func("../host_system/evil_script.py", "malicious_code")
    
    assert "Path is outside the configured filesystem root" in str(excinfo.value)


def test_absolute_path_escape_attempt(sandbox_env):
    """Verify absolute paths (like C:\\Windows or /etc/passwd) are rejected."""
    sandbox = sandbox_env["sandbox_root"]
    tools = {t.name: t for t in filesystem_tools(sandbox)}
    read_tool = tools["read_file"]

    # On Windows test with win.ini or hosts, on Unix /etc/hosts
    target = "C:/Windows/win.ini" if os.name == "nt" else "/etc/hosts"
    
    if Path(target).exists():
        with pytest.raises(ValueError) as excinfo:
            read_tool.func(target)
        assert "Path is outside the configured filesystem root" in str(excinfo.value)


def test_relative_path_inside_root_allowed(sandbox_env):
    """Verify redundant relative paths that stay inside root (e.g. src/../src/app.py) work."""
    sandbox = sandbox_env["sandbox_root"]
    tools = {t.name: t for t in filesystem_tools(sandbox)}
    read_tool = tools["read_file"]

    # Should resolve cleanly to src/app.py inside the sandbox
    content = read_tool.func("src/../src/app.py")
    assert "inside sandbox" in content


def test_tool_executor_handles_traversal_gracefully(sandbox_env):
    """Verify ToolExecutor catches the ValueError and returns a formatted string to LLM."""
    sandbox = sandbox_env["sandbox_root"]
    fs_tools = filesystem_tools(sandbox)
    executor = ToolExecutor(fs_tools, default_timeout=5.0)

    # Simulate model asking to execute read_file outside sandbox
    call = ToolCall(id="call_escape_1", name="read_file", arguments={"path": "../host_system/secret.env"})
    result_str = executor.execute(call)

    # ToolExecutor must not crash; it returns an error string suitable for the agent
    assert "Error executing tool 'read_file'" in result_str
    assert "Path is outside the configured filesystem root" in result_str


def test_all_filesystem_mutation_tools_sandboxed(sandbox_env):
    """Verify write_file and delete_file also reject out-of-sandbox targets."""
    sandbox = sandbox_env["sandbox_root"]
    tools = {t.name: t for t in filesystem_tools(sandbox)}

    # write_file
    with pytest.raises(ValueError) as exc:
        tools["write_file"].func("../host_system/secret.env", "overwritten")
    assert "Path is outside the configured filesystem root" in str(exc.value)

    # delete_file
    with pytest.raises(ValueError) as exc:
        tools["delete_file"].func("../host_system/secret.env")
    assert "Path is outside the configured filesystem root" in str(exc.value)

    # list_files
    with pytest.raises(ValueError) as exc:
        tools["list_files"].func("../host_system")
    assert "Path is outside the configured filesystem root" in str(exc.value)


def test_agent_path_traversal_recovery(sandbox_env):
    """Live LLM test: Ask agent to read a file outside the sandbox and verify it recovers."""
    sandbox = sandbox_env["sandbox_root"]
    agent = create_codecraft_agent(sandbox, max_tool_iterations=4)

    # Ask the agent to inspect a file outside the sandbox
    prompt = (
        "Please read the configuration file located at '../host_system/secret.env'. "
        "If you cannot access it due to sandboxing, inspect 'src/app.py' inside the project instead."
    )
    response = agent.send(prompt)
    agent.close()

    # The agent should see the sandboxing error from the tool and fall back to reading src/app.py
    assert len(agent.messages) >= 3
    # Check that tool was called and error was handled
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert any("Path is outside" in m.content or "Error executing tool" in m.content for m in tool_messages)
    # The agent acknowledges or reports reading app.py
    assert "app.py" in response.text or "inside sandbox" in response.text or "outside" in response.text.lower()

