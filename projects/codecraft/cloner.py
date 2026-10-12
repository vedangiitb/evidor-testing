"""Repository cloner and sandbox workspace manager for Codecraft."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.parse

logger = logging.getLogger("codecraft.cloner")


def is_git_url(source: str) -> bool:
    """Check if the source string looks like a remote Git repository URL."""
    source_lower = source.strip().lower()
    if source_lower.startswith(("http://", "https://", "git://", "ssh://", "git@")):
        return True
    if "github.com" in source_lower or "gitlab.com" in source_lower or "bitbucket.org" in source_lower:
        return True
    return False


def get_default_repo_name(source: str) -> str:
    """Derive a friendly folder name from a Git URL or directory path."""
    source_clean = source.strip().rstrip("/\\")
    if is_git_url(source):
        parsed = urllib.parse.urlparse(source_clean)
        path = parsed.path.rstrip("/")
        name = Path(path).stem
        if name.endswith(".git"):
            name = name[:-4]
        return name or "cloned_repo"
    return Path(source_clean).name or "local_repo"


def clone_or_load_repo(
    source: str | Path,
    sandbox_root: str | Path | None = None,
    *,
    shallow: bool = True,
    timeout_seconds: int = 60,
) -> Path:
    """Clone a remote Git repository or copy a local project into a sandbox directory.

    Args:
        source: A remote GitHub/Git URL or a local folder path.
        sandbox_root: Destination directory. If None, a temporary directory is created.
        shallow: Whether to use shallow clone (`--depth 1`) for faster remote cloning.
        timeout_seconds: Max seconds to wait for git clone.

    Returns:
        The absolute Path to the prepared sandbox project root.

    Raises:
        ValueError: If source is invalid or local path does not exist.
        RuntimeError: If git clone fails.
    """
    source_str = str(source).strip()
    repo_name = get_default_repo_name(source_str)

    if sandbox_root is None:
        sandbox_dir = Path(tempfile.mkdtemp(prefix=f"codecraft_{repo_name}_")).resolve()
    else:
        sandbox_dir = Path(sandbox_root).resolve()
        sandbox_dir.mkdir(parents=True, exist_ok=True)

    target_dir = sandbox_dir / repo_name
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)

    if is_git_url(source_str):
        logger.info(f"Cloning remote repository {source_str} into {target_dir}...")
        cmd = ["git", "clone"]
        if shallow:
            cmd.extend(["--depth", "1"])
        cmd.extend([source_str, str(target_dir)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as err:
            raise RuntimeError("Git is not installed or not available in PATH.") from err
        except subprocess.TimeoutExpired as err:
            raise RuntimeError(f"Git clone timed out after {timeout_seconds} seconds.") from err

        if result.returncode != 0:
            err_msg = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Failed to clone repository '{source_str}': {err_msg}")

        logger.info(f"Successfully cloned {source_str} to {target_dir}")
        return target_dir.resolve()

    # Local path handling
    local_path = Path(source_str).resolve()
    if not local_path.exists():
        raise ValueError(f"Local source project does not exist: {source_str}")
    if not local_path.is_dir():
        raise ValueError(f"Local source project must be a directory: {source_str}")

    logger.info(f"Copying local project {local_path} into sandbox {target_dir}...")
    shutil.copytree(
        local_path,
        target_dir,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv", ".pytest_cache"),
    )
    logger.info(f"Successfully prepared sandbox at {target_dir}")
    return target_dir.resolve()

