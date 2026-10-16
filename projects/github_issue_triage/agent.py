"""GitHub Issue Triage Agent initialized with Evidor and Remote GitHub MCP server."""

import os
import shutil
from dotenv import load_dotenv

from evidor import Agent, GeminiProvider, LangfuseSink, MCPServerConfig

load_dotenv()

DEFAULT_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

TRIAGE_SYSTEM_PROMPT = """You are a GitHub Issue Triage Agent.
Your job is to triage GitHub issues using GitHub MCP tools.

When given an issue or repository to triage:
1. Use your GitHub tools to fetch or search relevant issues, issue details, or comments.
2. Determine the category: Bug, Feature Request, Question, Documentation, or Chore.
3. Determine priority/severity: Low, Medium, High, or Critical.
4. Verify whether necessary reproduction steps, stack traces, or environment details are provided.
5. Provide a concise triage summary with:
   - Summary of the problem
   - Category and Priority
   - Recommended labels
   - Missing info or suggested next actions
"""


def create_triage_agent(
    github_token: str | None = None,
    mcp_url: str | None = None,
    model: str = "gemini-3.8-flash",
    enable_langfuse: bool = True,
    mode: str | None = None,
) -> Agent:
    """Creates and returns an Evidor Agent configured with either remote or local GitHub MCP server."""
    token = (
        github_token
        or os.getenv("GITHUB_TOKEN")
        or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        or os.getenv("GH_TOKEN")
        or ""
    )
    mcp_mode = (mode or os.getenv("GITHUB_MCP_MODE", "remote")).lower()

    if mcp_mode == "local":
        cmd = shutil.which("npx") or "npx"
        github_mcp = MCPServerConfig.stdio(
            name="github",
            command=cmd,
            args=["-y", "@modelcontextprotocol/server-github"],
            env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": token},
        )
    else:
        url = mcp_url or os.getenv("GITHUB_MCP_URL", DEFAULT_GITHUB_MCP_URL)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        github_mcp = MCPServerConfig.http(
            name="github",
            url=url,
            headers=headers,
        )

    telemetry = None
    if enable_langfuse and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        telemetry = LangfuseSink()

    return Agent(
        GeminiProvider(model=model),
        mcp_servers=[github_mcp],
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        max_tool_iterations=8,
        telemetry=telemetry,
    )

