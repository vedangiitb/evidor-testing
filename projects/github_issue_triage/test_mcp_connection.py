"""Quick verification script to test Evidor's MCP connection and tool discovery."""

import os
import sys
from dotenv import load_dotenv

from .agent import create_triage_agent

load_dotenv()


def main() -> None:
    token = (
        os.getenv("GITHUB_TOKEN")
        or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        or os.getenv("GH_TOKEN")
    )
    if not token:
        print("[!] GITHUB_TOKEN is not set in environment or .env file.")
        print("    Add GITHUB_TOKEN='ghp_...' or GITHUB_PERSONAL_ACCESS_TOKEN to .env")
        print("    to authenticate with the remote GitHub MCP server.")
        return

    use_local = "--local" in sys.argv
    mode = "local" if use_local else "remote"
    print(f"[*] Initializing Evidor agent with {mode.upper()} GitHub MCP server...")
    agent = create_triage_agent(mode=mode)

    try:
        print("[*] Discovering tools via MCP...")
        tools = agent.tools
        print(f"[+] Successfully connected to MCP server! Discovered {len(tools)} tools:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")
    finally:
        agent.close()


if __name__ == "__main__":
    main()

