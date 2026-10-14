"""Main entry point to test issue triage with Evidor and Remote GitHub MCP."""

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
        print("[!] GITHUB_TOKEN is not set.")
        print("    Please set GITHUB_TOKEN or GITHUB_PERSONAL_ACCESS_TOKEN in your .env file.")
        print("    Example: GITHUB_TOKEN=\"ghp_xxxxxxxxxxxx\"")
        return

    use_local = "--local" in sys.argv
    clean_argv = [a for a in sys.argv[1:] if a != "--local"]
    mode = "local" if use_local else (os.getenv("GITHUB_MCP_MODE") or "remote")

    # Accept repo and issue from command line args or use interactive/default prompt
    if len(clean_argv) >= 2:
        repo = clean_argv[0]
        issue_num = clean_argv[1]
        query = f"Please triage issue #{issue_num} from repository {repo}."
    elif len(clean_argv) == 1:
        repo = clean_argv[0]
        query = f"Please fetch and triage the latest open issues from repository {repo}."
    else:
        repo = input("Enter GitHub repository (owner/repo): ").strip()
        if not repo:
            print("No repository provided. Exiting.")
            return
        issue = input("Enter issue number (optional, press Enter to list open issues): ").strip()
        if issue:
            query = f"Please triage issue #{issue} from repository {repo}."
        else:
            query = f"Please fetch and triage the latest open issues from repository {repo}."

    print(f"[*] MCP Server Mode: {mode.upper()}")
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        print("[*] Langfuse telemetry: ACTIVE")

    print(f"\n[*] Starting triage agent with prompt:\n    '{query}'\n")

    agent = create_triage_agent(mode=mode)
    try:
        response = agent.send(query)
        print("=" * 60)
        print("TRIAGE RESULT:")
        print("=" * 60)
        print(response.text)
        print("=" * 60)
    finally:
        agent.close()


if __name__ == "__main__":
    main()

