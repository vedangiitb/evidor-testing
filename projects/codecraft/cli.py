"""Command-line interface for Codecraft."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from codecraft.engine import Codecraft

# Ensure terminal doesn't crash on UTF-8 characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional rich formatting with graceful fallback to standard print
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
    console = Console(safe_box=True)
except ImportError:
    HAS_RICH = False
    console = None


@contextmanager
def status_spinner(message: str):
    """Context manager for status display, using rich if available or simple print."""
    if HAS_RICH and console is not None:
        with console.status(message):
            yield
    else:
        print(f"\n--> {message.strip('.')}...")
        yield


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="codecraft",
        description="Autonomous code refactoring agent built with Evidor.",
    )
    parser.add_argument(
        "source",
        help="GitHub repository URL (https://github.com/...) or path to local project.",
    )
    parser.add_argument(
        "--sandbox",
        "-s",
        default=None,
        help="Optional sandbox directory path (defaults to a temp folder).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gemini-2.5-flash",
        help="Model to use (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--test-cmd",
        "-t",
        default=None,
        help="Optional test command override (e.g. 'npm test', 'cargo test', 'go test ./...'). Auto-detected by default.",
    )
    parser.add_argument(
        "--phoenix",
        action="store_true",
        help="Enable Arize Phoenix distributed trace export.",
    )
    parser.add_argument(
        "--phoenix-endpoint",
        default=None,
        help="Arize Phoenix OTLP HTTP endpoint (default: http://localhost:6006/v1/traces).",
    )
    parser.add_argument(
        "--phoenix-project",
        default="codecraft",
        help="Arize Phoenix project name (default: codecraft).",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=None,
        help="Custom refactoring instructions for the agent.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=25,
        help="Max tool iterations per turn (default: 25).",
    )

    args = parser.parse_args()

    phoenix_info = f"\nPhoenix Tracing: [cyan]{args.phoenix_endpoint or 'http://localhost:6006'} (project: {args.phoenix_project})[/cyan]" if args.phoenix else ""

    if HAS_RICH and console is not None:
        console.print(
            Panel.fit(
                f"[bold cyan]Codecraft[/bold cyan] [green]v1.0.0[/green]\n"
                f"[dim]Powered by Evidor Agent Runtime[/dim]\n\n"
                f"Target: [yellow]{args.source}[/yellow]\n"
                f"Model:  [magenta]{args.model}[/magenta]"
                + (f"\nTest Command: [blue]{args.test_cmd}[/blue]" if args.test_cmd else "")
                + phoenix_info,
                title="Codecraft Refactoring",
                border_style="cyan",
            )
        )
    else:
        print("=" * 60)
        print("  Codecraft v1.0.0 (Powered by Evidor)")
        print(f"  Target: {args.source}")
        print(f"  Model:  {args.model}")
        if args.test_cmd:
            print(f"  Test Command: {args.test_cmd}")
        if args.phoenix:
            print(f"  Phoenix Tracing: {args.phoenix_endpoint or 'http://localhost:6006'} (project: {args.phoenix_project})")
        print("=" * 60)

    with status_spinner("Preparing project sandbox and initializing Evidor agent"):
        orchestrator = Codecraft(
            source=args.source,
            sandbox_dir=args.sandbox,
            model_name=args.model,
            custom_test_cmd=args.test_cmd,
            enable_phoenix=args.phoenix,
            phoenix_endpoint=args.phoenix_endpoint,
            phoenix_project_name=args.phoenix_project,
            max_tool_iterations=args.max_iterations,
        )
        project_root = orchestrator.setup()

    if HAS_RICH and console is not None:
        console.print(f"Sandbox root: [bold white]{project_root}[/bold white]\n")
    else:
        print(f"Sandbox root: {project_root}\n")

    with status_spinner("Codecraft agent is analyzing, testing, and refactoring"):
        report = orchestrator.refactor(instructions=args.prompt)

    # Render summary table
    if HAS_RICH and console is not None:
        table = Table(title="Codecraft Refactoring Summary", border_style="cyan")
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="cyan")

        table.add_row("Baseline Tests", f"[{'green' if report.baseline_tests.get('success') else 'red'}]{report.baseline_tests.get('summary', 'N/A')}[/]")
        table.add_row("Verification Tests", f"[{'green' if report.verification_tests.get('success') else 'red'}]{report.verification_tests.get('summary', 'N/A')}[/]")
        table.add_row("Modified Files", ", ".join(report.modified_files) or "None")
        table.add_row("Total Tool Calls", str(report.total_tool_calls))
        table.add_row("Total LLM Calls", str(report.total_llm_calls))
        if report.phoenix_url:
            table.add_row("Phoenix Traces", f"[link={report.phoenix_url}]{report.phoenix_url}[/link]")
        table.add_row("Status", "[bold green]SUCCESS (All tests pass)[/]" if report.success else "[bold red]FAILED[/]")

        console.print(table)
        console.print()
        console.print(Panel(report.agent_summary, title="Agent Refactoring Report", border_style="green"))
    else:
        print("\n" + "=" * 60)
        print("  CODECRAFT REFACTORING SUMMARY")
        print("=" * 60)
        print(f"Baseline Tests:     {report.baseline_tests.get('summary', 'N/A')}")
        print(f"Verification Tests: {report.verification_tests.get('summary', 'N/A')}")
        print(f"Modified Files:     {', '.join(report.modified_files) or 'None'}")
        print(f"Total Tool Calls:   {report.total_tool_calls}")
        print(f"Total LLM Calls:    {report.total_llm_calls}")
        if report.phoenix_url:
            print(f"Phoenix Dashboard:  {report.phoenix_url}")
        print(f"Status:             {'SUCCESS (All tests pass)' if report.success else 'FAILED'}")
        print("-" * 60)
        print("Agent Report:\n")
        print(report.agent_summary)
        print("=" * 60)

    orchestrator.close()
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())
