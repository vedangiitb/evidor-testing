"""Command-line interface for running the Deep Research Agent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import webbrowser

# Ensure repository root is on sys.path regardless of execution directory
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_project_dir = Path(__file__).resolve().parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from dotenv import load_dotenv

load_dotenv()

from evidor import GeminiProvider, OpenAIProvider

try:
    from projects.deep_research.engine import DeepResearchEngine
    from projects.deep_research.search_provider import TavilySearchProvider, create_search_provider
except (ModuleNotFoundError, ImportError):
    try:
        from .engine import DeepResearchEngine
        from .search_provider import TavilySearchProvider, create_search_provider
    except (ImportError, ValueError):
        from engine import DeepResearchEngine
        from search_provider import TavilySearchProvider, create_search_provider


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evidor Deep Research Agent — Autonomous web research, claim comparison, and cited reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Compare the main Python agent frameworks and identify their trade-offs for production deployments.",
        help="Research topic or comparative question.",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["gemini", "openai"],
        default="gemini",
        help="Model provider to use.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-2.5-flash",
        help="Model name (e.g., 'gemini-2.5-flash', 'gpt-4o-mini', 'gpt-4o').",
    )
    parser.add_argument(
        "--tavily-key",
        type=str,
        default=os.getenv("TAVILY_API_KEY"),
        help="Tavily API key (defaults to TAVILY_API_KEY from environment/.env).",
    )
    parser.add_argument(
        "--no-phoenix",
        action="store_true",
        help="Disable Arize Phoenix telemetry tracing.",
    )
    parser.add_argument(
        "--phoenix-endpoint",
        type=str,
        default=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
        help="Arize Phoenix OTLP traces collector endpoint.",
    )
    parser.add_argument(
        "--phoenix-project",
        type=str,
        default="deep-research",
        help="Arize Phoenix project name.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=25,
        help="Maximum agent tool iterations.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output/reports",
        help="Directory to save final research markdown reports.",
    )
    parser.add_argument(
        "--open-phoenix",
        action="store_true",
        help="Automatically open Arize Phoenix dashboard in default web browser.",
    )
    return parser.parse_args(args)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    print("=" * 70)
    print(" 🚀 Evidor Deep Research Agent with Arize Phoenix Telemetry")
    print("=" * 70)
    print(f"Topic:            {args.query}")
    print(f"Provider:         {args.provider.upper()} ({args.model})")
    print(f"Phoenix Tracing:  {'Disabled' if args.no_phoenix else 'Enabled'}")
    print("-" * 70)

    # Provider setup
    if args.provider == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            print("ERROR: GEMINI_API_KEY is not set in environment or .env file.")
            sys.exit(1)
        provider = GeminiProvider(model=args.model)
    else:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("ERROR: OPENAI_API_KEY is not set in environment or .env file.")
            sys.exit(1)
        provider = OpenAIProvider(model=args.model)

    search_prov = create_search_provider(api_key=args.tavily_key)

    engine = DeepResearchEngine(
        output_dir=args.output_dir,
        provider=provider,
        model_name=args.model,
        search_provider=search_prov,
        enable_phoenix=not args.no_phoenix,
        phoenix_endpoint=args.phoenix_endpoint,
        phoenix_project_name=args.phoenix_project,
        max_tool_iterations=args.max_iterations,
    )

    if engine.phoenix_ui_url:
        print(f"📊 Arize Phoenix Dashboard: {engine.phoenix_ui_url}")
        if args.open_phoenix:
            webbrowser.open(engine.phoenix_ui_url)

    print("\n🔍 Conducting autonomous research and analyzing sources...\n")

    def progress_callback(action: str, details: dict):
        if action == "start":
            print("  ⏳ Agent initiated research plan.")
        elif action == "complete":
            print(f"  ✅ Research completed in {details.get('elapsed', 0):.1f}s.")

    try:
        report = engine.research(args.query, progress_callback=progress_callback)
    finally:
        engine.close()

    print("\n" + "=" * 70)
    print(" 📋 Research Summary & Telemetry Metrics")
    print("=" * 70)
    print(f"Discovered Sources:      {len(report.sources)}")
    print(f"Recorded Findings:       {len(report.findings)}")
    print(f"Claim Comparisons:       {len(report.comparisons)}")
    print(f"Total Tool Invocations:  {report.telemetry.total_tool_calls}")
    for tool_name, count in report.telemetry.tool_counts.items():
        print(f"  - {tool_name}: {count}")
    print(f"LLM Invocations:         {report.telemetry.total_llm_calls}")
    print(f"Total Tokens:            {report.telemetry.total_tokens:,}")
    print(f"Execution Time:          {report.telemetry.execution_time_seconds:.2f}s")
    if report.saved_file_path:
        print(f"\n💾 Saved Report:         {report.saved_file_path}")
    if report.telemetry.phoenix_url:
        print(f"📊 View Phoenix Traces:   {report.telemetry.phoenix_url}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
