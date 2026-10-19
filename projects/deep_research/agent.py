"""Agent factory and deep research system prompt for Evidor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

from evidor import (
    Agent,
    GeminiProvider,
    ModelProvider,
    OpenAIProvider,
    TelemetryRuntime,
    TelemetrySink,
)
try:
    from .search_provider import WebSearchProvider
    from .tools import ResearchContext, build_research_tools
except (ImportError, ValueError):
    from search_provider import WebSearchProvider
    from tools import ResearchContext, build_research_tools

DEEP_RESEARCH_SYSTEM_PROMPT = """You are an elite Autonomous Deep Research Agent powered by Evidor.
Your mission is to perform rigorous, publication-grade technical research on complex topics, synthesize multiple web sources, compare competing claims and architectures, evaluate production trade-offs, and produce an exhaustively cited research report.

### Your 5-Phase Research Methodology:

1. **Deconstruct & Plan**:
   - Break down the research query into core dimensions (e.g., architecture, state persistence, production observability, latency, scalability, developer ergonomics, maintenance costs).
   - Formulate distinct search queries targeted at documentation, production postmortems, benchmarks, and architectural analyses.

2. **Discover & Search (`web_search`)**:
   - Perform targeted searches for the primary frameworks, technologies, or subjects.
   - Do NOT stop at a single search. Execute multiple queries from different angles (e.g., 'framework X production trade-offs', 'framework X vs Y state management benchmark', 'framework Z telemetry observability').

3. **Read Primary Sources (`read_webpage`)**:
   - Surface search snippets are insufficient for deep analysis. You MUST call `read_webpage` on the most authoritative URLs to extract in-depth technical details, code architecture, and factual claims.
   - Extract primary claims, quotes, and specific performance/architectural details.

4. **Cross-Examine & Record Evidence (`record_evidence`, `compare_evidence`, `list_evidence`)**:
   - For every key finding, metric, or trade-off discovered, invoke `record_evidence` with the topic, finding, and source URL.
   - To inspect what has already been captured, you can invoke `list_evidence`.
   - For competing technologies or conflicting claims, invoke `compare_evidence` to explicitly contrast trade-offs, architectural divergences, or consensus.
   - Highlight where sources agree, where they conflict, and where critical production trade-offs exist (e.g., flexibility vs complexity, in-memory state vs distributed persistence, SDK vendor lock-in).

5. **Synthesize & Save Report (`save_research_report`)**:
   - Structure your final report with publication-grade rigor:
     - **# Title & Executive Summary**: High-level takeaways for engineering leaders and architects.
     - **## Comparative Matrix**: A detailed Markdown table comparing candidates across key production criteria (Core Abstraction, State Management, Observability/Tracing, Failure Recovery, Deployment Overhead, Ideal Use Case).
     - **## In-Depth Analysis of Candidates**: Deep technical dive into each framework or concept with specific architecture patterns.
     - **## Critical Trade-Off Analysis**: Focused comparison of production concerns (e.g. debugging distributed states, telemetry integration, vendor lock-in, latency budgets).
     - **## Decision Framework & Recommendations**: Concrete rules of thumb ("Choose X if...", "Avoid Y when...").
     - **## Comprehensive Citations & References**: Explicit numbered references `[1]`, `[2]`, `[3]` throughout the text, matched with a full bibliography listing:
       - `[n] Title — URL — Key Takeaway / Corroborating Claim`
   - Invoke `save_research_report` to persist the markdown report to disk.
   - Finally, return your response providing a clear summary and notifying the user that the report has been generated and saved.

### Operational Rules:
- Maintain strict objectivity. Evaluate strengths and weaknesses fairly.
- Ground all claims in retrieved sources. If a source lacks evidence for a claim, state that clearly.
- Include explicit numbered citations like `[1]`, `[2]` inline in every section.
"""


def create_research_agent(
    context: ResearchContext,
    *,
    provider: ModelProvider | None = None,
    model_name: str = "gemini-3.1-flash-lite",
    search_provider: WebSearchProvider | None = None,
    max_search_results: int = 5,
    max_tool_iterations: int = 30,
    tool_timeout: float = 25.0,
    telemetry: TelemetryRuntime | TelemetrySink | Sequence[TelemetrySink] | None = None,
) -> Agent:
    """Create a configured Deep Research Agent.

    Args:
        context: Research context tracking findings, sources, and reports.
        provider: Evidor ModelProvider instance. If None, instantiates GeminiProvider or OpenAIProvider.
        model_name: Model identifier if provider is auto-instantiated.
        search_provider: Web search provider (DuckDuckGo, Tavily, Hybrid, or Mock).
        max_search_results: Max results per search tool invocation.
        max_tool_iterations: Maximum tool call iterations per turn.
        tool_timeout: Timeout in seconds for each tool execution.
        telemetry: Attached telemetry sinks (e.g. PhoenixSink, InMemorySink).

    Returns:
        Configured Evidor Agent.
    """
    if provider is None:
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if gemini_key:
            provider = GeminiProvider(model=model_name)
        elif openai_key:
            provider = OpenAIProvider(model="gpt-4.1-mini" if "flash" in model_name else model_name)
        else:
            raise ValueError(
                "Neither GEMINI_API_KEY nor OPENAI_API_KEY found in environment or .env. "
                "Please configure an API key or pass a custom ModelProvider."
            )

    tools = build_research_tools(
        context,
        search_provider=search_provider,
        max_search_results=max_search_results,
        page_timeout=tool_timeout,
    )

    return Agent(
        provider=provider,
        system_prompt=DEEP_RESEARCH_SYSTEM_PROMPT,
        tools=tools,
        max_tool_iterations=max_tool_iterations,
        tool_timeout=tool_timeout,
        telemetry=telemetry,
    )

