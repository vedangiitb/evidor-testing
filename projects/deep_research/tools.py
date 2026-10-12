"""Specialized research tools for web search, source reading, claim comparison, and report generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from evidor import (
    ComparisonItem,
    EvidenceItem,
    EvidenceStore,
    TavilySearchProvider,
    Tool,
    WebSearchProvider,
    fetch_webpage_content,
    get_current_time,
    read_webpage_tool,
    tool,
    web_search,
)
try:
    from .models import ClaimComparison, ResearchFinding, SourceDocument
except (ImportError, ValueError):
    from models import ClaimComparison, ResearchFinding, SourceDocument


@dataclass
class ResearchContext:
    """In-memory research session context tracking sources, findings, and comparisons."""

    output_dir: Path
    evidence_store: EvidenceStore = field(default_factory=lambda: EvidenceStore(prefix="F", comparison_prefix="C"))
    sources: dict[str, SourceDocument] = field(default_factory=dict)
    activity_log: list[dict[str, Any]] = field(default_factory=list)
    final_report_md: str = ""
    final_report_path: Path | None = None

    @property
    def findings(self) -> list[EvidenceItem]:
        """All recorded findings and evidence items."""
        return self.evidence_store.items

    @property
    def comparisons(self) -> list[ComparisonItem]:
        """All recorded comparative claims and trade-offs."""
        return self.evidence_store.comparisons

    def log_activity(self, action: str, details: dict[str, Any]) -> None:
        self.activity_log.append({
            "action": action,
            "details": details,
        })


def fetch_and_clean_html(url: str, timeout: float = 12.0) -> tuple[str, str, str]:
    """Fetch URL and clean HTML content using Evidor's native webpage reader.

    Returns:
        tuple of (title, cleaned_text, status_message)
    """
    res = fetch_webpage_content(url, timeout=timeout)
    if res.startswith("Error") or res.startswith("Warning"):
        return "", "", res

    lines = [line.strip() for line in res.splitlines() if line.strip()]
    title = lines[0][:100] if lines else url
    return title, res, "ok"


def build_research_tools(
    context: ResearchContext,
    *,
    search_provider: WebSearchProvider | None = None,
    max_search_results: int = 5,
    page_timeout: float = 12.0,
) -> list[Tool]:
    """Construct the research agent toolkit directly from Evidor's native tools."""
    provider = search_provider or TavilySearchProvider()

    # 1. Report saving tool (the only domain-specific filesystem tool for this project)
    @tool(
        name="save_research_report",
        description="Persist the final comprehensive, cited markdown report to the local workspace disk.",
    )
    def save_research_report(topic: str, markdown_content: str) -> str:
        """Save the synthesized research report.

        Args:
            topic: The research topic title.
            markdown_content: The full, formatted Markdown report with citations [1], [2] and References.
        """
        context.output_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", topic.lower())[:50].strip("_")
        report_file = context.output_dir / f"report_{slug}.md"

        report_file.write_text(markdown_content, encoding="utf-8")
        context.final_report_md = markdown_content
        context.final_report_path = report_file
        context.log_activity("save_report", {"path": str(report_file), "bytes": len(markdown_content)})

        return f"Report successfully saved to {report_file.resolve()}."

    # Directly assemble native Evidor tools without redundant wrappers:
    # - web_search: Evidor native search tool
    # - read_webpage_tool: Evidor native webpage extractor
    # - context.evidence_store.tools: Evidor native record_evidence, list_evidence, compare_evidence
    # - get_current_time: Evidor native timestamp tool
    return [
        web_search(provider, max_results=max_search_results),
        read_webpage_tool(max_chars=8000, timeout=page_timeout),
        *context.evidence_store.tools,
        save_research_report,
        get_current_time,
    ]
