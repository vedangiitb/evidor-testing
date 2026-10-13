"""Data models and schemas for Deep Research Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import urllib.parse


@dataclass
class SearchResultItem:
    """A search result item with metadata."""

    title: str
    url: str
    snippet: str = ""
    domain: str = ""

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            try:
                parsed = urllib.parse.urlparse(self.url)
                self.domain = parsed.netloc.lower()
            except Exception:
                self.domain = ""


@dataclass
class SourceDocument:
    """Full text or cleaned article extracted from a web source."""

    url: str
    title: str
    content: str
    word_count: int = 0
    domain: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "ok"

    def __post_init__(self) -> None:
        if not self.word_count and self.content:
            self.word_count = len(self.content.split())
        if not self.domain and self.url:
            try:
                parsed = urllib.parse.urlparse(self.url)
                self.domain = parsed.netloc.lower()
            except Exception:
                self.domain = ""


@dataclass
class ResearchFinding:
    """An individual claim, fact, or trade-off extracted by the agent."""

    id: str
    claim: str
    source_url: str
    supporting_quote: str = ""
    category: str = "finding"  # e.g. finding, trade-off, limitation, architecture, benchmark
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "source_url": self.source_url,
            "supporting_quote": self.supporting_quote,
            "category": self.category,
            "timestamp": self.timestamp,
        }


@dataclass
class ClaimComparison:
    """Cross-source comparison between two claims or viewpoints."""

    id: str
    topic_aspect: str
    source_a: str
    claim_a: str
    source_b: str
    claim_b: str
    comparison_type: str  # agreement, discrepancy, nuance, trade-off
    analysis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic_aspect": self.topic_aspect,
            "source_a": self.source_a,
            "claim_a": self.claim_a,
            "source_b": self.source_b,
            "claim_b": self.claim_b,
            "comparison_type": self.comparison_type,
            "analysis": self.analysis,
        }


@dataclass
class TelemetryStats:
    """Metrics and observability telemetry collected from the run."""

    total_llm_calls: int = 0
    total_tool_calls: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    execution_time_seconds: float = 0.0
    phoenix_url: str | None = None
    phoenix_project: str = "deep-research"
    langfuse_enabled: bool = False
    langfuse_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "tool_counts": self.tool_counts,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "phoenix_url": self.phoenix_url,
            "phoenix_project": self.phoenix_project,
            "langfuse_enabled": self.langfuse_enabled,
            "langfuse_url": self.langfuse_url,
        }


@dataclass
class ResearchReport:
    """Final synthesized research report and artifacts."""

    topic: str
    markdown_report: str
    findings: list[ResearchFinding] = field(default_factory=list)
    comparisons: list[ClaimComparison] = field(default_factory=list)
    sources: list[SourceDocument] = field(default_factory=list)
    telemetry: TelemetryStats = field(default_factory=TelemetryStats)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    saved_file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "markdown_report": self.markdown_report,
            "findings_count": len(self.findings),
            "comparisons_count": len(self.comparisons),
            "sources_count": len(self.sources),
            "telemetry": self.telemetry.to_dict(),
            "generated_at": self.generated_at,
            "saved_file_path": self.saved_file_path,
        }

