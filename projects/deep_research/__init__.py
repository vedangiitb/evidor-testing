"""Evidor Deep Research Agent Package."""

from .agent import create_research_agent
from .engine import DeepResearchEngine
from .models import (
    ClaimComparison,
    ResearchFinding,
    ResearchReport,
    SearchResultItem,
    SourceDocument,
    TelemetryStats,
)
from .search_provider import (
    MockSearchProvider,
    TavilySearchProvider,
    WebSearchProvider,
    create_search_provider,
)
from .tools import ResearchContext, build_research_tools

__all__ = [
    "DeepResearchEngine",
    "create_research_agent",
    "ResearchReport",
    "ResearchFinding",
    "ClaimComparison",
    "SourceDocument",
    "SearchResultItem",
    "TelemetryStats",
    "TavilySearchProvider",
    "MockSearchProvider",
    "WebSearchProvider",
    "create_search_provider",
    "ResearchContext",
    "build_research_tools",
]
