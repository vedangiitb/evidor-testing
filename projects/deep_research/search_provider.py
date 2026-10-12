"""Tavily search provider and mock provider for deep research."""

from __future__ import annotations

import os
from typing import Sequence
import urllib.parse

from evidor import SearchResult, TavilySearchProvider, WebSearchProvider


class MockSearchProvider:
    """Mock search provider for deterministic unit testing without external network calls."""

    def __init__(self, predefined_results: dict[str, Sequence[SearchResult]] | None = None) -> None:
        self.predefined_results = predefined_results or {}
        self.recorded_queries: list[str] = []

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        self.recorded_queries.append(query)
        if query in self.predefined_results:
            return list(self.predefined_results[query])[:max_results]

        # Case-insensitive partial match
        for key, res in self.predefined_results.items():
            if key.lower() in query.lower() or query.lower() in key.lower():
                return list(res)[:max_results]

        # Default synthetic results
        return [
            SearchResult(
                title=f"Analysis of {query}",
                url=f"https://research.example.org/{urllib.parse.quote(query[:20])}",
                snippet=f"Detailed production breakdown, architecture, and benchmarks for {query}.",
            ),
            SearchResult(
                title=f"Benchmarking & Trade-offs: {query}",
                url=f"https://docs.agents.example.com/comparisons/{urllib.parse.quote(query[:20])}",
                snippet=f"Evaluating performance, developer experience, and deployment trade-offs for {query}.",
            ),
        ][:max_results]


def create_search_provider(
    api_key: str | None = None,
    mock_provider: MockSearchProvider | None = None,
) -> WebSearchProvider:
    """Instantiate Tavily search provider (or mock if provided for testing)."""
    if mock_provider is not None:
        return mock_provider

    key = api_key or os.getenv("TAVILY_API_KEY")
    return TavilySearchProvider(api_key=key)


__all__ = [
    "TavilySearchProvider",
    "MockSearchProvider",
    "WebSearchProvider",
    "create_search_provider",
]
