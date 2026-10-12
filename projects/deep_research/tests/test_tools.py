"""Unit tests for deep research tools and search providers."""

from pathlib import Path
import tempfile
import pytest

from evidor import SearchResult, TavilySearchProvider
from projects.deep_research.models import ResearchFinding, SourceDocument
from projects.deep_research.search_provider import (
    MockSearchProvider,
    create_search_provider,
)
from projects.deep_research.tools import (
    ResearchContext,
    build_research_tools,
    fetch_and_clean_html,
)


def test_mock_search_provider():
    """Verify MockSearchProvider returns results for matched or synthetic queries."""
    provider = MockSearchProvider(
        predefined_results={
            "agent frameworks": [
                SearchResult(
                    title="LangGraph Overview",
                    url="https://docs.langchain.com/langgraph",
                    snippet="Cyclic graph-based agent architecture.",
                )
            ]
        }
    )

    results = provider.search("agent frameworks", max_results=5)
    assert len(results) == 1
    assert results[0].title == "LangGraph Overview"

    # Synthetic fallback query
    synth_results = provider.search("arbitrary search query", max_results=2)
    assert len(synth_results) == 2
    assert "arbitrary search query" in synth_results[0].title


def test_create_search_provider():
    """Verify create_search_provider returns TavilySearchProvider or injected mock."""
    mock = MockSearchProvider()
    provider = create_search_provider(mock_provider=mock)
    assert provider is mock

    tavily_prov = create_search_provider(api_key="tvly-test-key")
    assert isinstance(tavily_prov, TavilySearchProvider)


def test_built_in_web_search_tool_integration():
    """Verify build_research_tools attaches Evidor built-in web_search tool directly."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = ResearchContext(output_dir=Path(tmp))
        mock_search = MockSearchProvider(
            predefined_results={
                "Python agents": [
                    SearchResult(
                        title="Python Agent Frameworks",
                        url="https://example.com/agents",
                        snippet="Comparison of Python frameworks.",
                    )
                ]
            }
        )
        tools = build_research_tools(ctx, search_provider=mock_search)

        tool_map = {t.name: t for t in tools}
        assert "web_search" in tool_map
        assert "read_webpage" in tool_map
        assert "record_evidence" in tool_map
        assert "list_evidence" in tool_map
        assert "compare_evidence" in tool_map
        assert "save_research_report" in tool_map
        assert "get_current_time" in tool_map

        # Verify built-in web_search tool runs directly and returns serialized search results
        search_res = tool_map["web_search"]("Python agents")
        assert len(search_res) == 1
        assert search_res[0]["title"] == "Python Agent Frameworks"
        assert search_res[0]["url"] == "https://example.com/agents"

        # Test native record_evidence tool from EvidenceStore
        e_res = tool_map["record_evidence"](
            topic="LangGraph",
            finding="Durable execution with Postgres checkpointers",
            source="https://docs.langchain.com/langgraph",
            aspect="persistence",
            category="architecture",
        )
        assert "F1" in e_res
        assert len(ctx.findings) == 1
        assert ctx.findings[0].topic == "LangGraph"
        assert ctx.findings[0].category == "architecture"

        # Test native list_evidence tool
        l_res = tool_map["list_evidence"](topic="langgraph")
        assert "Found 1 evidence item" in l_res
        assert "LangGraph" in l_res

        # Test native compare_evidence tool
        c_res = tool_map["compare_evidence"](
            aspect="State Management",
            subject_a="LangGraph",
            claim_a="Requires explicit graph state reducers and checkpointers",
            subject_b="CrewAI",
            claim_b="Relies on role-playing memory and sequential agents",
            comparison_type="trade-off",
            analysis="LangGraph provides deterministic state recovery while CrewAI is higher-level.",
        )
        assert "C1" in c_res
        assert len(ctx.comparisons) == 1
        assert ctx.comparisons[0].comparison_type == "trade-off"
        assert ctx.comparisons[0].topic_aspect == "State Management"

        # Test save report
        s_res = tool_map["save_research_report"](
            topic="Test Agents",
            markdown_content="# Test Report\n\nContent here.",
        )
        assert "successfully saved" in s_res
        assert ctx.final_report_path is not None
        assert ctx.final_report_path.exists()
        assert "# Test Report" in ctx.final_report_path.read_text(encoding="utf-8")


def test_fetch_and_clean_html_invalid_url():
    """Verify clean error message when URL is invalid."""
    title, text, status = fetch_and_clean_html("http://invalid-domain-that-does-not-exist.local")
    assert status != "ok"
    assert "Error reading" in status
