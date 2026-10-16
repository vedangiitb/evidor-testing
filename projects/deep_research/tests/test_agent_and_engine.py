"""End-to-end tests for DeepResearchEngine using ScriptedMockProvider."""

from pathlib import Path
import tempfile
import pytest

from evidor import GenerationRequest, GenerationResponse, ToolCall
from projects.deep_research.engine import DeepResearchEngine
from projects.deep_research.models import ResearchReport
from projects.deep_research.search_provider import MockSearchProvider


class ScriptedResearchMockProvider:
    """Mock provider executing a 3-step research trajectory deterministically."""

    def __init__(self) -> None:
        self.model = "mock-research-model"
        self.context_strategy = "stateless"
        self.turns = [
            # Turn 1: Call web_search
            GenerationResponse(
                text="I will search for the main Python agent frameworks.",
                tool_calls=(
                    ToolCall(
                        id="call_search",
                        name="web_search",
                        arguments={"query": "Python agent frameworks comparison"},
                    ),
                ),
                model=self.model,
            ),
            # Turn 2: Record evidence and compare trade-offs
            GenerationResponse(
                text="I will record the findings and compare trade-offs.",
                tool_calls=(
                    ToolCall(
                        id="call_find",
                        name="record_evidence",
                        arguments={
                            "topic": "LangGraph",
                            "finding": "LangGraph uses explicit cyclic graphs and Postgres checkpointers for durability.",
                            "source": "https://research.example.org/langgraph",
                            "aspect": "durability",
                            "category": "trade-off",
                        },
                    ),
                    ToolCall(
                        id="call_comp",
                        name="compare_evidence",
                        arguments={
                            "aspect": "State Persistence",
                            "subject_a": "LangGraph",
                            "claim_a": "Deterministic checkpointing over graph transitions.",
                            "subject_b": "CrewAI",
                            "claim_b": "High-level role-playing memory abstraction.",
                            "comparison_type": "trade-off",
                            "analysis": "LangGraph offers lower-level control and fault recovery, while CrewAI prioritizes rapid prototyping.",
                        },
                    ),
                ),
                model=self.model,
            ),
            # Turn 3: Save research report
            GenerationResponse(
                text="Synthesizing final research report.",
                tool_calls=(
                    ToolCall(
                        id="call_report",
                        name="save_research_report",
                        arguments={
                            "topic": "Python Agent Frameworks Production Comparison",
                            "markdown_content": (
                                "# Python Agent Frameworks for Production\n\n"
                                "## Executive Summary\n"
                                "LangGraph [1], CrewAI [2], and Evidor [3] present distinct trade-offs.\n\n"
                                "## Comparative Matrix\n"
                                "| Framework | Core Paradigm | State Management | Production Trade-off |\n"
                                "|---|---|---|---|\n"
                                "| LangGraph | Cyclic Graph | Durable Checkpoints | Steep learning curve, high ceremony |\n"
                                "| CrewAI | Role-Playing Teams | Memory Subsystem | Less deterministic control |\n"
                                "| Evidor | Clean Provider-Agnostic Runtime | Provider-Managed Context | Zero-overhead telemetry, built-in resilience |\n\n"
                                "## References\n"
                                "[1] LangGraph Docs — https://research.example.org/langgraph\n"
                                "[2] CrewAI Docs — https://research.example.org/crewai\n"
                                "[3] Evidor Docs — https://evidorlabs.com\n"
                            ),
                        },
                    ),
                ),
                model=self.model,
            ),
            # Turn 4: Final user message
            GenerationResponse(
                text="The deep research report on Python agent frameworks has been successfully compiled and saved.",
                model=self.model,
            ),
        ]

    def with_model(self, model: str) -> "ScriptedResearchMockProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.turns:
            return self.turns.pop(0)
        return GenerationResponse(text="Research complete.", model=self.model)


def test_deep_research_engine_end_to_end():
    """Verify DeepResearchEngine completes autonomous multi-turn research, captures telemetry, and saves report."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_provider = ScriptedResearchMockProvider()
        mock_search = MockSearchProvider()

        engine = DeepResearchEngine(
            output_dir=Path(tmp_dir),
            provider=mock_provider,
            search_provider=mock_search,
            enable_phoenix=False,  # Test without requiring active network collector
            max_tool_iterations=10,
        )

        progress_events = []

        def callback(action: str, details: dict):
            progress_events.append((action, details))

        report = engine.research(
            "Compare the main Python agent frameworks and identify their trade-offs for production deployments.",
            progress_callback=callback,
        )
        engine.close()

        # Assertions on ResearchReport
        assert isinstance(report, ResearchReport)
        assert len(report.findings) >= 1
        assert report.findings[0].category == "trade-off"
        assert len(report.comparisons) >= 1
        assert report.comparisons[0].topic_aspect == "State Persistence"
        assert "Python Agent Frameworks for Production" in report.markdown_report
        assert report.saved_file_path is not None
        assert Path(report.saved_file_path).exists()

        # Assertions on TelemetryStats
        assert report.telemetry.total_llm_calls >= 3
        assert report.telemetry.total_tool_calls >= 4
        assert "web_search" in report.telemetry.tool_counts
        assert "record_evidence" in report.telemetry.tool_counts
        assert "compare_evidence" in report.telemetry.tool_counts
        assert "save_research_report" in report.telemetry.tool_counts

        # Verify progress callbacks
        assert any(event[0] == "start" for event in progress_events)
        assert any(event[0] == "complete" for event in progress_events)

