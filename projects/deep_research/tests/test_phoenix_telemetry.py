"""Test Arize Phoenix telemetry integration in Deep Research Agent."""

from pathlib import Path
import tempfile
import pytest

from evidor import GenerationRequest, GenerationResponse, PhoenixSink
from projects.deep_research.engine import DeepResearchEngine
from projects.deep_research.search_provider import MockSearchProvider


class MinimalResearchMockProvider:
    def __init__(self) -> None:
        self.model = "mock-model"

    def with_model(self, model: str) -> "MinimalResearchMockProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(text="Phoenix trace test completed.", model=self.model)


def test_deep_research_phoenix_sink_registration():
    """Verify DeepResearchEngine initializes PhoenixSink with correct project and endpoint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = DeepResearchEngine(
            output_dir=Path(tmp_dir),
            provider=MinimalResearchMockProvider(),
            search_provider=MockSearchProvider(),
            enable_phoenix=True,
            phoenix_endpoint="http://localhost:6006/v1/traces",
            phoenix_project_name="deep-research-test",
        )

        assert engine.enable_phoenix is True
        assert engine.phoenix_project_name == "deep-research-test"
        assert engine.phoenix_endpoint == "http://localhost:6006/v1/traces"
        assert engine._phoenix_sink is not None
        assert isinstance(engine._phoenix_sink, PhoenixSink)

        report = engine.research("Test topic for Phoenix tracing")
        engine.close()

        assert report.telemetry.phoenix_url is not None
        assert "deep-research-test" in report.telemetry.phoenix_url
        assert report.telemetry.total_llm_calls >= 1

