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
    """Verify Phoenix wiring without exporting spans to a local collector.

    Phoenix uses a synchronous OTLP exporter.  A real sink would therefore try
    to reach ``localhost:6006`` for every completed span, which makes this
    registration test depend on a running Phoenix server and can make it slow.
    Export behavior is covered by the Evidor Phoenix adapter tests.
    """
    from unittest.mock import MagicMock, patch

    mock_sink = MagicMock(spec=PhoenixSink)
    with patch("evidor.PhoenixSink", return_value=mock_sink) as phoenix_sink:
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
            assert engine._phoenix_sink is mock_sink
            phoenix_sink.assert_called_once_with(
                endpoint="http://localhost:6006/v1/traces",
                project_name="deep-research-test",
            )

            engine.close()

            assert mock_sink.close.called


def test_deep_research_langfuse_sink_registration():
    """Verify DeepResearchEngine initializes LangfuseSink with credentials."""
    from unittest.mock import MagicMock, patch
    from evidor import LangfuseSink

    mock_sink = MagicMock(spec=LangfuseSink)
    with patch("evidor.LangfuseSink", return_value=mock_sink):
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DeepResearchEngine(
                output_dir=Path(tmp_dir),
                provider=MinimalResearchMockProvider(),
                search_provider=MockSearchProvider(),
                enable_phoenix=False,
                enable_langfuse=True,
                langfuse_public_key="pk-lf-test",
                langfuse_secret_key="sk-lf-test",
                langfuse_base_url="https://us.cloud.langfuse.com",
            )

            assert engine.enable_langfuse is True
            assert engine._langfuse_sink is mock_sink

            report = engine.research("Test topic for Langfuse tracing")
            engine.close()

            assert report.telemetry.langfuse_enabled is True
            assert report.telemetry.langfuse_url == "https://us.cloud.langfuse.com"
            assert report.telemetry.total_llm_calls >= 1
            assert mock_sink.close.called

