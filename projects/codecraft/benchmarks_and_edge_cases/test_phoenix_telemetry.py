"""Test Arize Phoenix Telemetry Integration in Codecraft."""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from evidor import GenerationRequest, GenerationResponse
import evidor.telemetry.adapters.phoenix as phoenix_adapter
from codecraft.engine import Codecraft


class MinimalMockProvider:
    def __init__(self) -> None:
        self.model = "mock-model"

    def with_model(self, model: str) -> "MinimalMockProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(text="Phoenix trace test completed.", model=self.model)


def test_codecraft_phoenix_sink_initialization():
    """Verify Codecraft initializes PhoenixSink and registers OTLP traces."""
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = Path(tmp) / "mock_project"
        sandbox.mkdir()
        (sandbox / "app.py").write_text("print('hello')\n")

        orchestrator = Codecraft(
            source=sandbox,
            provider=MinimalMockProvider(),
            enable_phoenix=True,
            phoenix_endpoint="http://localhost:6006/v1/traces",
            phoenix_project_name="test-phoenix-project",
        )

        assert orchestrator.enable_phoenix is True
        assert orchestrator.phoenix_project_name == "test-phoenix-project"
        assert orchestrator.phoenix_ui_url is not None
        assert "6006" in orchestrator.phoenix_ui_url
        assert hasattr(orchestrator, "_phoenix_sink")

        # Run refactor with mock provider
        report = orchestrator.refactor(instructions="Test message for Phoenix trace")
        orchestrator.close()

        assert report.phoenix_url is not None
        assert "test-phoenix-project" in report.phoenix_url
        assert "Phoenix trace test completed." in report.agent_summary


def test_codecraft_uses_direct_otlp_fallback_for_newer_exporters(monkeypatch):
    """Phoenix tracing remains enabled when its registration helper uses old OTEL internals."""
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = Path(tmp) / "mock_project"
        sandbox.mkdir()

        def incompatible_register(**_kwargs):
            raise AttributeError("'HTTPSpanExporter' object has no attribute '_headers'")

        monkeypatch.setattr(phoenix_adapter.potel, "register", incompatible_register)
        orchestrator = Codecraft(
            source=sandbox,
            provider=MinimalMockProvider(),
            enable_phoenix=True,
            phoenix_endpoint="http://localhost:6006/v1/traces",
            phoenix_project_name="fallback-project",
        )

        assert orchestrator.enable_phoenix is True
        assert hasattr(orchestrator, "_phoenix_sink")
        assert orchestrator._phoenix_sink._tracer_provider is not None

