"""Orchestration engine for Deep Research with Arize Phoenix telemetry."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from evidor import (
    Agent,
    InMemorySink,
    ModelProvider,
    TelemetrySink,
)
from evidor.telemetry import (
    LLMCallEndEvent,
    ToolCallStartEvent,
)

try:
    from .agent import create_research_agent
    from .models import (
        ResearchReport,
        TelemetryStats,
    )
    from .search_provider import WebSearchProvider, create_search_provider
    from .tools import ResearchContext
except (ImportError, ValueError):
    from agent import create_research_agent
    from models import (
        ResearchReport,
        TelemetryStats,
    )
    from search_provider import WebSearchProvider, create_search_provider
    from tools import ResearchContext

logger = logging.getLogger("deep_research.engine")


class DeepResearchEngine:
    """Orchestrates deep research execution, Phoenix telemetry, and report synthesis."""

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        provider: ModelProvider | None = None,
        model_name: str = "gemini-3.8-flash",
        search_provider: WebSearchProvider | None = None,
        enable_phoenix: bool = True,
        phoenix_endpoint: str | None = None,
        phoenix_project_name: str = "deep-research",
        enable_langfuse: bool | None = None,
        langfuse_public_key: str | None = None,
        langfuse_secret_key: str | None = None,
        langfuse_base_url: str | None = None,
        max_tool_iterations: int = 30,
        tool_timeout: float = 30.0,
        extra_sinks: Sequence[TelemetrySink] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir or Path(__file__).parent / "output" / "reports").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.provider = provider
        self.model_name = model_name
        self.search_provider = search_provider or create_search_provider()
        self.max_tool_iterations = max_tool_iterations
        self.tool_timeout = tool_timeout

        self.enable_phoenix = enable_phoenix
        self.phoenix_endpoint = phoenix_endpoint or os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
        )
        self.phoenix_project_name = phoenix_project_name
        self.phoenix_ui_url: str | None = None
        self._phoenix_sink: Any = None

        # Langfuse telemetry configuration
        self.langfuse_public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.langfuse_base_url = (
            langfuse_base_url
            or os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST")
            or "https://cloud.langfuse.com"
        )
        if enable_langfuse is None:
            self.enable_langfuse = bool(self.langfuse_public_key and self.langfuse_secret_key)
        else:
            self.enable_langfuse = enable_langfuse
        self.langfuse_ui_url: str | None = None
        self._langfuse_sink: Any = None

        # Telemetry setup
        self._in_memory_sink = InMemorySink()
        sinks: list[TelemetrySink] = [self._in_memory_sink]
        if extra_sinks:
            sinks.extend(extra_sinks)

        if self.enable_phoenix:
            self._setup_phoenix()
            if self._phoenix_sink is not None:
                sinks.append(self._phoenix_sink)

        if self.enable_langfuse:
            self._setup_langfuse()
            if self._langfuse_sink is not None:
                sinks.append(self._langfuse_sink)

        self._sinks = sinks
        self.context: ResearchContext = ResearchContext(output_dir=self.output_dir)
        self.agent: Agent | None = None

    def _setup_phoenix(self) -> None:
        """Initialize Arize Phoenix telemetry pointing to the running phoenix server."""
        try:
            from evidor import PhoenixSink

            self._phoenix_sink = PhoenixSink(
                endpoint=self.phoenix_endpoint,
                project_name=self.phoenix_project_name,
            )

            base = self.phoenix_endpoint.replace("/v1/traces", "").rstrip("/")
            self.phoenix_ui_url = f"{base}/projects/{self.phoenix_project_name}" if "localhost" in base else base
            logger.info(f"Arize Phoenix telemetry connected for project '{self.phoenix_project_name}' ({self.phoenix_ui_url})")
        except Exception as exc:
            logger.warning(f"Failed to initialize Arize Phoenix telemetry: {exc}")
            self.enable_phoenix = False
            self.phoenix_ui_url = None

    def _setup_langfuse(self) -> None:
        """Initialize Langfuse telemetry pointing to cloud or self-hosted Langfuse."""
        try:
            from evidor import LangfuseSink

            kwargs: dict[str, Any] = {}
            if self.langfuse_public_key:
                kwargs["public_key"] = self.langfuse_public_key
            if self.langfuse_secret_key:
                kwargs["secret_key"] = self.langfuse_secret_key
            if self.langfuse_base_url:
                kwargs["base_url"] = self.langfuse_base_url

            self._langfuse_sink = LangfuseSink(**kwargs)
            self.langfuse_ui_url = self.langfuse_base_url.rstrip("/")
            logger.info(f"Langfuse telemetry connected ({self.langfuse_ui_url})")
        except Exception as exc:
            logger.warning(f"Failed to initialize Langfuse telemetry: {exc}")
            self.enable_langfuse = False
            self.langfuse_ui_url = None
            self._langfuse_sink = None

    def _get_agent(self) -> Agent:
        """Lazily initialize the Evidor Agent instance."""
        if self.agent is None:
            self.agent = create_research_agent(
                context=self.context,
                provider=self.provider,
                model_name=self.model_name,
                search_provider=self.search_provider,
                max_tool_iterations=self.max_tool_iterations,
                tool_timeout=self.tool_timeout,
                telemetry=self._sinks,
            )
        return self.agent

    def research(
        self,
        topic: str,
        *,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ResearchReport:
        """Execute a deep research run on a given topic.

        Args:
            topic: The research question or topic (e.g. 'Compare the main Python agent frameworks...').
            progress_callback: Optional callable receiving (action_name, details_dict) for real-time UI updates.

        Returns:
            Structured ResearchReport containing the final report, findings, sources, and telemetry stats.
        """
        start_time = time.time()
        agent = self._get_agent()

        if progress_callback:
            progress_callback("start", {"topic": topic, "model": self.model_name})

        prompt = (
            f"Please conduct an in-depth, rigorous research study on the following topic:\n\n"
            f"\"{topic}\"\n\n"
            f"Follow your 5-phase research workflow: search thoroughly from multiple angles, "
            f"read primary sources in detail, record key findings and cross-compare trade-offs between options, "
            f"and synthesize a publication-grade, cited report. Save the final report using save_research_report."
        )

        response = agent.send(prompt)
        output_text = response.text or ""

        elapsed_time = time.time() - start_time

        # Allow non-blocking background telemetry worker to flush events
        time.sleep(0.35)

        # Extract telemetry stats from InMemorySink
        llm_ends = self._in_memory_sink.filter(LLMCallEndEvent)
        tool_starts = self._in_memory_sink.filter(ToolCallStartEvent)

        tool_counts: dict[str, int] = {}
        for evt in tool_starts:
            tool_counts[evt.tool_name] = tool_counts.get(evt.tool_name, 0) + 1

        prompt_tokens = sum(evt.token_usage.prompt_tokens for evt in llm_ends if evt.token_usage)
        completion_tokens = sum(evt.token_usage.completion_tokens for evt in llm_ends if evt.token_usage)
        total_tokens = sum(evt.token_usage.total_tokens for evt in llm_ends if evt.token_usage)

        telemetry_stats = TelemetryStats(
            total_llm_calls=len(llm_ends),
            total_tool_calls=len(tool_starts),
            tool_counts=tool_counts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            execution_time_seconds=elapsed_time,
            phoenix_url=self.phoenix_ui_url,
            phoenix_project=self.phoenix_project_name,
            langfuse_enabled=self.enable_langfuse and self._langfuse_sink is not None,
            langfuse_url=self.langfuse_ui_url,
        )

        report_md = self.context.final_report_md or output_text
        report_path = self.context.final_report_path

        if progress_callback:
            progress_callback("complete", {
                "report_path": str(report_path) if report_path else None,
                "total_tools": len(tool_starts),
                "total_tokens": total_tokens,
                "elapsed": elapsed_time,
            })

        return ResearchReport(
            topic=topic,
            markdown_report=report_md,
            findings=list(self.context.findings),
            comparisons=list(self.context.comparisons),
            sources=list(self.context.sources.values()),
            telemetry=telemetry_stats,
            saved_file_path=str(report_path) if report_path else None,
        )

    def close(self) -> None:
        """Clean up agent, telemetry sinks, and background workers."""
        if self.agent is not None:
            self.agent.close()
            return

        # The agent owns and closes the runtime once it has been created.  If
        # this engine is closed before its first research run, no runtime
        # exists, so release directly configured telemetry resources here.
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                pass
