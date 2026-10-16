# Telemetry & Observability Reference

Evidor includes a decoupled, non-blocking telemetry subsystem designed for zero overhead on the agent's critical execution path.

---

## Architecture & Design Principles

- **Background Actor Runtime**: Events emitted by the agent are placed on an in-memory queue in under 1 microsecond. Formatting, network serialization, and exports run on a background thread named `Evidor-Telemetry-Worker`.
- **Fault-Isolated**: Sinks and exporters are isolated; if an external logging service or OpenTelemetry collector times out or fails, agent execution is never interrupted or crashed.
- **Trace Context Propagation**: Evidor tracks parent-child span hierarchy across synchronous and asynchronous operations using `contextvars`.
- **Zero-Dependency Sinks**: `ConsoleSink` and `InMemorySink` work out of the box with `pip install evidor`.

```text
agent.run                     [openinference.span.kind = "AGENT"]
  ├── llm.gemini-3.8-flash    [openinference.span.kind = "LLM", status = "ok"]
  └── tool.calculator         [openinference.span.kind = "TOOL"]
```

---

## 1. Built-in Sinks

### ConsoleSink (Developer Visibility)

Outputs formatted structured events to standard error or standard output:

```python
import sys
from evidor import Agent, GeminiProvider, ConsoleSink

sink = ConsoleSink(stream=sys.stdout)
agent = Agent(GeminiProvider("gemini-3.8-flash"), telemetry=sink)
agent.send("Hello agent!")
agent.close()
```

### InMemorySink (Testing & Offline Assertions)

Captures all telemetry events in an internal list for test assertions:

```python
from evidor import Agent, GeminiProvider, InMemorySink
from evidor.telemetry import (
    AgentRunStartEvent,
    AgentRunEndEvent,
    LLMCallStartEvent,
    LLMCallEndEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ErrorEvent,
)

sink = InMemorySink()
agent = Agent(GeminiProvider("gemini-3.8-flash"), telemetry=sink)
response = agent.send("What is 10 + 20?")
agent.close()

# Filter specific events
llm_events = sink.filter(LLMCallEndEvent)
print(f"Number of LLM calls: {len(llm_events)}")
if llm_events[0].token_usage:
    print(f"Total tokens: {llm_events[0].token_usage.total_tokens}")
    print(f"Prompt tokens: {llm_events[0].token_usage.prompt_tokens}")
    print(f"Completion tokens: {llm_events[0].token_usage.completion_tokens}")

# Check error events
errors = sink.filter(ErrorEvent)
assert len(errors) == 0
```

---

## 2. Observability Adapters

### OpenTelemetry (`evidor[otel]`)

Converts agent events into OpenTelemetry spans compliant with OpenInference conventions:

```bash
pip install "evidor[otel]"
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from evidor import Agent, GeminiProvider, OpenTelemetrySink

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

sink = OpenTelemetrySink(tracer_provider=provider)
agent = Agent(GeminiProvider("gemini-3.8-flash"), telemetry=sink)
agent.send("Explain quantum superposition.")
agent.close()
```

### Langfuse (`evidor[langfuse]`)

Traces agent runs, prompt versions, and token usage into Langfuse:

```bash
pip install "evidor[langfuse]"
```

```python
from evidor import Agent, GeminiProvider, LangfuseSink

# Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env
sink = LangfuseSink()
agent = Agent(GeminiProvider("gemini-3.8-flash"), telemetry=sink)
agent.send("Write a test plan.")
agent.close()  # Flushes all traces to Langfuse backend
```

### Arize Phoenix (`evidor[phoenix]`)

Export traces to Arize Phoenix (local or cloud) with native OpenInference UI:

```bash
pip install "evidor[phoenix]"
```

```python
from evidor import Agent, GeminiProvider, PhoenixSink

sink = PhoenixSink(
    endpoint="http://localhost:6006/v1/traces",
    project_name="evidor-experiments",
)
agent = Agent(GeminiProvider("gemini-3.8-flash"), telemetry=sink)
agent.send("Analyze customer feedback.")
agent.close()
```

### Prometheus Metrics (`evidor[prometheus]`)

Exposes metrics counters, gauges, and latency histograms:

```bash
pip install "evidor[prometheus]"
```

```python
from evidor import Agent, GeminiProvider, PrometheusSink

prom_sink = PrometheusSink()
agent = Agent(GeminiProvider("gemini-3.8-flash"), telemetry=prom_sink)
agent.send("Summarize metrics.")

# Scrape / export formatted Prometheus text
exposition = prom_sink.export_text()
print(exposition)
```

Exported metrics include:
- `evidor_agent_runs_total`
- `evidor_llm_calls_total`
- `evidor_tokens_total`
- `evidor_tool_calls_total`
- `evidor_errors_total`

---

## 3. Privacy & PII Protection (Metadata-Only Mode)

In sensitive or regulated environments where user prompts and LLM answers cannot be sent to third-party tracing services, enable metadata-only mode with `capture_content=False`:

```python
from evidor import Agent, GeminiProvider, OpenTelemetrySink, LangfuseSink

sink = OpenTelemetrySink(capture_content=False)
agent = Agent(GeminiProvider("gemini-3.8-flash"), telemetry=sink)
```

| Field / Attribute | `capture_content=True` | `capture_content=False` |
| :--- | :--- | :--- |
| Trace / Span / Run IDs | Preserved | Preserved |
| Timestamps & Durations | Preserved | Preserved |
| Token Usage Counts | Preserved | Preserved |
| Provider & Model Names | Preserved | Preserved |
| Tool Names & Statuses | Preserved | Preserved |
| Exceptions & Error Types | Preserved | Preserved |
| Prompts & Message Content | Recorded | **Omitted / Redacted** |
| Model Output Text | Recorded | **Omitted / Redacted** |
| Tool Arguments & Results | Recorded | **Omitted / Redacted** |

---

## 4. Multiple Sinks & Custom `TelemetryRuntime`

Pass multiple sinks directly to `Agent(telemetry=[sink1, sink2])` or customize queue parameters:

```python
from evidor import Agent, GeminiProvider, ConsoleSink, InMemorySink
from evidor.telemetry import TelemetryRuntime

runtime = TelemetryRuntime(
    sinks=[ConsoleSink(), InMemorySink()],
    queue_size=5000,                # Max buffered events before dropping
    batch_size=50,                  # Events dispatched per flush cycle
    flush_interval_seconds=0.25,    # Max delay before dispatch
    overflow_strategy="drop_oldest", # "drop_oldest" or "drop_newest"
)

agent = Agent(GeminiProvider("gemini-3.8-flash"), telemetry=runtime)
```
