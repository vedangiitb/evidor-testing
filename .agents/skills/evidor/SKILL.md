---
name: evidor
description: >-
  Comprehensive guide, runbook, and reference for building, testing, and debugging AI agent
  projects using the Evidor Python runtime (evidor). Use this skill whenever creating new test
  projects, experiment scripts, or benchmarks with Evidor, configuring model providers
  (OpenAI, Anthropic, Gemini, or custom), managing context windows and automatic summarization,
  defining tools or integrating Model Context Protocol (MCP) servers, tuning LLM retries,
  and configuring telemetry sinks (Console, InMemory, OpenTelemetry, Langfuse, Phoenix, Prometheus).
---

# Evidor Agent Development & Testing Skill

Evidor is an open-source, provider-agnostic runtime for building AI agents in Python. It provides a clean, unified interface for conversational agents across multiple model providers, built-in context window management with automatic summarization, unified retries with exponential backoff, function calling, Model Context Protocol (MCP) integration, and decoupled non-blocking telemetry.

Official Website & Docs: [https://evidorlabs.com/](https://evidorlabs.com/)  
Source Code: `../evidor-core` (or `https://github.com/evidor/evidor-core`)

---

## Quick Reference & Cheatsheet

```python
from dotenv import load_dotenv
load_dotenv()

from evidor import (
    Agent,
    GeminiProvider,
    OpenAIProvider,
    AnthropicProvider,
    tool,
    calculator,
    get_current_time,
    filesystem_tools,
    web_search,
    TavilySearchProvider,
    RetryConfig,
    ConsoleSink,
    InMemorySink,
)

# 1. Basic Agent
agent = Agent(GeminiProvider(model="gemini-2.5-flash"))
response = agent.send("Hello world!")
print(response.text)

# 2. Agent with Tools & System Prompt
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

agent = Agent(
    GeminiProvider(model="gemini-2.5-flash"),
    system_prompt="You are a helpful math assistant.",
    tools=[multiply, calculator, get_current_time],
    tool_timeout=10.0,
    max_tool_iterations=5,
)
```

---

## Playground Setup & Environment

When working in `evidor-testing`, ensure the Python environment has `evidor` and the required provider packages installed.

### 1. Installing Evidor

To test the latest local development build directly from source:
```bash
# Editable install from local evidor-core with all optional extras
pip install -e "../evidor-core[all]"
```

Or from PyPI:
```bash
pip install "evidor[all]"
```

Available provider extras:
- `evidor[gemini]`: Google Gemini (`google-genai`)
- `evidor[openai]`: OpenAI (`openai`)
- `evidor[anthropic]`: Anthropic (`anthropic`)
- `evidor[mcp]`: Model Context Protocol (`mcp`)
- `evidor[otel]`: OpenTelemetry (`opentelemetry-api`, `opentelemetry-sdk`)
- `evidor[langfuse]`: Langfuse tracing (`langfuse`)
- `evidor[phoenix]`: Arize Phoenix (`arize-phoenix-otel`, `openinference-semantic-conventions`)
- `evidor[prometheus]`: Prometheus metrics (`prometheus-client`)
- `evidor[all]`: Installs all above adapters and dependencies

### 2. Environment Variables & Credentials

Create or verify `.env` in the workspace root with provider keys:
```env
GEMINI_API_KEY="AIzaSy..."
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
TAVILY_API_KEY="tvly-..."
```
Always call `dotenv.load_dotenv()` or ensure environment variables are present before instantiating providers.

---

## Core Capabilities & Architecture

| Feature | Primary Classes / Functions | Key Behavior |
| :--- | :--- | :--- |
| **Agent Core** | `Agent`, `Message` | Stateful multi-turn conversation, session lifecycle, synchronous `send()` and native asynchronous `send_async()`. |
| **Context Engine** | `ConversationContext`, `DEFAULT_CONTEXT_WINDOW` (16k), `DEFAULT_MAX_MESSAGES` (50) | Automatically estimates tokens and turn count. Triggers background summarization into a bounded summary message when limits are exceeded. |
| **Summarization** | `summarization_model` arg in `Agent` | Can use a faster/cheaper model or provider (e.g., `gemini-2.5-flash` or `gpt-4.1-mini`) for summarization while using a larger model for the main agent. |
| **Retries & Backoff** | `RetryConfig`, `max_retries` | Automated exponential backoff with full jitter and `Retry-After` support. Narrow retry boundary around provider API calls. SDK retries are disabled to prevent multiplied attempts. |
| **Tools & Functions** | `@tool`, `Tool`, `ToolExecutor` | Automatic JSON Schema generation from type annotations and docstrings. Async execution, timeouts, and error self-correction. |
| **Built-in Toolkits** | `calculator`, `get_current_time`, `filesystem_tools`, `web_search` | Ready-to-use zero-setup tools. Scoped directory access and neutral web search adapters (Tavily, Exa, Brave). |
| **MCP Integration** | `mcp_servers`, `mcp_client`, `MCPServerConfig`, `MCPClient` | First-class Model Context Protocol client. Connects over stdio, modern Streamable HTTP, or SSE. Dynamic tool discovery and multi-server composition. |
| **Telemetry & Observability** | `ConsoleSink`, `InMemorySink`, `OpenTelemetrySink`, `LangfuseSink`, `PhoenixSink`, `PrometheusSink` | Decoupled background actor worker (`Evidor-Telemetry-Worker`). Zero agent overhead (<1µs event queueing). Full parent-child trace trees and metadata-only privacy mode (`capture_content=False`). |

---

## Standard Workflow for Creating Test Projects

When creating a new test project or experiment in `evidor-testing`:

1. **Create an isolated project directory**:
   Keep experiments self-contained, e.g.:
   - `projects/01_gemini_basics/`
   - `projects/02_mcp_filesystem/`
   - `projects/03_telemetry_langfuse/`
2. **Configure dependencies & environment**:
   Include a local `.env` loader and verify credentials before running.
3. **Select provider & model**:
   Use `GeminiProvider(model="gemini-2.5-flash")` for fast testing or `OpenAIProvider(model="gpt-4.1-mini")` / `AnthropicProvider(model="claude-3-5-sonnet-20241022")`.
4. **Attach tools & config**:
   Define functions with standard docstrings and type annotations decorated with `@tool`.
5. **Add telemetry for visibility**:
   Attach `ConsoleSink()` for live terminal logs or `InMemorySink()` to make programmatic test assertions on tokens, tool calls, and retries.
6. **Verify and run**:
   Run with `python <script_name>.py` or write `pytest` suites asserting on `agent.messages` and `sink.events`.

---

## Detailed References

For in-depth guides and API specs, consult the reference documents:

- [Agent & Context Management](./references/core-agent-and-context.md) — Multi-turn turns, compaction, summarization models, retry configs, and async API.
- [Providers & Low-level Generation](./references/providers-and-models.md) — Supported providers, credentials, custom `ModelProvider` protocol, and `GenerationRequest`.
- [Tools & Model Context Protocol (MCP)](./references/tools-and-mcp.md) — Tool authoring, built-in tools, filesystem tools, search adapters, and MCP stdio/HTTP/SSE configs.
- [Telemetry & Observability](./references/telemetry-and-observability.md) — Sinks (Console, InMemory, OTel, Langfuse, Phoenix, Prometheus), privacy mode, and trace hierarchies.
- [Testing & Playground Patterns](./references/testing-and-playground-patterns.md) — Mock providers, zero-cost deterministic testing, pytest recipes, and playground project organization.

---

## Runnable Reference Examples

Check the [examples/](./examples/) folder for tested, end-to-end scripts:
- [Quickstart with Gemini](./examples/quickstart_agent.py)
- [Custom Tools & Built-ins](./examples/tools_agent.py)
- [Context Compaction & Summarization](./examples/context_compaction.py)
- [Model Context Protocol (MCP)](./examples/mcp_agent.py)
- [Telemetry & Assertions](./examples/telemetry_agent.py)
- [Async Agent Conversations](./examples/async_agent.py)

