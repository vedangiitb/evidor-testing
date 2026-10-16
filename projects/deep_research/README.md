# 🔬 Evidor Deep Research Agent with Arize Phoenix Telemetry & Streamlit

An autonomous, multi-turn AI deep research agent built on top of the **[Evidor](https://evidorlabs.com/)** Python runtime. The agent investigates complex technical questions across the public web using **Tavily**, reads primary sources, extracts and cross-compares factual claims, detects architectural trade-offs, and synthesizes publication-grade, rigorously cited markdown reports.

All agent operations are instrumented with **Arize Phoenix** OpenInference telemetry and visualized through a modern **Streamlit** web interface.

---

## 🌟 Key Capabilities

1. **Native Built-in Web Search**: Uses Evidor's built-in `web_search` tool backed directly by `TavilySearchProvider(api_key=...)`. No custom wrapper layers—ensures full OpenInference tool span formatting in Arize Phoenix.
2. **Deep Source Reading**: Bypasses shallow search snippets by scraping and parsing full primary articles, stripping boilerplate HTML, and analyzing authoritative technical documentation.
3. **Structured Claim Extraction & Evidence**: Calls `record_finding` to track quotes, claims, categories (`trade-off`, `architecture`, `limitation`, `adoption`), and URLs.
4. **Cross-Source Claim Comparison**: Calls `compare_claims` to explicitly contrast distinct viewpoints on critical dimensions (e.g., *Orchestration Determinism*, *State Persistence*, *Operational Complexity*).
5. **Publication-Grade Cited Reports**: Produces markdown reports featuring an Executive Summary, Comparative Matrix, Evidence Log, and complete numbered citations `[1]`, `[2]` linked to source references.
6. **Arize Phoenix Observability**: Emits OpenInference spans (`AGENT`, `LLM`, `TOOL`) via Evidor's decoupled background worker (`Evidor-Telemetry-Worker`) directly to Arize Phoenix.
7. **Interactive Streamlit UI**: User-friendly web application with real-time research progress tracking, interactive comparison tabs, token metrics, report download, and past report browsing.

---

## 🏗️ Architecture & Component Flow

```
User Query (Streamlit / CLI)
           │
           ▼
  ┌─────────────────────────────────────────────────────────┐
  │              DeepResearchEngine (Evidor)                │
  │  - Provider-Managed Context Chaining                    │
  │  - Multi-Turn Autonomous Tool Orchestration             │
  │  - Non-blocking Telemetry Actor Worker                  │
  └────────────────┬───────────────────┬────────────────────┘
                   │                   │
         Tools Attached                │ Telemetry Sinks Attached
                   │                   │
  ┌────────────────┴───────────────┐   ├────────────────────────────┐
  │ • web_search (Tavily Provider) │   │ PhoenixSink (OpenInference)│
  │   [Evidor Built-in Tool]       │   │ └── Arize Phoenix UI (6006)│
  │ • read_source (HTML Parser)    │   │                            │
  │ • record_finding               │   │ InMemorySink               │
  │ • compare_claims               │   │ └── Metrics & Test Asserts │
  │ • save_research_report         │   └────────────────────────────┘
  │ • get_current_time             │
  └────────────────────────────────┘
                   │
                   ▼
  Final Cited Markdown Report (output/reports/*.md)
```

---

## 🚀 Quickstart

### 1. Prerequisites & API Keys
Ensure dependencies are installed and API keys are set in `.env` (or exported in shell):

```bash
# In projects root or evidor-testing root (.env)
GEMINI_API_KEY="AIzaSy..."
TAVILY_API_KEY="tvly-..."
# Optional:
OPENAI_API_KEY="sk-..."
```

### 2. Launch the Streamlit Frontend

```bash
streamlit run projects/deep_research/app.py
```

The Streamlit UI will open in your default browser at `http://localhost:8501`.
- Choose your model provider (**Google Gemini** or **OpenAI**).
- Provide or verify your **Tavily API Key** in the sidebar.
- Toggle **Arize Phoenix Tracing** on/off.
- Click one of the pre-loaded example buttons or enter a custom technical query.
- Monitor live tool calls and download the final cited report (`.md`).

---

### 3. Run via Command-Line Interface (CLI)

Run research directly in your terminal:

```bash
python -m projects.deep_research.cli --query "Compare the main Python agent frameworks and identify their trade-offs for production deployments."
```

#### CLI Options:
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--query`, `-q` | Frameworks query | Research topic or comparative question |
| `--provider`, `-p` | `gemini` | `gemini` or `openai` |
| `--model`, `-m` | `gemini-3.8-flash` | Model identifier |
| `--tavily-key` | Env `TAVILY_API_KEY` | Tavily API Key |
| `--no-phoenix` | `False` | Disable Arize Phoenix telemetry |
| `--phoenix-endpoint` | `http://localhost:6006/v1/traces` | OTLP HTTP collector endpoint |
| `--max-iterations` | `25` | Maximum loop iterations |
| `--output-dir`, `-o` | `output/reports` | Destination directory for markdown reports |

---

## 📊 Arize Phoenix Telemetry & Observability

Evidor integrates natively with Arize Phoenix using OpenInference semantic standards:

1. **Standalone Server**: Start Phoenix in a terminal using:
   ```bash
   phoenix serve
   ```
2. **Dashboard URL**: Navigate to `http://localhost:6006` to inspect live traces.
3. **Span Kind Hierarchy**:
   - `[openinference.span.kind = "AGENT"]`: Entire research session run.
   - `[openinference.span.kind = "LLM"]`: Individual model reasoning turn (tracks prompt/completion tokens, latency).
   - `[openinference.span.kind = "TOOL"]`: Invocations of built-in `web_search`, `read_source`, `record_finding`, etc.
3. **Decoupled Architecture**: Telemetry events are serialized and exported in the background with zero agent overhead (<1µs enqueue).

---

## 🧪 Automated Testing & Verification

Deterministic, zero-token-cost unit and end-to-end integration tests are included:

```bash
python -m pytest projects/deep_research/tests -v
```

Test suite coverage:
- `test_tools.py`: Tests `TavilySearchProvider`, built-in `web_search` tool execution, HTML scraper, finding recorder, claim comparator, and report writer.
- `test_agent_and_engine.py`: Tests the entire multi-turn research lifecycle end-to-end using `ScriptedResearchMockProvider` (deterministic, instant execution).
- `test_phoenix_telemetry.py`: Tests Arize Phoenix telemetry sink initialization, project registration, and trace routing.
