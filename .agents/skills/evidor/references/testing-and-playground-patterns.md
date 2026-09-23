# Testing & Playground Patterns Reference

This guide outlines recipes and best practices for creating experimental test projects and unit test suites in the `evidor-testing` playground.

---

## Recommended Playground Project Structure

To test various Evidor features cleanly, structure experiments as subdirectories in `projects/` or standalone test scripts:

```text
evidor-testing/
├── .env                                # API keys for real provider tests
├── .agents/skills/evidor/              # This skill
├── projects/
│   ├── 01_gemini_basics/
│   │   └── main.py
│   ├── 02_custom_tools/
│   │   └── test_tools.py
│   ├── 03_context_compaction/
│   │   └── compaction_benchmark.py
│   ├── 04_mcp_servers/
│   │   ├── weather_server.py
│   │   └── test_mcp_agent.py
│   └── 05_observability/
│       └── trace_pipeline.py
└── tests/
    └── test_offline_agents.py          # Pytest offline test suite
```

---

## Pattern 1: Zero-Cost Deterministic Testing with Mock Providers

When testing agent coordination, tool calling loops, or context compaction, you do not need to consume paid API credits. Implement a mock provider:

```python
from evidor import Agent, GenerationRequest, GenerationResponse, ToolCall, tool

class ScriptedMockProvider:
    """Mock provider that returns scripted responses."""

    def __init__(self, responses: list[GenerationResponse]) -> None:
        self.responses = list(responses)
        self.call_history: list[GenerationRequest] = []
        self.model = "scripted-mock"

    def with_model(self, model: str) -> "ScriptedMockProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.call_history.append(request)
        if not self.responses:
            return GenerationResponse(text="Default fallback response", model=self.model)
        return self.responses.pop(0)

# Example: Testing tool invocation loop deterministically
@tool
def lookup_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny and 25C in {city}"

mock_responses = [
    # Turn 1: Model asks to call tool
    GenerationResponse(
        text="",
        tool_calls=(
            ToolCall(id="call_1", name="lookup_weather", arguments={"city": "Tokyo"}),
        ),
    ),
    # Turn 2: Model provides final answer after seeing tool output
    GenerationResponse(text="The weather in Tokyo is sunny and 25C."),
]

mock_provider = ScriptedMockProvider(mock_responses)
agent = Agent(mock_provider, tools=[lookup_weather])
response = agent.send("What's the weather in Tokyo?")

assert response.text == "The weather in Tokyo is sunny and 25C."
assert len(agent.messages) == 4  # system + user + assistant(tool_call) + tool(result)
```

---

## Pattern 2: Asserting with `InMemorySink`

Use `InMemorySink` in tests to verify token metrics, retry behavior, and tool invocations:

```python
import pytest
from evidor import Agent, InMemorySink, GeminiProvider
from evidor.telemetry import LLMCallEndEvent, ToolCallEndEvent, ErrorEvent

def test_agent_telemetry():
    sink = InMemorySink()
    agent = Agent(GeminiProvider("gemini-2.5-flash"), telemetry=sink)
    
    response = agent.send("Say hi in exactly one word.")
    agent.close()

    llm_calls = sink.filter(LLMCallEndEvent)
    assert len(llm_calls) >= 1
    assert llm_calls[0].status == "ok"
    assert llm_calls[0].duration_ms > 0
    assert len(sink.filter(ErrorEvent)) == 0
```

---

## Pattern 3: Testing Context Compaction

To test conversation compaction without generating hundreds of real messages, configure tight thresholds:

```python
from evidor import Agent, GeminiProvider

agent = Agent(
    GeminiProvider("gemini-2.5-flash"),
    context_window=200,   # Very low token threshold to trigger compaction early
    max_messages=4,       # Only keep 4 messages before summarizing
)

# Advance turns
agent.send("Message 1: Remember the secret code is ALPHA-99.")
agent.send("Message 2: What is the capital of France?")
agent.send("Message 3: What is 2 + 2?")
agent.send("Message 4: Tell me a joke.")

# Inspect messages to check for summary presence
summary_messages = [m for m in agent.messages if m.is_summary]
print("Summaries generated:", len(summary_messages))
if summary_messages:
    print("Summary content:", summary_messages[0].content)

# Verify retention of original system prompt and fact recall
response = agent.send("What was the secret code mentioned in the beginning?")
print("Recall response:", response.text)
```

---

## Pattern 4: Verifying Retries on Transient Failures

You can verify that transient errors are properly retried and backoff occurs:

```python
from evidor import Agent, RetryConfig, GenerationRequest, GenerationResponse
from evidor.telemetry import InMemorySink, ErrorEvent, LLMCallEndEvent

class TransientErrorProvider:
    def __init__(self, failure_count: int = 2) -> None:
        self.failures_remaining = failure_count
        self.attempts = 0
        self.model = "transient-mock"

    def with_model(self, model: str) -> "TransientErrorProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.attempts += 1
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            # Simulate a 503 Service Unavailable / ConnectionError
            raise ConnectionError("Temporary upstream outage")
        return GenerationResponse(text="Success after retries!", model=self.model)

sink = InMemorySink()
agent = Agent(
    TransientErrorProvider(failure_count=2),
    retry_config=RetryConfig(max_retries=3, initial_delay=0.01),
    telemetry=sink,
)

response = agent.send("Test message")
agent.close()

assert response.text == "Success after retries!"
# Verify that retrying errors were recorded
retrying_errors = [e for e in sink.filter(ErrorEvent) if e.outcome == "retrying"]
assert len(retrying_errors) == 2
```

