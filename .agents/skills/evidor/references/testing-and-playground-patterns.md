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
│   ├── 03_context_chaining/
│   │   └── chaining_benchmark.py
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

When testing agent coordination, tool calling loops, or context chaining, you do not need to consume paid API credits. Implement a mock provider:

```python
from evidor import Agent, GenerationRequest, GenerationResponse, ToolCall, tool
from evidor.agent.providers.base import BaseProvider

class ScriptedMockProvider(BaseProvider):
    """Mock provider that returns scripted responses."""

    def __init__(self, responses: list[GenerationResponse], context_strategy: str = "stateless") -> None:
        super().__init__(model="scripted-mock")
        self._strategy = context_strategy
        self.responses = list(responses)
        self.call_history: list[GenerationRequest] = []

    @property
    def context_strategy(self) -> str:
        return self._strategy

    def with_model(self, model: str) -> "ScriptedMockProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.call_history.append(request)
        if not self.responses:
            return GenerationResponse(id="fallback", text="Default fallback response", model=self._model)
        return self.responses.pop(0)

# Example: Testing tool invocation loop deterministically
@tool
def lookup_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny and 25C in {city}"

mock_responses = [
    # Turn 1: Model asks to call tool
    GenerationResponse(
        id="resp_1",
        text="",
        tool_calls=(
            ToolCall(id="call_1", name="lookup_weather", arguments={"city": "Tokyo"}),
        ),
    ),
    # Turn 2: Model provides final answer after seeing tool output
    GenerationResponse(id="resp_2", text="The weather in Tokyo is sunny and 25C."),
]

mock_provider = ScriptedMockProvider(mock_responses)
agent = Agent(mock_provider, tools=[lookup_weather])
response = agent.send("What's the weather in Tokyo?")

assert response.text == "The weather in Tokyo is sunny and 25C."
assert len(agent.messages) == 4  # system + user + assistant(tool_call) + tool(result)
```

---

## Pattern 2: Asserting with `InMemorySink`

Use `InMemorySink` in tests to verify token metrics, span latencies, and tool invocations:

```python
import pytest
from evidor import Agent, InMemorySink, GeminiProvider
from evidor.telemetry import LLMCallEndEvent, ToolCallEndEvent, ErrorEvent

def test_agent_telemetry():
    sink = InMemorySink()
    agent = Agent(GeminiProvider("gemini-3.8-flash"), telemetry=sink)
    
    response = agent.send("Say hi in exactly one word.")
    agent.close()

    llm_calls = sink.filter(LLMCallEndEvent)
    assert len(llm_calls) >= 1
    assert llm_calls[0].status == "ok"
    assert llm_calls[0].duration_ms > 0
    assert len(sink.filter(ErrorEvent)) == 0
```

---

## Pattern 3: Testing Provider-Managed Context Chaining

To test conversation state chaining across multiple turns without real LLM credits, verify `agent.strategy` and `agent.last_response_id`:

```python
from evidor import Agent, GenerationRequest, GenerationResponse
from evidor.agent.providers.base import BaseProvider

class StatefulChainMockProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(model="test-chain-model")
        self.requests: list[GenerationRequest] = []
        self._turn = 0

    @property
    def context_strategy(self) -> str:
        return "response_chain"

    def with_model(self, model: str) -> "StatefulChainMockProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        self._turn += 1
        return GenerationResponse(
            id=f"resp_{self._turn}",
            text=f"Turn {self._turn} response",
            model=self._model,
        )

provider = StatefulChainMockProvider()
agent = Agent(provider, system_prompt="Test Assistant")

assert agent.strategy == "response_chain"
assert agent.last_response_id is None

# Turn 1
agent.send("First message")
assert agent.last_response_id == "resp_1"
assert provider.requests[0].previous_response_id is None
assert provider.requests[0].system == "Test Assistant"

# Turn 2: Chained via previous_response_id
agent.send("Second message")
assert agent.last_response_id == "resp_2"
assert provider.requests[1].previous_response_id == "resp_1"

# Clear history and verify chain resets
agent.clear_history()
assert agent.last_response_id is None
assert len(agent.messages) == 1  # Only system prompt retained
```

---

## Pattern 4: Testing Tool Calling with Response ID Chaining

Verify that tool-calling iterations properly carry the response ID from step to step:

```python
from evidor import Agent, GenerationRequest, GenerationResponse, ToolCall, tool
from evidor.agent.providers.base import BaseProvider

@tool
def calculate_tax(amount: float) -> str:
    """Calculate 10% tax."""
    return f"{amount * 0.1:.2f}"

class MultiStepChainProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(model="multi-step-model")
        self.step = 0
        self.requests: list[GenerationRequest] = []

    @property
    def context_strategy(self) -> str:
        return "response_chain"

    def with_model(self, model: str) -> "MultiStepChainProvider":
        return self

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        self.step += 1
        if self.step == 1:
            return GenerationResponse(
                id="step_1",
                text="",
                model=self._model,
                tool_calls=(ToolCall(id="tc_1", name="calculate_tax", arguments={"amount": 100.0}),),
            )
        return GenerationResponse(
            id="step_2",
            text="Total tax is 10.00",
            model=self._model,
        )

provider = MultiStepChainProvider()
agent = Agent(provider, tools=[calculate_tax])

response = agent.send("Calculate tax for 100")
assert response.text == "Total tax is 10.00"
assert agent.last_response_id == "step_2"

# Second turn during tool resolution used step_1 as previous_response_id
assert provider.requests[1].previous_response_id == "step_1"
assert provider.requests[1].messages[0].role == "tool"
```
