# Agent & Context Management Reference

## The `Agent` Class

The `Agent` class coordinates model interaction, conversational history, context window compaction, tools execution, retries, and telemetry.

### Constructor Parameters

```python
from evidor import Agent, GeminiProvider, RetryConfig

agent = Agent(
    provider=GeminiProvider(model="gemini-2.5-flash"),
    system_prompt="You are a senior code reviewer.",
    tools=[...],
    mcp_servers={...},          # Or mcp_client=client
    max_tool_iterations=10,     # Max loop turns per send() before stopping
    tool_timeout=15.0,          # Per-tool execution timeout in seconds
    context_window=16_000,      # Max estimated token budget before compaction
    max_messages=50,            # Max turn count before compaction
    summarization_model="gemini-2.5-flash", # String or separate ModelProvider
    telemetry=...,              # Sink, list of sinks, or TelemetryRuntime
    retry_config=RetryConfig(max_retries=3), # Or max_retries=3 or retry_config=False
)
```

---

## Conversation Lifecycle

### 1. Advancing the Conversation (`send` and `send_async`)

```python
# Synchronous turn
response = agent.send("What is the difference between a process and a thread?")
print(response.text)

# Native async turn (executes async tools natively on caller loop)
response = await agent.send_async("Explain green threads vs OS threads.")
print(response.text)
```

### 2. Inspecting Messages

The agent retains all active conversation turns:

```python
for msg in agent.messages:
    prefix = "[SUMMARY] " if msg.is_summary else ""
    print(f"{prefix}{msg.role}: {msg.content}")
```

Each `Message` object contains:
- `role`: `"system"`, `"user"`, `"assistant"`, or `"tool"`
- `content`: Message text string
- `tool_calls`: Optional tuple of `ToolCall` objects (assistant role)
- `tool_call_id`: Associated tool call identifier (tool role)
- `name`: Tool name if role is `"tool"`
- `is_summary`: Boolean flag indicating if this message is a compacted summary

### 3. Resetting History

```python
# Clears user, assistant, and summary messages, but preserves initial system_prompt
agent.clear_history()
```

### 4. Cleanup & Context Manager

```python
# Using context managers automatically closes MCP client subprocesses and telemetry sinks
with Agent(provider, mcp_servers=...) as agent:
    agent.send(...)

# Or async context manager
async with Agent(provider, mcp_servers=...) as agent:
    await agent.send_async(...)

# Or manual close
agent.close()
```

---

## Context Window Management & Compaction

Evidor automatically tracks token estimates and turn counts via `ConversationContext`.

### Compaction Mechanics

1. **Thresholds**: Compaction triggers when:
   - Total estimated token count exceeds `context_window` (default: 16,000)
   - Or message count exceeds `max_messages` (default: 50)
2. **Preservation**:
   - The permanent `system_prompt` is **never lost**.
   - Recent message turns (the most recent conversational turns) remain verbatim.
3. **Summarization**:
   - The older turns are gathered into a prompt: `"Summarize the following conversation history concisely..."`
   - The summary provider generates a structured summary.
   - The summary is inserted as a bounded summary message: `Message(role="system", content=..., is_summary=True)`.

### Configurable Summarization Model

You can assign a lightweight, fast, or cheaper model specifically for summarization:

```python
from evidor import Agent, OpenAIProvider, GeminiProvider

# Option A: Different model string on the primary provider
agent = Agent(
    OpenAIProvider(model="gpt-4.1"),
    summarization_model="gpt-4.1-mini",
)

# Option B: An entirely distinct provider instance
agent = Agent(
    OpenAIProvider(model="gpt-4.1"),
    summarization_model=GeminiProvider(model="gemini-2.5-flash"),
)
```

---

## Automatic Retries & Exponential Backoff

Evidor implements a unified, coordinated retry engine:

### Configuration Options

```python
from evidor import Agent, GeminiProvider, RetryConfig

# 1. Simple retry count
agent = Agent(GeminiProvider("gemini-2.5-flash"), max_retries=5)

# 2. Disable retries completely
agent = Agent(GeminiProvider("gemini-2.5-flash"), max_retries=0)
# or:
agent = Agent(GeminiProvider("gemini-2.5-flash"), retry_config=False)

# 3. Fine-grained RetryConfig
agent = Agent(
    GeminiProvider("gemini-2.5-flash"),
    retry_config=RetryConfig(
        max_retries=3,              # Max attempts after initial failure
        initial_delay=0.5,          # Initial backoff delay in seconds
        max_delay=60.0,             # Maximum ceiling backoff cap
        backoff_factor=2.0,         # Exponential multiplier (e.g. 0.5s, 1.0s, 2.0s)
        jitter=True,                # Full random jitter to prevent stampedes
        retryable_exceptions=None,  # None = auto-detect transient network/provider errors
    ),
)
```

### Key Architectural Behaviors:
- **Narrow Boundary**: Retries only wrap the raw provider LLM call (`provider.generate`). Tool executions have their own error handling and do not consume LLM retry budgets.
- **SDK Retries Disabled**: Evidor explicitly disables internal retry loops in OpenAI, Anthropic, and Google GenAI SDK clients (`attempts=1`), ensuring a single unified retry budget and telemetry source of truth.
- **Rate Limit & `Retry-After` Header Handling**: When a 429 or rate-limit error includes a `Retry-After` delay, Evidor respects the requested duration (bounded by `max_delay`).
- **Async Non-Blocking Backoff**: In `send_async()`, backoff pauses with `await asyncio.sleep()`, avoiding thread blocking.

