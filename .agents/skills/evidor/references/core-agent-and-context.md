# Agent & Context Management Reference

## The `Agent` Class

The `Agent` class coordinates model interactions, conversational history, provider-native context chaining, tool execution, and telemetry.

### Constructor Parameters

```python
from evidor import Agent, GeminiProvider, OpenAIProvider, AnthropicProvider

agent = Agent(
    provider=GeminiProvider(model="gemini-3.8-flash"),
    system_prompt="You are a senior code reviewer.",
    tools=[...],                        # Custom @tool functions or Tool instances
    mcp_servers={...},                  # Or mcp_client=client
    max_tool_iterations=10,             # Max loop turns per send() before stopping
    tool_timeout=15.0,                  # Per-tool execution timeout in seconds
    previous_response_id="resp_abc123", # Optional: resume an existing response/interaction chain
    conversation_id="conv_xyz789",      # Optional: attach to an external conversation resource
    store=True,                         # Optional: instruct provider to persist completion
    telemetry=...,                      # Sink, list of sinks, or TelemetryRuntime
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

### 2. Inspecting Context State & Response IDs

Evidor delegates context management directly to provider-native mechanisms and exposes the state through properties:

```python
# The context strategy employed by the active provider:
# 'response_chain'   -> OpenAI Responses API chaining
# 'interaction_chain' -> Gemini Interactions API chaining
# 'prompt_caching'    -> Anthropic ephemeral prompt caching
# 'stateless_replay'  -> Full history replay fallback
print(agent.strategy)

# The ID of the latest provider response or interaction in this chain
print(agent.last_response_id)  # e.g., 'resp_6789' or 'int_1234'

# The optional external conversation identifier
print(agent.conversation_id)
```

### 3. Inspecting Messages

The agent records all conversation turns for telemetry, auditing, and fallback replays:

```python
for msg in agent.messages:
    print(f"[{msg.role}]: {msg.content}")
```

Each `Message` object contains:
- `role`: `"system"`, `"user"`, `"assistant"`, or `"tool"`
- `content`: Message text string
- `tool_calls`: Optional tuple of `ToolCall` objects (assistant role)
- `tool_call_id`: Associated tool call identifier (tool role)
- `name`: Tool name if role is `"tool"`
- `thinking`: Optional internal reasoning text
- `thought_blocks`: Optional raw thought block payloads with cryptographic signatures (e.g. Gemini)

### 4. Resetting History

```python
# Clears user, assistant, and tool messages, and resets last_response_id to None
# Preserves the initial system_prompt
agent.clear_history()

# reset() is an alias for clear_history()
agent.reset()
```

### 5. Cleanup & Context Managers

```python
# Using context managers automatically closes MCP client subprocesses and flushes telemetry sinks
with Agent(provider, mcp_servers=...) as agent:
    agent.send(...)

# Or async context manager
async with Agent(provider, mcp_servers=...) as agent:
    await agent.send_async(...)

# Or manual close
agent.close()
```

---

## Provider-Managed Context Management

Modern LLM providers maintain context server-side or via hardware-accelerated KV prefix caching. Rather than running client-side token counters, sliding windows, or summarization loops that break prompt caching, Evidor delegates context management to each provider's native mechanism:

| Provider | Strategy (`agent.strategy`) | Native Mechanism | How It Works |
| :--- | :--- | :--- | :--- |
| **OpenAIProvider** | `"response_chain"` | Responses API (`previous_response_id`) | Each turn sends only delta messages or tool outputs chained to `previous_response_id` (and optional `conversation_id`). OpenAI manages context server-side. |
| **GeminiProvider** | `"interaction_chain"` | Interactions API (`previous_interaction_id`) | Each turn chains to the prior `interaction_id`. Gemini manages conversation turns server-side. |
| **AnthropicProvider** | `"prompt_caching"` | Ephemeral Prompt Caching | Replays conversation turns while setting ephemeral prompt caching headers (`cache_control={"type": "ephemeral"}`). Anthropic caches the prefix with up to 90% latency/cost reduction. |
| **Custom / Base** | `"stateless_replay"` | Stateless history replay | Replays accumulated message history on each request. |

### Resuming Existing Chains

To continue a conversation across worker processes or sessions, pass the prior response ID to the constructor:

```python
# Resume OpenAI conversation
agent = Agent(
    OpenAIProvider(model="gpt-6-astra"),
    previous_response_id="resp_prev_987",
)
agent.send("Continue the analysis from our previous run.")
```

---

## User-Managed Message History (Stateless LLM Calls)

When your application handles its own persistence (e.g., PostgreSQL, Redis, custom vector store, or custom state machine), you maintain the message list and pass it directly to `provider.generate()`. Evidor never modifies, prunes, or summarises your messages behind your back:

```python
from evidor import GenerationRequest, Message, GeminiProvider

provider = GeminiProvider(model="gemini-3.8-flash")

chat_history = [
    Message(role="system", content="You are a database tuning expert."),
    Message(role="user", content="What are the optimal settings for PostgreSQL work_mem?"),
]

# Turn 1
response = provider.generate(GenerationRequest(messages=chat_history))
chat_history.append(Message(role="assistant", content=response.text))

# Turn 2
chat_history.append(Message(role="user", content="How does that impact sort performance?"))
response = provider.generate(GenerationRequest(messages=chat_history))
```
