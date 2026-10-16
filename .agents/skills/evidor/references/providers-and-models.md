# Providers & Models Reference

Evidor provides a clean, modern interface for multi-provider conversational agents and stateless LLM calls. Evidor embraces breaking changes to maintain clean, modern integrations with zero legacy fallbacks.

---

## Active Recommended Models & Provider Matrix

| Provider Class | Underlying SDK | Env Variable | Active Recommended Models | Outdated / Restricted | Extra Requirement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GeminiProvider` | `google-genai` | `GEMINI_API_KEY` | `gemini-3.8-flash`<br>`gemini-3.5-flash-lite`<br>`gemini-3.1-pro-preview`<br>`gemini-3.8-live` | `gemini-2.5-flash`<br>`gemini-2.5-pro`<br>`gemini-2.0-flash`<br>`gemini-1.5-*` | `pip install "evidor[gemini]"` |
| `OpenAIProvider` | `openai` | `OPENAI_API_KEY` | `gpt-6-astra`<br>`gpt-6.1-sol`<br>`gpt-6-luna`<br>`gpt-4.1`<br>`gpt-4.1-mini` | `gpt-4o`<br>`gpt-4o-mini`<br>`o1`, `o3-mini`<br>`gpt-4.5-preview`<br>`gpt-3.5-turbo` | `pip install "evidor[openai]"` |
| `AnthropicProvider` | `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-7-sonnet-20250219` | `claude-3-5-sonnet-20241022`<br>`claude-3-haiku-20240307` | `pip install "evidor[anthropic]"` |

---

## 1. Google Gemini (`GeminiProvider`)

- **Modern SDK**: Uses `google-genai` (`from google import genai`, `from google.genai import types`). Legacy packages (`google-generativeai` or `@google/generative-ai`) are strictly prohibited.
- **Interactions API**: Exclusively uses Google GenAI's Interactions API (`client.interactions.create`) with server-side context chaining via `previous_interaction_id`.
- **Thinking Configuration**:
  - In Gemini 3.x, configure reasoning using `thinking_level` string enum (`"minimal"`, `"low"`, `"medium"`, `"high"`):
    ```python
    from evidor import GeminiProvider, GeminiThinkingConfig

    # Shorthand string enum
    provider = GeminiProvider(model="gemini-3.8-flash", thinking="high")

    # Shorthand bool (False disables thinking; True enables dynamic thinking)
    provider = GeminiProvider(model="gemini-3.8-flash", thinking=False)

    # Typed configuration
    provider = GeminiProvider(
        model="gemini-3.8-flash",
        thinking=GeminiThinkingConfig(thinking_level="medium", include_thoughts=True),
    )
    ```
  - **Sampling Notice**: Do **NOT** set `temperature`, `top_p`, or `top_k` for Gemini 3.x models; they are calibrated and optimized for default reasoning settings.

---

## 2. OpenAI (`OpenAIProvider`)

- **Modern SDK**: Uses `openai` v1.x/v3.x+ (`from openai import OpenAI`). Pre-1.0 legacy syntax is prohibited.
- **Responses API**: Exclusively uses OpenAI's modern Responses API (`client.responses.create`) with full support for `previous_response_id` chaining, conversation IDs, server-side persistence (`store=True`), and reasoning models.
- **Reasoning Configuration**:
  - Configure `reasoning={"effort": "..."}` with supported levels: `"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`:
    ```python
    from evidor import OpenAIProvider, OpenAIThinkingConfig

    # Shorthand effort string
    provider = OpenAIProvider(model="gpt-6-astra", thinking="high")

    # Typed configuration
    provider = OpenAIProvider(
        model="gpt-6-astra",
        thinking=OpenAIThinkingConfig(effort="medium", summary="detailed"),
    )
    ```

---

## 3. Anthropic (`AnthropicProvider`)

- **Messages API**: Modern Messages API with native ephemeral prompt caching (`cache_control={"type": "ephemeral"}`).
- **Thinking Configuration**:
  - Configure extended token budgets (`budget_tokens >= 1024`) or adaptive effort:
    ```python
    from evidor import AnthropicProvider, AnthropicThinkingConfig

    # Shorthand int (budget tokens cap >= 1024)
    provider = AnthropicProvider(model="claude-3-7-sonnet-20250219", thinking=4096)

    # Typed configuration
    provider = AnthropicProvider(
        model="claude-3-7-sonnet-20250219",
        thinking=AnthropicThinkingConfig(budget_tokens=8192),
    )
    ```

---

## Low-Level Direct Generation (`GenerationRequest`)

You can invoke model providers directly without session history or agent state using `GenerationRequest` and `GenerationResponse`:

```python
from evidor import GenerationRequest, Message, GeminiProvider

provider = GeminiProvider(model="gemini-3.8-flash")

request = GenerationRequest(
    messages=[
        Message(role="system", content="You are a senior systems architect."),
        Message(role="user", content="Contrast Kafka and RabbitMQ in two bullet points."),
    ],
    # Optional parameters:
    # previous_response_id="resp_123", # Prior response ID to chain
    # conversation_id="conv_abc",      # External conversation resource ID
    # store=True,                      # Whether provider should persist completion
    # system="Developer instructions",  # Top-level system prompt
)
response = provider.generate(request)

print(f"Response (ID: {response.id}):\n{response.text}")
print("Model used:", response.model)

if response.thinking:
    print(f"Internal reasoning:\n{response.thinking}")

if response.usage:
    print(
        f"Tokens: prompt={response.usage.prompt_tokens}, "
        f"completion={response.usage.completion_tokens}, "
        f"cached={response.usage.cached_tokens}, "
        f"reasoning={response.usage.reasoning_tokens}"
    )
```

### Single-Prompt Shortcut

```python
request = GenerationRequest(prompt="Summarize the theory of relativity.")
response = provider.generate(request)
```

---

## Custom Model Providers (`ModelProvider` Protocol)

Any class implementing the `ModelProvider` protocol (or subclassing `BaseProvider`) can be plugged into `Agent`:

```python
from evidor import GenerationRequest, GenerationResponse, Agent
from evidor.agent.providers.base import BaseProvider

class CustomMockProvider(BaseProvider):
    """Deterministic mock provider for testing agents offline."""

    def __init__(self, model: str = "mock-model") -> None:
        super().__init__(model=model)

    @property
    def context_strategy(self) -> str:
        return "stateless"  # or 'response_chain' / 'interaction_chain' / 'prompt_caching'

    def with_model(self, model: str) -> "CustomMockProvider":
        return CustomMockProvider(model=model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        last_msg = request.messages[-1].content if request.messages else request.prompt
        return GenerationResponse(
            id="mock_resp_1",
            text=f"Echo: {last_msg}",
            model=self._model,
        )

# Use with Agent
agent = Agent(CustomMockProvider())
response = agent.send("Testing custom provider")
assert response.text == "Echo: Testing custom provider"
```
