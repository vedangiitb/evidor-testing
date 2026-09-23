# Providers & Models Reference

Evidor is built on a clean, provider-agnostic abstraction. You can switch between providers or use multiple providers in the same application without changing your business logic or agent loop code.

---

## Supported Providers

| Provider Class | Underlying SDK | Env Variable | Example Models | Extra Requirement |
| :--- | :--- | :--- | :--- | :--- |
| `GeminiProvider` | `google-genai` | `GEMINI_API_KEY` | `gemini-2.5-flash`, `gemini-2.5-pro` | `pip install "evidor[gemini]"` |
| `OpenAIProvider` | `openai` | `OPENAI_API_KEY` | `gpt-4.1-mini`, `gpt-4.1`, `gpt-4o` | `pip install "evidor[openai]"` |
| `AnthropicProvider` | `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307` | `pip install "evidor[anthropic]"` |

### 1. GeminiProvider

```python
from evidor import GeminiProvider

# Automatically reads GEMINI_API_KEY from environment
provider = GeminiProvider(model="gemini-2.5-flash")

# Or pass explicitly:
provider = GeminiProvider(
    model="gemini-2.5-flash",
    api_key="your-api-key",
)
```

### 2. OpenAIProvider

```python
from evidor import OpenAIProvider

# Automatically reads OPENAI_API_KEY from environment
provider = OpenAIProvider(model="gpt-4.1-mini")

# Or pass explicitly:
provider = OpenAIProvider(
    model="gpt-4.1-mini",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",  # Optional custom base URL
)
```

### 3. AnthropicProvider

```python
from evidor import AnthropicProvider

# Automatically reads ANTHROPIC_API_KEY from environment
provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")

# Or pass explicitly:
provider = AnthropicProvider(
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",
)
```

---

## Low-Level Direct Generation

You can invoke model providers directly without session history or agent state using `GenerationRequest` and `GenerationResponse`:

```python
from evidor import GenerationRequest, Message, GeminiProvider

provider = GeminiProvider(model="gemini-2.5-flash")

# Multi-message generation
request = GenerationRequest(
    messages=[
        Message(role="system", content="You are a senior systems architect."),
        Message(role="user", content="Contrast Kafka and RabbitMQ in two bullet points."),
    ],
    temperature=0.2,       # Optional parameter passed to provider
)
response = provider.generate(request)

print(response.text)
print("Model used:", response.model)
if response.usage:
    print(f"Tokens: prompt={response.usage.input_tokens}, completion={response.usage.output_tokens}")
```

### Single-Prompt Shortcut:
```python
request = GenerationRequest(prompt="Summarize the theory of relativity.")
response = provider.generate(request)
```

---

## Custom Model Providers (`ModelProvider` Protocol)

Any class implementing the `ModelProvider` protocol can be plugged into `Agent` or `summarization_model`:

```python
from typing import Protocol
from evidor import GenerationRequest, GenerationResponse, Agent

class CustomMockProvider:
    """A deterministic mock provider for testing agents offline."""
    
    def __init__(self, model: str = "mock-model") -> None:
        self.model = model

    def with_model(self, model: str) -> "CustomMockProvider":
        return CustomMockProvider(model=model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        last_user_message = request.messages[-1].content if request.messages else request.prompt
        return GenerationResponse(
            text=f"Echo: {last_user_message}",
            model=self.model,
        )

# Use in agent
agent = Agent(CustomMockProvider())
response = agent.send("Testing custom provider")
assert response.text == "Echo: Testing custom provider"
```

