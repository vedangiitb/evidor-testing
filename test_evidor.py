"""End-to-end test script showcasing the latest Evidor features."""

import asyncio
import os
from pathlib import Path

from evidor import Agent, GeminiProvider, GenerationRequest, Message, tool


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overwriting shell variables."""
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def test_basic_send(provider: GeminiProvider) -> None:
    print("\n--- 1. Basic Single-Turn Send ---")
    agent = Agent(provider, system_prompt="You are a concise test assistant.")
    response = agent.send("Reply with exactly: Evidor smoke test passed")
    print(f"Model: {response.model}")
    print(f"Response: {response.text.strip()}")


def test_multiturn_conversation(provider: GeminiProvider) -> None:
    print("\n--- 2. Multi-Turn Conversation ---")
    agent = Agent(provider, system_prompt="You are a helpful and concise assistant.")

    print("User: My favorite fruit is mango.")
    res1 = agent.send("My favorite fruit is mango.")
    print(f"Assistant: {res1.text.strip()}")

    print("User: What is my favorite fruit?")
    res2 = agent.send("What is my favorite fruit?")
    print(f"Assistant: {res2.text.strip()}")


def test_history_and_clear(provider: GeminiProvider) -> None:
    print("\n--- 3. History Inspection & Clear History ---")
    agent = Agent(provider, system_prompt="You are a helpful assistant.")
    agent.send("Hello!")

    print(f"Total messages in history: {len(agent.messages)}")
    for i, msg in enumerate(agent.messages):
        prefix = "[SUMMARY] " if msg.is_summary else ""
        print(f"  [{i}] ({prefix}{msg.role}): {msg.content}")

    print("Clearing conversation history...")
    agent.clear_history()
    print(f"Messages retained after clear: {len(agent.messages)}")
    assert len(agent.messages) == 1
    assert agent.messages[0].role == "system"
    print("Verified: System prompt preserved, conversational turns cleared.")


def test_context_compaction_and_summarization(provider: GeminiProvider, model_name: str) -> None:
    print("\n--- 4. Context Compaction & Configurable Summarization Model ---")
    agent = Agent(
        provider,
        system_prompt="You are a concise assistant.",
        max_messages=4,
        summarization_model=model_name,
    )
    print(f"Summarization provider configured with model: {agent.summary_provider.model}")

    print("Advancing turns to trigger automatic conversation compaction...")
    agent.send("Fact 1: The project name is Evidor.")
    agent.send("Fact 2: It is written in Python.")
    agent.send("Fact 3: It supports OpenAI, Anthropic, and Gemini.")

    summary_messages = [m for m in agent.messages if m.is_summary]
    if summary_messages:
        print("Compaction occurred! Generated summary message:")
        print(f"  {summary_messages[0].content}")
    else:
        print("Conversation kept within budget.")

    res = agent.send("What is the project name and language?")
    print(f"Assistant after compaction: {res.text.strip()}")


def test_direct_provider_generation(provider: GeminiProvider) -> None:
    print("\n--- 5. Low-Level GenerationRequest & Message ---")
    request = GenerationRequest(
        messages=[
            Message(role="system", content="You are a concise translator."),
            Message(role="user", content="Translate 'hello world' to French. Output only the translation."),
        ]
    )
    response = provider.generate(request)
    print(f"Direct generation response: {response.text.strip()}")


def test_agent_tool_calling(provider: GeminiProvider) -> None:
    print("\n--- 6. Tool Definition & Agent Tool Calling Loop ---")

    @tool
    def get_crypto_price(ticker: str) -> str:
        """Fetch the current price of a cryptocurrency ticker.

        Args:
            ticker: The cryptocurrency symbol, e.g. BTC, ETH, SOL.
        """
        prices = {"BTC": "$65,000", "ETH": "$3,500", "SOL": "$150"}
        return prices.get(ticker.upper(), f"Unknown ticker {ticker}")

    agent = Agent(
        provider,
        system_prompt="You are a helpful assistant. Use the available tools when asked for prices.",
        tools=[get_crypto_price],
    )
    print("User: What is the current price of BTC and SOL?")
    response = agent.send("What is the current price of BTC and SOL?")
    print(f"Assistant: {response.text.strip()}")

    tool_messages = [m for m in agent.messages if m.role == "tool"]
    print(f"Tool executions recorded in conversation: {len(tool_messages)}")
    for tm in tool_messages:
        print(f"  Tool [{tm.name}]: {tm.content}")
    assert len(tool_messages) >= 1
    print("Verified: Tool call was dispatched, executed, and fed back to agent.")


async def test_agent_async_execution(provider: GeminiProvider) -> None:
    print("\n--- 7. Async Conversation & Async Tool Execution (send_async) ---")

    @tool
    async def fetch_stock_quote(ticker: str) -> str:
        """Fetch the latest stock price asynchronously.

        Args:
            ticker: The stock ticker symbol, e.g. AAPL, GOOGL, MSFT.
        """
        await asyncio.sleep(0.05)  # Simulate async network I/O
        quotes = {"AAPL": "$220.50", "GOOGL": "$180.25", "MSFT": "$430.00"}
        return quotes.get(ticker.upper(), f"Unknown stock ticker: {ticker}")

    agent = Agent(
        provider,
        system_prompt="You are a financial assistant. Use tools when asked for stock prices.",
        tools=[fetch_stock_quote],
    )

    print("User (async): What is the current price of AAPL?")
    response = await agent.send_async("What is the current price of AAPL?")
    print(f"Assistant: {response.text.strip()}")

    tool_messages = [m for m in agent.messages if m.role == "tool"]
    print(f"Async tool executions recorded: {len(tool_messages)}")
    for tm in tool_messages:
        print(f"  Tool [{tm.name}]: {tm.content}")
    assert len(tool_messages) >= 1
    print("Verified: Native async execution (send_async) with coroutine tool succeeded.")


def main() -> None:
    load_env_file(Path(__file__).with_name(".env"))

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY in .env or in your shell before running this script.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"Running Evidor feature test suite with Gemini model: {model_name}")

    provider = GeminiProvider(model=model_name)

    test_basic_send(provider)
    test_multiturn_conversation(provider)
    test_history_and_clear(provider)
    test_context_compaction_and_summarization(provider, model_name)
    test_direct_provider_generation(provider)
    test_agent_tool_calling(provider)
    asyncio.run(test_agent_async_execution(provider))

    print("\n==========================================")
    print("All Evidor end-to-end tests completed successfully!")
    print("==========================================")


if __name__ == "__main__":
    main()
