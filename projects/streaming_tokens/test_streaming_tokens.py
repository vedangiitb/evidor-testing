"""Real end-to-end Python script demonstrating and testing Evidor streaming tokens,
tool execution, and Langfuse telemetry observability.

Usage:
    python test_streaming_tokens.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import time
from dotenv import load_dotenv

# Safe utf-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load credentials from .env
load_dotenv()
_testing_env = Path(__file__).resolve().parent.parent / "evidor-testing" / ".env"
if _testing_env.exists():
    load_dotenv(_testing_env)

from evidor import (
    Agent,
    GeminiProvider,
    InMemorySink,
    LangfuseSink,
    OpenAIProvider,
    tool,
)
from evidor.telemetry.events import (
    AgentRunEndEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    ToolCallEndEvent,
)


# ==============================================================================
# Define Tools
# ==============================================================================


@tool
def get_stock_price(ticker: str) -> str:
    """Retrieve current stock price and daily market change for a ticker symbol.

    Args:
        ticker: The stock ticker symbol (e.g. 'AAPL', 'GOOGL', 'NVDA').
    """
    ticker_clean = ticker.strip().upper()
    quotes = {
        "AAPL": {"price": 150.25, "change": "+1.2%", "currency": "USD"},
        "GOOGL": {"price": 175.80, "change": "+0.8%", "currency": "USD"},
        "NVDA": {"price": 128.50, "change": "+3.4%", "currency": "USD"},
        "MSFT": {"price": 420.50, "change": "-0.3%", "currency": "USD"},
    }
    data = quotes.get(
        ticker_clean,
        {"price": 100.00, "change": "0.0%", "currency": "USD"},
    )
    return (
        f"{ticker_clean}: ${data['price']:.2f} {data['currency']} "
        f"({data['change']} today, Market: Open)"
    )


@tool
def calculate_portfolio(shares: float, price_per_share: float, tax_rate: float = 0.15) -> str:
    """Calculate gross value, estimated capital gains tax, and net proceeds for a stock position.

    Args:
        shares: The number of shares owned.
        price_per_share: Current price per share.
        tax_rate: Estimated tax rate as a decimal (defaults to 0.15 for 15%).
    """
    gross = shares * price_per_share
    tax = gross * tax_rate
    net = gross - tax
    return (
        f"Position: {shares} shares @ ${price_per_share:.2f} = Gross: ${gross:,.2f} | "
        f"Estimated Tax ({tax_rate * 100:.0f}%): ${tax:,.2f} | Net Proceeds: ${net:,.2f}"
    )


# ==============================================================================
# Helper to Select Available Provider
# ==============================================================================


def get_configured_provider():
    """Select the best configured model provider based on available credentials."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        print("  Provider: Google Gemini (gemini-3.5-flash-lite)")
        return GeminiProvider(model="gemini-3.5-flash-lite")
    elif openai_key:
        print("  Provider: OpenAI (gpt-4.1-mini)")
        return OpenAIProvider(model="gpt-4.1-mini")
    else:
        print("ERROR: Neither GEMINI_API_KEY nor OPENAI_API_KEY is configured in .env.")
        sys.exit(1)


# ==============================================================================
# Part 1: Synchronous Streaming with Background Tool Calling
# ==============================================================================


def run_sync_streaming_demo(agent: Agent, mem_sink: InMemorySink) -> None:
    print("\n" + "=" * 70)
    print(" 1. SYNCHRONOUS STREAMING & TOOL EXECUTION (agent.send(..., stream=True))")
    print("=" * 70)

    query = (
        "What is the stock price of AAPL? Then calculate the portfolio value "
        "for 40 shares and summarize the position."
    )
    print(f"\n[User Request]:\n  {query}\n")
    print("[Live Stream Response]:")
    print("-" * 50)

    t_start = time.perf_counter()
    stream = agent.send(query, stream=True)

    for chunk in stream:
        # Check if the model triggered a tool call during this streaming turn
        if chunk.tool_calls:
            for tc in chunk.tool_calls:
                print(f"\n\n  [TOOL CALL DETECTED] -> {tc.name}({tc.arguments})")
                print("  [Executing tool in background...]")
                print("  [Resuming follow-up stream]:\n  ", end="", flush=True)

        # Print text tokens in real time as they arrive
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n" + "-" * 50)
    elapsed_total = (time.perf_counter() - t_start) * 1000

    # Streaming metrics
    ttft = stream.time_to_first_token_ms
    ttft_str = f"{ttft:.1f}ms" if ttft is not None else "N/A"
    print(f"\n[Streaming Performance Metrics]:")
    print(f"  - Time to First Token (TTFT): {ttft_str}")
    print(f"  - Total Elapsed Turn Duration: {elapsed_total:.1f}ms")
    if stream.response and stream.response.usage:
        u = stream.response.usage
        print(f"  - Token Usage: {u.prompt_tokens} input, {u.completion_tokens} output ({u.total_tokens} total)")


# ==============================================================================
# Part 2: Asynchronous Streaming (send_async)
# ==============================================================================


async def run_async_streaming_demo(agent: Agent) -> None:
    print("\n" + "=" * 70)
    print(" 2. ASYNCHRONOUS STREAMING (await agent.send_async(..., stream=True))")
    print("=" * 70)

    query = "Check the stock price of NVDA and tell me the daily change in one sentence."
    print(f"\n[User Request]:\n  {query}\n")
    print("[Async Live Stream Response]:")
    print("-" * 50)

    t_start = time.perf_counter()
    stream = await agent.send_async(query, stream=True)

    async for chunk in stream:
        if chunk.tool_calls:
            for tc in chunk.tool_calls:
                print(f"\n\n  [ASYNC TOOL CALL] -> {tc.name}({tc.arguments})")
                print("  [Resuming follow-up async stream]:\n  ", end="", flush=True)

        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n" + "-" * 50)
    elapsed_total = (time.perf_counter() - t_start) * 1000
    ttft = stream.time_to_first_token_ms
    ttft_str = f"{ttft:.1f}ms" if ttft is not None else "N/A"
    print(f"\n[Async Metrics]:")
    print(f"  - TTFT: {ttft_str} | Total Duration: {elapsed_total:.1f}ms")


# ==============================================================================
# Part 3: Telemetry & Langfuse Tracing Verification
# ==============================================================================


def print_telemetry_summary(mem_sink: InMemorySink, has_langfuse: bool) -> None:
    print("\n" + "=" * 70)
    print(" 3. TELEMETRY & LANGFUSE OBSERVABILITY SUMMARY")
    print("=" * 70)

    llm_starts = mem_sink.filter(LLMCallStartEvent)
    llm_ends = mem_sink.filter(LLMCallEndEvent)
    tool_ends = mem_sink.filter(ToolCallEndEvent)
    run_ends = mem_sink.filter(AgentRunEndEvent)

    print(f"  Agent Runs Completed: {len(run_ends)}")
    print(f"  Total LLM Calls:      {len(llm_ends)} (all streaming: {all(e.is_streaming for e in llm_ends)})")
    print(f"  Total Tool Calls:     {len(tool_ends)}")

    print("\n  [Detailed LLM Streaming Spans]:")
    for i, e in enumerate(llm_ends, 1):
        ttft_str = f"{e.time_to_first_token_ms:.1f}ms" if e.time_to_first_token_ms else "N/A"
        tok_str = f"{e.token_usage.total_tokens} tokens" if e.token_usage else "N/A"
        print(f"    Span {i}: model={e.model} | streaming={e.is_streaming} | TTFT={ttft_str} | duration={e.duration_ms:.1f}ms | usage={tok_str}")

    print("\n  [Detailed Tool Execution Spans]:")
    for i, e in enumerate(tool_ends, 1):
        print(f"    Tool {i}: name={e.tool_name} | status={e.status} | duration={e.duration_ms:.1f}ms")

    if has_langfuse:
        base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        print(f"\n  [Langfuse Tracing Status]:")
        print(f"    Traces dispatched and flushed to: {base_url}")
        print("    Observations include:")
        print("      * Agent Run Traces ('agent.run')")
        print("      * LLM Streaming Generations ('llm.<model>') with TTFT metrics & is_streaming flag")
        print("      * Tool Execution Spans ('tool.<name>') with parsed input/output")
    else:
        print("\n  [Langfuse]: LANGFUSE_PUBLIC_KEY / SECRET_KEY not configured. InMemorySink captured all spans.")


# ==============================================================================
# Main Orchestrator
# ==============================================================================


def main() -> None:
    print("=" * 70)
    print(" EVIDOR STREAMING TOKENS, TOOL USAGE & LANGFUSE TELEMETRY DEMO")
    print("=" * 70)

    # 1. Setup Provider & Sinks
    provider = get_configured_provider()

    mem_sink = InMemorySink()
    sinks = [mem_sink]

    lf_pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    lf_sec = os.getenv("LANGFUSE_SECRET_KEY")
    has_langfuse = bool(lf_pub and lf_sec)

    if has_langfuse:
        print(f"  Langfuse Sink: Connected (Public Key: {lf_pub[:8]}...)")
        lf_sink = LangfuseSink()
        sinks.append(lf_sink)
    else:
        print("  Langfuse Sink: Disabled (Keys not in .env)")

    # 2. Instantiate Agent with Tools and Telemetry
    agent = Agent(
        provider=provider,
        tools=[get_stock_price, calculate_portfolio],
        telemetry=sinks,
        system_prompt="You are a professional financial assistant. Use available tools when appropriate.",
    )

    try:
        # Part 1: Sync streaming turn with tool calls
        # run_sync_streaming_demo(agent, mem_sink)

        # Part 2: Async streaming turn
        asyncio.run(run_async_streaming_demo(agent))

    finally:
        # 3. Clean up agent and flush telemetry to Langfuse
        print("\n[Flushing telemetry and shutting down agent...]")
        agent.close()

    # Part 3: Telemetry inspection
    print_telemetry_summary(mem_sink, has_langfuse)

    print("\n" + "=" * 70)
    print(" STREAMING, TOOL USAGE & LANGFUSE DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
