"""Simple script demonstrating streaming tokens with Langfuse telemetry (no tools).

Usage:
    python simple_streaming_telemetry.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Safe utf-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables
load_dotenv()
_testing_env = Path(__file__).resolve().parent.parent / "evidor-testing" / ".env"
if _testing_env.exists():
    load_dotenv(_testing_env)

from evidor import Agent, GeminiProvider, LangfuseSink, OpenAIProvider


def get_provider():
    if os.getenv("GEMINI_API_KEY"):
        return GeminiProvider(model="gemini-3.5-flash-lite")
    elif os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider(model="gpt-4.1-mini")
    else:
        raise ValueError("Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file.")


def main() -> None:
    # 1. Attach Langfuse telemetry sink (no tools attached)
    provider = get_provider()
    telemetry = LangfuseSink() if os.getenv("LANGFUSE_PUBLIC_KEY") else None

    agent = Agent(
        provider=provider,
        telemetry=telemetry,
    )

    prompt = "Tell me about quantum computing and why it matters. Give examples"
    print(f"Prompt: {prompt}\n")
    print("Streaming Response:")
    print("-" * 50)

    # 2. Stream tokens in real time
    stream = agent.send(prompt, stream=True)
    for chunk in stream:
        print(chunk, end="", flush=True)

    print("\n" + "-" * 50)

    # 3. Print streaming performance metrics
    if stream.time_to_first_token_ms is not None:
        print(f"Time to first token (TTFT): {stream.time_to_first_token_ms:.1f}ms")
    if stream.response and stream.response.usage:
        u = stream.response.usage
        print(f"Token usage: {u.prompt_tokens} prompt + {u.completion_tokens} completion = {u.total_tokens} total")

    # 4. Flush and close telemetry background worker
    agent.close()
    if telemetry:
        print("Telemetry trace successfully flushed to Langfuse.")


if __name__ == "__main__":
    main()

