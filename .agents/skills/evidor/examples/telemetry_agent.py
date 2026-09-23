"""Example demonstrating ConsoleSink and InMemorySink for observability and programmatic assertions."""

from dotenv import load_dotenv

load_dotenv()

from evidor import Agent, GeminiProvider, ConsoleSink, InMemorySink, calculator
from evidor.telemetry import LLMCallEndEvent, ToolCallEndEvent

def main() -> None:
    # 1. Attach both ConsoleSink (for live terminal output) and InMemorySink (for programmatic verification)
    console_sink = ConsoleSink()
    memory_sink = InMemorySink()

    agent = Agent(
        GeminiProvider(model="gemini-2.5-flash"),
        tools=[calculator],
        telemetry=[console_sink, memory_sink],
    )

    print("Sending calculation request...")
    response = agent.send("What is (45 * 12) + 150?")
    print(f"\nFinal response: {response.text}\n")

    # Clean up agent and flush background telemetry worker
    agent.close()

    # 2. Inspect recorded events in InMemorySink
    llm_events = memory_sink.filter(LLMCallEndEvent)
    tool_events = memory_sink.filter(ToolCallEndEvent)

    print(f"Recorded {len(llm_events)} LLM call(s)")
    for i, event in enumerate(llm_events):
        print(f"  Call {i+1}: model={event.model}, status={event.status}, duration={event.duration_ms:.1f}ms")
        if event.token_usage:
            print(f"    Tokens: prompt={event.token_usage.input_tokens}, completion={event.token_usage.output_tokens}")

    print(f"Recorded {len(tool_events)} tool call(s)")
    for event in tool_events:
        print(f"  Tool: {event.tool_name}, duration={event.duration_ms:.1f}ms")

if __name__ == "__main__":
    main()

