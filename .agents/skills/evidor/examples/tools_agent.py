"""Example demonstrating custom tools with @tool and built-in tools (calculator, get_current_time)."""

from dotenv import load_dotenv

load_dotenv()

from evidor import Agent, GeminiProvider, calculator, get_current_time, tool

@tool
def convert_currency(amount: float, from_curr: str, to_curr: str) -> dict:
    """Convert an amount from one fiat currency to another.

    Args:
        amount: The monetary value to convert.
        from_curr: 3-letter currency code, e.g. 'USD'.
        to_curr: 3-letter target currency code, e.g. 'EUR'.
    """
    # Simple fixed mock rates for demonstration
    rates = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09, ("USD", "JPY"): 155.0}
    rate = rates.get((from_curr.upper(), to_curr.upper()), 1.0)
    converted = amount * rate
    return {
        "original_amount": amount,
        "from": from_curr.upper(),
        "to": to_curr.upper(),
        "rate": rate,
        "result": round(converted, 2),
    }

def main() -> None:
    provider = GeminiProvider(model="gemini-3.8-flash")

    # Equip agent with custom and built-in tools
    agent = Agent(
        provider=provider,
        tools=[convert_currency, calculator, get_current_time],
        system_prompt="You are a financial assistant. Use provided tools whenever calculations or conversions are requested.",
        tool_timeout=10.0,
        max_tool_iterations=5,
    )

    query = "What is 250 USD in EUR, and what is the current UTC time?"
    print(f"User: {query}\n")
    response = agent.send(query)
    print(f"Agent: {response.text}\n")

    # Inspect messages to see tool calls and tool responses
    print("--- Conversation Trace ---")
    for msg in agent.messages:
        role_label = f"[{msg.role.upper()}]"
        if msg.tool_calls:
            calls = [f"{tc.name}({tc.arguments})" for tc in msg.tool_calls]
            print(f"{role_label} Tool Calls: {', '.join(calls)}")
        elif msg.role == "tool":
            print(f"{role_label} Tool '{msg.name}' returned: {msg.content}")
        else:
            print(f"{role_label}: {msg.content[:80]}...")

if __name__ == "__main__":
    main()

