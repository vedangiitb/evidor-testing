"""Example demonstrating native asynchronous conversations with send_async() and async tools."""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from evidor import Agent, GeminiProvider, tool

@tool(timeout=5.0)
async def fetch_stock_quote(symbol: str) -> dict:
    """Fetch current price and trading volume for a ticker symbol.

    Args:
        symbol: Uppercase stock ticker, e.g. 'AAPL' or 'GOOGL'.
    """
    # Simulate async network I/O
    await asyncio.sleep(0.1)
    mock_data = {
        "AAPL": {"price": 225.50, "currency": "USD", "volume": "42M"},
        "GOOGL": {"price": 182.20, "currency": "USD", "volume": "21M"},
    }
    return mock_data.get(symbol.upper(), {"error": f"Symbol {symbol} not found."})

async def main() -> None:
    async with Agent(
        GeminiProvider(model="gemini-3.8-flash"),
        tools=[fetch_stock_quote],
        system_prompt="You are a financial market assistant.",
    ) as agent:
        print("Sending async query...")
        response = await agent.send_async("What is the current stock price of AAPL?")
        print(f"Agent response: {response.text}")

if __name__ == "__main__":
    asyncio.run(main())

