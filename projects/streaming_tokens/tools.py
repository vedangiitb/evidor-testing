"""Tools for testing streaming tokens and tool calling workflows in Evidor."""

from __future__ import annotations

from evidor import tool


@tool
def get_stock_price(ticker: str) -> str:
    """Retrieve the current market stock price for a given ticker symbol.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT').
    """
    ticker_clean = ticker.strip().upper()
    quotes = {
        "AAPL": "$150.25 USD (+1.2%)",
        "GOOGL": "$175.80 USD (+0.8%)",
        "MSFT": "$420.50 USD (-0.3%)",
        "AMZN": "$185.10 USD (+2.1%)",
    }
    return quotes.get(ticker_clean, f"{ticker_clean}: $100.00 USD (Simulated Quote)")


@tool
def calculate_portfolio_value(shares: float, price_per_share: float) -> str:
    """Calculate the total valuation of a share holding.

    Args:
        shares: The number of shares held.
        price_per_share: The price per individual share.
    """
    total = shares * price_per_share
    return f"Total Value: ${total:,.2f} USD ({shares} shares @ ${price_per_share:,.2f})"
