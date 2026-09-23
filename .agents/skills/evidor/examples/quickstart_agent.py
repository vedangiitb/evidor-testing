"""Quickstart example demonstrating basic multi-turn conversation with GeminiProvider."""

import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

from evidor import Agent, GeminiProvider

def main() -> None:
    # 1. Initialize the provider (reads GEMINI_API_KEY from environment)
    provider = GeminiProvider(model="gemini-2.5-flash")

    # 2. Create the agent with an optional system prompt
    agent = Agent(
        provider=provider,
        system_prompt="You are a knowledgeable and concise technical tutor.",
    )

    # 3. First turn
    print("User: Hello! In one short sentence, what is an event loop?")
    resp1 = agent.send("Hello! In one short sentence, what is an event loop?")
    print(f"Agent: {resp1.text}\n")

    # 4. Multi-turn context memory
    print("User: Give a code example in Python of what you just explained.")
    resp2 = agent.send("Give a code example in Python of what you just explained.")
    print(f"Agent: {resp2.text}\n")

    # 5. Inspect retained history
    print(f"Total turns in history: {len(agent.messages)}")

if __name__ == "__main__":
    main()

