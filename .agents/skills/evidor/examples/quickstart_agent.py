"""Quickstart example demonstrating basic multi-turn conversation with GeminiProvider and native context chaining."""

import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

from evidor import Agent, GeminiProvider

def main() -> None:
    # 1. Initialize the provider (reads GEMINI_API_KEY from environment)
    # Uses Gemini 3.8 Flash with native Interactions API
    provider = GeminiProvider(model="gemini-2.5-flash")

    # 2. Create the agent with an optional system prompt
    agent = Agent(
        provider=provider,
        system_prompt="You are a knowledgeable and concise technical tutor.",
    )

    print(f"Agent Context Strategy: {agent.strategy}")  # 'interaction_chain'

    # 3. First turn
    print("User: Hello! In one short sentence, what is an event loop?")
    resp1 = agent.send("Hello! In one short sentence, what is an event loop?")
    print(f"Agent (ID: {resp1.id}): {resp1.text}\n")
    print(f"Last Response ID: {agent.last_response_id}\n")

    # 4. Multi-turn context memory (chained via provider-native interaction ID)
    print("User: Give a code example in Python of what you just explained.")
    resp2 = agent.send("Give a code example in Python of what you just explained.")
    print(f"Agent (ID: {resp2.id}): {resp2.text}\n")

    # 5. Inspect retained history
    print(f"Total turns in history: {len(agent.messages)}")
    print(f"Final Response ID: {agent.last_response_id}")

if __name__ == "__main__":
    main()


