"""Example demonstrating provider-managed context chaining and state preservation.

Modern LLM providers (OpenAI Responses API, Google Gemini Interactions API, Anthropic)
manage conversation state natively server-side or via prefix caching. Evidor exposes
this cleanly via Agent without error-prone client-side token counting or summarization.
"""

from dotenv import load_dotenv

load_dotenv()

from evidor import Agent, GeminiProvider, GenerationRequest, Message

def main() -> None:
    # 1. Provider-Managed Conversations via Agent
    # GeminiProvider uses the native Google GenAI Interactions API ('interaction_chain')
    provider = GeminiProvider(model="gemini-3.8-flash")
    agent = Agent(
        provider,
        system_prompt="You are a precise technical assistant.",
    )

    print(f"Context Strategy: {agent.strategy}")  # 'interaction_chain'
    print(f"Initial Last Response ID: {agent.last_response_id}\n")

    # Turn 1: Introduce a key piece of information
    print("Turn 1: Setting up context...")
    resp1 = agent.send("Remember: The deployment codename is 'Nebula-9' and the target region is 'us-east-1'.")
    print(f"Agent: {resp1.text}")
    print(f"Interaction ID: {agent.last_response_id}\n")

    # Turn 2: Continue conversation (agent chains to previous interaction ID automatically)
    print("Turn 2: Multi-turn interaction...")
    resp2 = agent.send("What are the best practices for latency reduction in AWS?")
    print(f"Agent: {resp2.text[:120]}...")
    print(f"Interaction ID: {agent.last_response_id}\n")

    # Turn 3: Verify the initial context was preserved natively by the provider
    print("Turn 3: Recalling the initial codename and region...")
    resp3 = agent.send("What were the deployment codename and target region I gave you?")
    print(f"Agent: {resp3.text}")
    print(f"Final Interaction ID: {agent.last_response_id}\n")

    # 2. Resetting / Clearing History
    print("Resetting conversation history...")
    agent.clear_history()  # Or agent.reset()
    print(f"Post-reset Last Response ID: {agent.last_response_id}")
    print(f"Preserved system prompt messages: {len(agent.messages)}\n")

    # 3. User-Managed Message History (Stateless generation)
    print("--- User-Managed Message History (Stateless) ---")
    custom_messages = [
        Message(role="system", content="You are a concise mathematician."),
        Message(role="user", content="What is the square root of 1764?"),
    ]
    stateless_resp = provider.generate(GenerationRequest(messages=custom_messages))
    print(f"Stateless Response: {stateless_resp.text}")

if __name__ == "__main__":
    main()

