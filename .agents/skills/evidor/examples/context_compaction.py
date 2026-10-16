"""Example demonstrating provider-managed context management.

NOTE: Client-side token counting and sliding window compaction have been superseded
by native provider-managed context (OpenAI Responses API chaining, Gemini Interactions API
chaining, and Anthropic prompt caching). Evidor delegates context to provider-native
mechanisms directly via Agent.
"""

from dotenv import load_dotenv

load_dotenv()

from evidor import Agent, GeminiProvider

def main() -> None:
    # Evidor delegates context directly to provider-native state via Agent:
    # - OpenAI: Chained via previous_response_id ('response_chain')
    # - Gemini: Chained via previous_interaction_id ('interaction_chain')
    # - Anthropic: Replayed with ephemeral prompt caching ('prompt_caching')
    agent = Agent(
        GeminiProvider(model="gemini-3.8-flash"),
        system_prompt="You are a helpful assistant with high attention to detail.",
    )

    print(f"Context Strategy: {agent.strategy}")

    print("Step 1: Introducing a critical fact...")
    agent.send("My project name is 'Operation Phoenix' and the secret key is 7734.")
    print(f"Interaction ID: {agent.last_response_id}")

    print("\nStep 2: Adding conversational turns...")
    agent.send("What is the speed of sound in air?")
    agent.send("How far is the Moon from Earth?")
    agent.send("What is Archimedes' principle?")

    # Inspect current message history
    print("\n--- Conversation State ---")
    for msg in agent.messages:
        print(f"[{msg.role}]: {msg.content[:60]}...")

    # Verify that the initial fact survived across turns via provider-native context
    print("\nStep 3: Querying the knowledge across turns...")
    resp = agent.send("What was my secret key and project name?")
    print(f"Agent: {resp.text}")
    print(f"Final Interaction ID: {agent.last_response_id}")

if __name__ == "__main__":
    main()
