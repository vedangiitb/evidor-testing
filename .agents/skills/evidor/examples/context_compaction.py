"""Example demonstrating automatic conversation compaction and background summarization."""

from dotenv import load_dotenv

load_dotenv()

from evidor import Agent, GeminiProvider

def main() -> None:
    # Set low context_window and max_messages to trigger compaction quickly
    agent = Agent(
        GeminiProvider(model="gemini-2.5-flash"),
        system_prompt="You are a helpful assistant with high attention to detail.",
        context_window=1000,    # Token limit before compaction
        max_messages=4,         # Keep max 4 messages before compaction
    )

    print("Step 1: Introducing a critical fact...")
    agent.send("My project name is 'Operation Phoenix' and the secret key is 7734.")

    print("Step 2: Adding conversational turns to fill context...")
    agent.send("What is the speed of sound in air?")
    agent.send("How far is the Moon from Earth?")
    agent.send("What is Archimedes' principle?")

    # Inspect current messages to check for summary presence
    print("\n--- History State ---")
    summary_found = False
    for msg in agent.messages:
        if msg.is_summary:
            summary_found = True
            print(f"[COMPACTED SUMMARY]: {msg.content}")
        else:
            print(f"[{msg.role}]: {msg.content[:60]}...")

    print(f"\nCompaction triggered: {summary_found}")

    # Verify that the initial fact survived compaction via the summary
    print("\nStep 3: Querying the compacted knowledge...")
    resp = agent.send("What was my secret key and project name?")
    print(f"Agent: {resp.text}")

if __name__ == "__main__":
    main()

