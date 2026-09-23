"""Example demonstrating Model Context Protocol (MCP) server configuration."""

from dotenv import load_dotenv

load_dotenv()

from evidor import Agent, GeminiProvider, MCPServerConfig

def main() -> None:
    # Example configuring multiple MCP servers:
    # 1. Stdio server (e.g. SQLite, GitHub, or Filesystem MCP)
    # 2. Remote Streamable HTTP server
    agent = Agent(
        GeminiProvider(model="gemini-2.5-flash"),
        mcp_servers=[
            # Stdio server configured via MCPServerConfig
            MCPServerConfig.stdio(
                name="sqlite-mcp",
                command="uvx",
                args=["mcp-server-sqlite", "--db-path", "./test.db"],
            ),
            # Remote Streamable HTTP server
            MCPServerConfig.http(
                name="hosted-search",
                url="https://mcp.internal.example/mcp",
                headers={"Authorization": "Bearer sample-token"},
            ),
        ],
    )

    print("Configured agent with MCP servers:", list(agent.mcp_client._configs))
    # Tools from the MCP servers are automatically discovered when agent.send() is called:
    # response = agent.send("List all tables in the database.")
    # print(response.text)

    # Clean up subprocesses
    agent.close()

if __name__ == "__main__":
    main()

