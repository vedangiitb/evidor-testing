# Tools & Model Context Protocol (MCP) Reference

Evidor offers a comprehensive function-calling execution loop and deep support for the Model Context Protocol (MCP).

---

## 1. Defining Custom Tools with `@tool`

The `@tool` decorator converts standard Python functions into provider-neutral tool schemas.

```python
from evidor import tool

# Basic tool: schema derived from type annotations and docstring
@tool
def get_user_profile(user_id: str, include_billing: bool = False) -> dict:
    """Fetch user account metadata and permissions.

    Args:
        user_id: Unique string identifier for the user.
        include_billing: Whether to include subscription and billing status.
    """
    return {
        "user_id": user_id,
        "name": "Alice Smith",
        "plan": "enterprise" if include_billing else "standard",
    }

# Explicit name, description, and timeout overrides
@tool(name="calc_add", description="Add two floating point numbers together.", timeout=5.0)
def add_numbers(x: float, y: float) -> float:
    return x + y
```

### Async Tools
Async functions execute on the caller's event loop when invoked via `await agent.send_async(...)`:

```python
import asyncio
from evidor import tool

@tool(timeout=10.0)
async def query_remote_service(endpoint: str) -> dict:
    """Fetch status from a remote API."""
    await asyncio.sleep(0.05)
    return {"status": "healthy", "endpoint": endpoint}
```

### Manual `Tool` Creation
For programmatic tools or dynamic schemas without fixed Python function signatures:

```python
from evidor import Tool

custom_tool = Tool(
    name="execute_sql",
    description="Run read-only analytical query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL query string"}
        },
        "required": ["query"],
    },
    func=lambda query: f"Rows returned for {query}",
    timeout=5.0,
)
```

---

## 2. Built-in Toolkits

Evidor includes zero-setup built-in tools that can be attached directly to agents:

### Calculator & Current Time

```python
from evidor import Agent, GeminiProvider, calculator, get_current_time

agent = Agent(
    GeminiProvider("gemini-3.8-flash"),
    tools=[calculator, get_current_time],
)

response = agent.send("What is the current time in UTC and what is 1432 multiplied by 87?")
```

- `calculator`: Evaluates basic arithmetic expressions safely without executing raw Python code.
- `get_current_time`: Returns current ISO 8601 timestamp; accepts an optional IANA timezone string (e.g., `"America/New_York"`).

### Scoped Filesystem Tools (`filesystem_tools`)

Creates 6 tools scoped strictly within a designated directory root:
- `list_files(path=".")`
- `read_file(path)`
- `search_files(query, path=".")`
- `create_file(path, content)`
- `write_file(path, content)`
- `delete_file(path)`

```python
from evidor import Agent, GeminiProvider, filesystem_tools

agent = Agent(
    GeminiProvider("gemini-3.8-flash"),
    tools=[
        *filesystem_tools(
            root_dir="./sandbox_data",
            max_file_bytes=500_000,   # Max bytes read or written per call
            max_results=50,           # Max items in listing/search
            max_files_scanned=200,    # Max files inspected recursively
        )
    ],
)
```

### Provider-Neutral Web Search (`web_search`)

Evidor provides a unified web search tool supporting multiple providers without third-party dependencies (uses Python standard library HTTP):

```python
import os
from evidor import Agent, GeminiProvider, web_search, TavilySearchProvider, BraveSearchProvider, ExaSearchProvider

# 1. Tavily (requires TAVILY_API_KEY)
search_tool = web_search(TavilySearchProvider(), max_results=5)

# 2. Brave Search (requires BRAVE_SEARCH_API_KEY)
# search_tool = web_search(BraveSearchProvider(), max_results=5)

# 3. Exa Search (requires EXA_API_KEY)
# search_tool = web_search(ExaSearchProvider(), max_results=5)

agent = Agent(GeminiProvider("gemini-3.8-flash"), tools=[search_tool])
```

---

## 3. Model Context Protocol (MCP)

Evidor includes first-class integration with MCP servers, supporting local child processes (**stdio**), modern **Streamable HTTP**, and legacy **SSE**.

### A. Agent-Managed MCP Servers (`mcp_servers`)

Pass server configurations directly into `Agent(mcp_servers=...)`. The agent manages startup and shutdown automatically:

```python
from evidor import Agent, GeminiProvider, MCPServerConfig

agent = Agent(
    GeminiProvider("gemini-3.8-flash"),
    mcp_servers={
        # Stdio command
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"],
        },
        # Remote Streamable HTTP server
        "remote_api": {
            "url": "https://mcp.internal.domain/mcp",
            "transport": "streamable-http",
            "headers": {"Authorization": "Bearer YOUR_TOKEN"},
        },
    },
)

response = agent.send("List all documents in the project data folder.")
print(response.text)
agent.close()  # Cleans up MCP subprocesses
```

### B. Explicit `MCPServerConfig` Factory Methods

```python
from evidor import MCPServerConfig

# Stdio server
stdio_cfg = MCPServerConfig.stdio(
    name="my_stdio_server",
    command="uvx",
    args=["some-mcp-package"],
    env={"DEBUG": "1"},
)

# Streamable HTTP (modern remote standard)
http_cfg = MCPServerConfig.http(
    name="hosted_tools",
    url="https://api.example.com/mcp",
    headers={"x-api-key": "secret"},
)

# Legacy Server-Sent Events (SSE)
sse_cfg = MCPServerConfig.sse(
    name="legacy_service",
    url="http://localhost:8080/sse",
)
```

### C. Shared `MCPClient` Across Multiple Agents

For long-running backends (e.g. FastAPI) where multiple agents need access to the same MCP server without spawning redundant subprocesses:

```python
from evidor import Agent, MCPClient, GeminiProvider

client = MCPClient({
    "docs": {
        "command": "uvx",
        "args": ["mcp-server-docs"],
    }
})
client.connect()

# Both agents reuse the exact same client connection
researcher = Agent(GeminiProvider("gemini-3.8-flash"), mcp_client=client)
writer = Agent(GeminiProvider("gemini-3.8-flash"), mcp_client=client)

# Clean up client on application shutdown
client.close()
```

