# GitHub Issue Triage Agent (Evidor + GitHub MCP)

A minimal project designed to test the **Model Context Protocol (MCP)** capabilities of the **Evidor** agent framework using both **Remote** and **Local (stdio)** GitHub MCP servers.

---

## Supported MCP Modes

1. **Remote MCP Server (`streamable-http`)** [Default]:
   Connects to GitHub's hosted remote MCP server endpoint (`https://api.githubcopilot.com/mcp/`) over Streamable HTTP using your GitHub Personal Access Token.
2. **Local MCP Server (`stdio`)**:
   Spawns the local reference GitHub MCP server directly via `npx -y @modelcontextprotocol/server-github` as a child process communicating over stdin/stdout.

---

## Configuration

Add your credentials to `.env` in the root directory:

```env
# Required
GITHUB_TOKEN="ghp_your_personal_access_token_here"
GEMINI_API_KEY="AIzaSy..."

# Optional: Default MCP mode ('remote' or 'local')
GITHUB_MCP_MODE="remote"

# Optional: Langfuse Telemetry (automatically enabled when present)
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"
```

---

## Usage

### 1. Test MCP Tool Discovery

* **Remote MCP Server**:
  ```bash
  python -m projects.github_issue_triage.test_mcp_connection
  ```

* **Local MCP Server (`stdio` via `npx`)**:
  ```bash
  python -m projects.github_issue_triage.test_mcp_connection --local
  ```

### 2. Run Issue Triage

* **Using Remote MCP**:
  ```bash
  python -m projects.github_issue_triage.main "owner/repo" 123
  ```

* **Using Local MCP**:
  ```bash
  python -m projects.github_issue_triage.main "owner/repo" 123 --local
  ```

