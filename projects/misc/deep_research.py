from evidor import Agent, GeminiProvider, ThinkingConfig, TavilySearchProvider, web_search, read_webpage,MCPServerConfig
from dotenv import load_dotenv
import asyncio
from evidor.tools import tool
import os
from evidor import LangfuseSink,PhoenixSink

load_dotenv()

GEMINI_MODEL = "gemini-3.6-flash"
CONTEXT_7_MCP_URL = "https://mcp.context7.com/mcp"
CONTEXT_7_MCP_API_KEY = os.getenv("CONTEXT_7_MCP_API_KEY")
langfuse_base = os.getenv("LANGFUSE_BASE_URL")
search = web_search(TavilySearchProvider(), max_results=5)
phoenix_endpoint="http://localhost:6006/v1/traces"
phoenix_project_name="research-agent-new"

langfuse_sink = LangfuseSink(capture_thinking=True)
# phoenix_sink = PhoenixSink(capture_thinking=True,
#                            endpoint=phoenix_endpoint,
#                            project_name=phoenix_project_name
#                            )

@tool
async def get_stock_price(ticker:str):
    await asyncio.sleep(1)
    return 121.45

async def get_response():
    agent = Agent(GeminiProvider(model=GEMINI_MODEL,
                                 thinking=ThinkingConfig(effort="medium", summary="auto")),
                #   tools=[search,read_webpage],
                  mcp_servers=[
                      MCPServerConfig.http(
                        name="Context7 MCP",
                        url=CONTEXT_7_MCP_URL,
                        headers={"Authorization": f"Bearer {CONTEXT_7_MCP_API_KEY}"},
                    )
                  ],
                  telemetry=[langfuse_sink],
                  )
    print("Sending first agent message... \n \"Tell me about thought signatures from claude (refer claude docs) \" ")
    response = await agent.send_async("Tell me about thought signatures from claude (refer claude docs)", stream=True)
    async for chunk in response:
        print(chunk, end="", flush=True)
    # response = await agent.send_async("Hello bro")
    # print(f"Response: {response.text}")
    
    for msg in agent.messages:
        print(f"Role: {msg.role}\nContent:{msg.content}\nThinking: {msg.thinking}\nTool Calls: {msg.tool_calls}")
        print("--------------------------------------------------")
        
asyncio.run(get_response())