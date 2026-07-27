import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "uv",
                "args": ["run", "python", "/home/kecoo/aiproject/project/langchain-sample/07_mcp/01_mcp_server_math.py"],
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools from MCP server:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(llm, tools)

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's (3 + 5) * 12?"}]}
    )
    print(f"\nAgent Response: {response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())