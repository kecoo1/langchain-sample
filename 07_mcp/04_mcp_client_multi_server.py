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
            },
            "weather": {
                "transport": "stdio",
                "command": "uv",
                "args": ["run", "python", "/home/kecoo/aiproject/project/langchain-sample/07_mcp/02_mcp_server_weather.py"],
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools from MCP servers:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(llm, tools)

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's 15 * 8 - 20?"}]}
    )
    print(f"\nMath Response: {math_response['messages'][-1].content}")

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's the weather in Beijing?"}]}
    )
    print(f"\nWeather Response: {weather_response['messages'][-1].content}")

    combined_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "If today's temperature in Shanghai is 30°C and it drops by 5°C tomorrow, what will be the temperature tomorrow?"}]}
    )
    print(f"\nCombined Response: {combined_response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())