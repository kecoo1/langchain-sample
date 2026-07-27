import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "/home/kecoo/aiproject/project/langchain-sample/07_mcp/01_mcp_server_math.py"],
        cwd="/home/kecoo/aiproject/project/langchain-sample",
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print(f"Available tools: {tools_result}")

            result = await session.call_tool('add', {'a': 10, 'b': 20})
            print(f"\n10 + 20 = {result}")

            result = await session.call_tool('multiply', {'a': 5, 'b': 8})
            print(f"5 * 8 = {result}")

if __name__ == "__main__":
    asyncio.run(main())