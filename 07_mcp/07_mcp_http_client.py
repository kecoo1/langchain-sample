import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client("http://localhost:49883/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            result = await session.call_tool('add', {'a': 100, 'b': 200})
            print(f"\n100 + 200 = {result.structuredContent.get('result')}")

            result = await session.call_tool('multiply', {'a': 15, 'b': 25})
            print(f"15 * 25 = {result.structuredContent.get('result')}")

            result = await session.call_tool('divide', {'a': 100, 'b': 4})
            print(f"100 / 4 = {result.structuredContent.get('result')}")

if __name__ == "__main__":
    asyncio.run(main())