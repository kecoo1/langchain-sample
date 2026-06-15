"""并行工具调用 —— Agent 同时调用多个工具提升效率。

模型原生支持一次返回多个 tool_calls，LangChain 可同时执行它们。
对于互不依赖的查询（查天气 + 查股价 + 查新闻），并行比串行快数倍。

本文件覆盖 3 个场景：
1. 基础并行：Agent 同时调用多个工具
2. 依赖任务：有的任务可并行，有的串行
3. 批量处理：大量同类数据处理
"""

import asyncio
import time

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 示例 1：基础并行 —— Agent 同时查询多个信息
# ==============================================================================
print("=== 示例 1: 基础并行工具调用 ===")


@tool
def search_flight(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a given date.

    Args:
        origin: Departure city code
        destination: Arrival city code
        date: Travel date in YYYY-MM-DD format
    """
    # 模拟不同的查询延迟和数据
    time.sleep(0.5)  # 模拟 API 延迟
    return f"Flight {origin}-{destination} on {date}: ${(len(origin) + len(destination)) * 50}, 1 stop"


@tool
def search_hotel(city: str, check_in: str, check_out: str) -> str:
    """Search for hotels in a city for given dates.

    Args:
        city: City name
        check_in: Check-in date
        check_out: Check-out date
    """
    time.sleep(0.3)
    price_per_night = len(city) * 20
    return f"Hotels in {city}: ${price_per_night}/night, avg rating 4.2"


@tool
def get_weather_forecast(city: str, date: str) -> str:
    """Get weather forecast for a city on a date.

    Args:
        city: City name
        date: Date in YYYY-MM-DD format
    """
    time.sleep(0.2)
    return f"Weather in {city} on {date}: Sunny, 25°C"


# 带并行调用的 Agent
tools = [search_flight, search_hotel, get_weather_forecast]
model_with_tools = model.bind_tools(tools)

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


tool_node = ToolNode(tools)


def call_model(state: AgentState) -> AgentState:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

graph = builder.compile()

print("1. 并行工具调用:")
start = time.time()
result = graph.invoke({
    "messages": [HumanMessage(content="""Plan a trip from New York to Tokyo for 2026-07-15:
1. Search for flights from NYC to TYO on that date
2. Check hotel availability in Tokyo (check-in 07-15, check-out 07-20)
3. Get weather forecast for Tokyo on 07-15
Do all three at once if possible.""")]
})
elapsed = time.time() - start

# 统计
tool_calls_count = 0
for msg in result["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls_count += len(msg.tool_calls)

print(f"   总工具调用数: {tool_calls_count}")
print(f"   总计耗时: {elapsed:.2f}s")
print(f"   如果串行: ~{(0.5 + 0.3 + 0.2):.2f}s (并行节省 {(0.5+0.3+0.2)/elapsed:.1f}x)")
print(f"   最终回答: {result['messages'][-1].content[:150]}...\n")

print("   并行调用的关键条件：")
print('   1. 工具之间"无依赖"——查天气不需要查航班的结果')
print("   2. 模型原生支持一次返回多个 tool_calls")
print("   3. ToolNode 自动并行执行所有工具")
print()


# ==============================================================================
# 示例 2：依赖任务 —— 有的并行，有的串行
# ==============================================================================
print("=== 示例 2: 依赖任务处理 ===")

# 现实中任务往往有依赖关系：
#   并行组 1：搜航班 + 搜酒店（无依赖）
#          ↓
#   串行：根据航班+酒店结果生成推荐
#          ↓
#   并行组 2：查天气 + 查地图（无依赖）


@tool
def get_city_info(city: str) -> str:
    """Get general information about a city.

    Args:
        city: City name
    """
    info = {
        "tokyo": "Capital of Japan, population 14M, known for tech and culture",
        "paris": "Capital of France, population 2.1M, famous for art and cuisine",
        "new york": "Largest US city, population 8.5M, financial and cultural hub",
    }
    return info.get(city.lower(), f"Info not available for {city}")


@tool
def get_transportation(city: str) -> str:
    """Get transportation options in a city.

    Args:
        city: City name
    """
    options = {
        "tokyo": "Subway, bullet train, buses, taxis. Suica card recommended.",
        "paris": "Metro, RER, buses, Velib bikes. Navigo card recommended.",
        "new york": "Subway, buses, yellow taxis, ferries. MetroCard recommended.",
    }
    return options.get(city.lower(), f"Transport info not available for {city}")


# 约束：模型在一次调用时，可以并行调用多个工具
# 然后根据工具结果，决定下一步是回答问题还是继续调工具
tools_with_deps = [get_city_info, get_transportation, get_weather_forecast]
model_deps = model.bind_tools(tools_with_deps)

tool_node_deps = ToolNode(tools_with_deps)


def call_model_deps(state: AgentState) -> AgentState:
    response = model_deps.invoke(state["messages"])
    return {"messages": [response]}


def should_continue_deps(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


builder_deps = StateGraph(AgentState)
builder_deps.add_node("agent", call_model_deps)
builder_deps.add_node("tools", tool_node_deps)
builder_deps.set_entry_point("agent")
builder_deps.add_conditional_edges("agent", should_continue_deps, {"tools": "tools", "end": END})
builder_deps.add_edge("tools", "agent")

graph_deps = builder_deps.compile()

result = graph_deps.invoke({
    "messages": [HumanMessage(content="""I'm visiting Tokyo next week.
Can you get me info about the city, transportation, and weather forecast?
Do the independent queries in parallel.""")]
})

tool_call_rounds = 0
for msg in result["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_call_rounds += 1
        tools_used = [tc["name"] for tc in msg.tool_calls]
        print(f"   并行查询轮: {tools_used}")

print(f"   总工具调用轮数: {tool_call_rounds}")
print(f"   第一轮并行调用 {result['messages'][1].tool_calls if hasattr(result['messages'][1],'tool_calls') else '多个'} 工具")
print()

print("   依赖任务模式：")
print("   - 第一轮：并行查询无依赖信息（城市+交通+天气）")
print("   - 第二轮：基于所有信息综合回答")
print("   对比串行：需要等第一次返回才能进行下一次")
print()


# ==============================================================================
# 示例 3：批量数据处理
# ==============================================================================
print("=== 示例 3: 批量并行处理 ===")


@tool
def analyze_stock(symbol: str) -> str:
    """Get stock analysis for a given symbol.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL)
    """
    time.sleep(0.3)  # 模拟 API 延迟
    prices = {"AAPL": 178, "GOOGL": 141, "MSFT": 378, "AMZN": 178, "TSLA": 245}
    price = prices.get(symbol, 100)
    recommendation = "BUY" if price < 200 else "HOLD"
    return f"{symbol}: ${price}, recommendation: {recommendation}"


@tool
def get_company_news(symbol: str) -> str:
    """Get recent news headlines for a company.

    Args:
        symbol: Stock ticker symbol
    """
    time.sleep(0.2)
    news = {
        "AAPL": "Apple announces new MacBook Pro with M4 chip",
        "GOOGL": "Google launches Gemini 2.0 API",
        "MSFT": "Microsoft reports record cloud revenue",
    }
    return news.get(symbol, f"No recent news for {symbol}")


batch_tools = [analyze_stock, get_company_news]
batch_model = model.bind_tools(batch_tools)

tool_node_batch = ToolNode(batch_tools)


def call_batch_model(state: AgentState) -> AgentState:
    response = batch_model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue_batch(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


builder_batch = StateGraph(AgentState)
builder_batch.add_node("agent", call_batch_model)
builder_batch.add_node("tools", tool_node_batch)
builder_batch.set_entry_point("agent")
builder_batch.add_conditional_edges("agent", should_continue_batch, {"tools": "tools", "end": END})
builder_batch.add_edge("tools", "agent")

graph_batch = builder_batch.compile()

start = time.time()
result = graph_batch.invoke({
    "messages": [HumanMessage(content="""Analyze these stocks: AAPL, GOOGL, MSFT, AMZN, TSLA.
For each, get the stock analysis and recent news.
Do all of them in parallel.""")]
})
elapsed = time.time() - start

tool_calls_batch = sum(
    1 for msg in result["messages"]
    if hasattr(msg, "tool_calls") and msg.tool_calls
)

print(f"3. 批量并行处理:")
print(f"   处理 5 只股票 × 2 个工具 = 10 个工具调用")
print(f"   并行耗时: {elapsed:.2f}s")
if elapsed > 0:
    print(f"   如果串行: ~{10 * 0.25:.2f}s (并行节省 {10 * 0.25 / elapsed:.1f}x)")
print()

print("   批量处理最佳实践：")
print("   - 同类数据一次提交（10 只股票一起查）")
print("   - 模型会一次性生成所有 tool_calls")
print("   - ToolNode 用 asyncio.gather 并行执行")
print("   - 适合：批量翻译、批量摘要、批量查询")
print()


# ==============================================================================
# 教学备注：并行工具调用 —— 让 Agent 事半功倍
# ==============================================================================
# 核心问题：Agent 为什么要并行调用工具？
#   - 时间效率：5 个独立查询串行 5 秒 vs 并行 1 秒
#   - Token 效率：一次生成所有 tool_calls，减少模型调用轮数
#   - 用户体验：用户可以一次性看到多个结果
#
# 并行 vs 串行决策指南：
#
#   工具调用之间是否有依赖？
#   ├── 是（B 需要 A 的结果）
#   │   └── 串行：A → 看结果 → B
#   └── 否（所有工具互不依赖）
#       ├── 10 个以内 → 全部并行
#       └── 超过 10 个
#           ├── 分 3-5 个一组并行
#           └── 考虑批量工具合并
#
# LangChain 并行调用的实现机制：
#   1. 模型输出中包含 List[tool_calls]
#      {tool_calls: [{name: "search_flight", args: {...}}, {name: "search_hotel", args: {...}}]}
#   2. ToolNode 遍历所有 tool_calls，同时执行
#   3. 结果以 List[ToolMessage] 返回给模型
#   4. 模型整合所有工具结果，生成最终回答
#
# 常见问题：
#   Q: 模型不并行调用怎么办？
#   A: 在 system prompt 中明确提示 "可以一次调用多个工具"
#
#   Q: 并行调用太多，API 限流怎么办？
#   A: 控制每轮最大并行数（5-10 个）；使用带速率限制的工具执行器
#
#   Q: 部分工具失败怎么处理？
#   A: 成功的结果先返回，失败的单独重试
#      LangGraph 的 ToolNode 默认不中断整体流程
