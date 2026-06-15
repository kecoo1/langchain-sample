"""实战项目：数据分析报告生成 Pipeline。

从原始数据到分析报告的完整流程：数据生成 → 清洗 → 分析 → 洞察 → 报告。
模拟电商运营数据分析场景，展示 Agent 如何自动完成数据分析师的日常工作。

=== 架构说明 ===

整体是一个三阶段的 Agent 循环，用 LangGraph StateGraph 实现：

        用户提问
           │
    ┌─ agent 节点 ←────────────┐
    │     │                    │
    │     ├─ 有 tool_call      │
    │     │   → tools 节点     │
    │     │     (并行执行工具)  │
    │     │   → 回到 agent ────┘
    │     │
    │     └─ 没有 tool_call
    │       → reporter 节点
    │         (生成报告) → END
    │
    └── should_continue 控制循环

三阶段详解：

  阶段 1 - 数据摘要（agent + tools 循环）
    Agent 收到问题后，模型决定需要调用哪些分析工具。
    这些工具是纯 Python 计算函数（不是调外部 API）：
      - load_csv_data：读 CSV，返回总行数/总金额/分类摘要
      - analyze_by_category：按品类分组聚合
      - analyze_by_city：按城市分组聚合
    关键设计："能算的不要问 LLM"。
    原始15行直接给LLM也能分析，但15万行呢？
    所以让 Python 工具先聚合，只把摘要给 LLM。

  阶段 2 - 洞察生成
    工具返回摘要后 Agent 再次被调用。
    这次没有 tool_calls（数据够了），走向 reporter。

  阶段 3 - 报告撰写（reporter 节点）
    把前面所有对话（工具返回的聚合数据 + Agent 中间分析）
    拼起来，用专门的 prompt 生成结构化报告。

  状态管理：
    messages: Annotated[list, operator.add]
      → 每次节点的返回值自动合并到列表中，不是覆盖
      → 每个节点只返回自己产出的那部分消息，框架帮你拼

  Agent 方式 vs 普通代码：

    普通代码：
      data = load_csv(path)
      summary = summarize(data)
      by_cat = analyze_category(data)
      report = llm.generate_report(summary, by_cat)

    Agent 方式（本示例）：
      Agent 自己决定：先加载 → 按品类分析 → 按城市分析
      如果发现数据异常，它可能自动追加分析步骤
      最终觉得信息够了，再去写报告

    Agent 优势：灵活 —— 自主决定工具、顺序、是否需要额外步骤
    Agent 劣势：不可预测 —— 同样输入可能走不同路径
"""

import json
import csv
import io
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 准备：模拟销售数据
# ==============================================================================
SALES_DATA = """date,product,category,units,price,total,city
2026-01-05,Wireless Mouse,Electronics,15,29.99,449.85,Beijing
2026-01-06,Mechanical Keyboard,Electronics,8,89.99,719.92,Shanghai
2026-01-07,USB-C Hub,Electronics,25,49.99,1249.75,Shenzhen
2026-01-08,Notebook Set,Stationery,40,12.99,519.60,Beijing
2026-01-09,Desk Lamp,Furniture,12,39.99,479.88,Shanghai
2026-01-10,Monitor Stand,Furniture,6,59.99,359.94,Shenzhen
2026-01-12,Wireless Mouse,Electronics,20,29.99,599.80,Guangzhou
2026-01-14,Mechanical Keyboard,Electronics,5,89.99,449.95,Beijing
2026-01-15,Notebook Set,Stationery,35,12.99,454.65,Shanghai
2026-01-16,USB-C Hub,Electronics,18,49.99,899.82,Guangzhou
2026-01-18,Desk Lamp,Furniture,10,39.99,399.90,Shenzhen
2026-01-20,Monitor Stand,Furniture,8,59.99,479.92,Beijing
2026-01-22,Wireless Mouse,Electronics,12,29.99,359.88,Shanghai
2026-01-25,Mechanical Keyboard,Electronics,10,89.99,899.90,Shenzhen
2026-01-28,Notebook Set,Stationery,50,12.99,649.50,Guangzhou
"""

CSV_PATH = "/tmp/sales_data.csv"
Path(CSV_PATH).write_text(SALES_DATA)

print("=" * 60)
print("实战项目：数据分析报告生成 Pipeline")
print("=" * 60)
print(f"模拟数据: {CSV_PATH}")
print(f"记录数: {len(SALES_DATA.strip().split(chr(10)))} 行\n")


# ==============================================================================
# 阶段 1：数据摘要（Data Profiling）
# ==============================================================================
print("--- 阶段 1: 数据摘要 ---")


@tool
def load_csv_data(filepath: str) -> str:
    """Load CSV data and return summary statistics.
    
    Args:
        filepath: Path to the CSV file
    """
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    total_revenue = sum(float(r["total"]) for r in rows)
    total_units = sum(int(r["units"]) for r in rows)
    categories = list(set(r["category"] for r in rows))
    cities = list(set(r["city"] for r in rows))
    products = list(set(r["product"] for r in rows))
    
    return json.dumps({
        "total_records": len(rows),
        "total_revenue": round(total_revenue, 2),
        "total_units": total_units,
        "categories": categories,
        "cities": cities,
        "products": products,
        "avg_order_value": round(total_revenue / len(rows), 2),
    }, indent=2)


@tool
def analyze_by_category(filepath: str) -> str:
    """Analyze sales grouped by category.
    
    Args:
        filepath: Path to the CSV file
    """
    from collections import defaultdict
    cats = defaultdict(lambda: {"revenue": 0, "units": 0})
    
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            cats[cat]["revenue"] += float(row["total"])
            cats[cat]["units"] += int(row["units"])
    
    return json.dumps({
        cat: {"revenue": round(v["revenue"], 2), "units": v["units"],
              "avg_price": round(v["revenue"] / v["units"], 2)}
        for cat, v in sorted(cats.items(), key=lambda x: -x[1]["revenue"])
    }, indent=2)


@tool
def analyze_by_city(filepath: str) -> str:
    """Analyze sales grouped by city.
    
    Args:
        filepath: Path to the CSV file
    """
    from collections import defaultdict
    cities = defaultdict(lambda: {"revenue": 0, "units": 0})
    
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            city = row["city"]
            cities[city]["revenue"] += float(row["total"])
            cities[city]["units"] += int(row["units"])
    
    return json.dumps({
        city: {"revenue": round(v["revenue"], 2), "units": v["units"]}
        for city, v in sorted(cities.items(), key=lambda x: -x[1]["revenue"])
    }, indent=2)


analysis_tools = [load_csv_data, analyze_by_category, analyze_by_city]
tool_node = ToolNode(analysis_tools)

# bind_tools 让模型"知道"有哪些工具可用
# 推理时，模型会自己判断：当前需要调工具吗？调哪个？传什么参数？
model_with_tools = model.bind_tools(analysis_tools)


class AnalysisState(TypedDict):
    # messages 用 operator.add 注解：每个节点的返回值会自动追加到列表末尾
    # 而不是覆盖——这是 LangGraph 状态管理的关键机制
    messages: Annotated[list, operator.add]
    report: str


def call_agent(state: AnalysisState) -> AnalysisState:
    """Agent 节点：让模型根据当前消息决定下一步动作。

    模型可能返回两种结果之一：
    1. 有 tool_calls → 需要调用工具（should_continue 返回 "tools"）
    2. 无 tool_calls → 可以直接回答或生成报告（走向 "generate_report"）
    """
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AnalysisState) -> str:
    """条件路由：判断 Agent 的输出是否需要继续调工具。

    这是 Agent 循环的核心逻辑：
      - 有 tool_calls → 去执行工具（tools 节点）
      - 没有 tool_calls → 信息够了，去写报告（reporter 节点）
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "generate_report"


def generate_report(state: AnalysisState) -> AnalysisState:
    """Reporter 节点：把所有分析结果汇总成结构化报告。

    把所有工具返回的聚合数据 + Agent 的中间分析结果，
    拼成上下文字符串，用专门的 prompt 生成最终报告。
    context[:3000] 做了截断，防止超出模型上下文窗口。
    """
    context = "\n".join(m.content for m in state["messages"] if hasattr(m, "content") and m.content)
    
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analyst. Write a business report based on the analysis results.
Include: executive summary, key findings (with numbers), actionable recommendations.
Use markdown formatting."""),
        ("human", "Analysis data:\n{data}"),
    ])
    chain = report_prompt | model | StrOutputParser()
    report = chain.invoke({"data": context[:3000]})
    return {"report": report}


# ─── 构建状态图 ──────────────────────────────────────────────
# 图结构（三节点循环）：
#
#   agent ──(有 tool_calls)──→ tools ──→ agent（继续循环）
#      │
#      └──(无 tool_calls)──→ reporter ──→ END
#
# 为什么 tools 要连回 agent？
#   工具执行完返回结果后，agent 需要"看到"结果，再决定下一步。
#   可能：继续调工具（需要更多数据），或者去写报告（信息够了）。
builder = StateGraph(AnalysisState)
builder.add_node("agent", call_agent)
builder.add_node("tools", tool_node)
builder.add_node("reporter", generate_report)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "generate_report": "reporter"})
builder.add_edge("tools", "agent")
builder.add_edge("reporter", END)

analysis_graph = builder.compile()

result = analysis_graph.invoke({
    "messages": [HumanMessage(content=f"""Analyze the sales data at {CSV_PATH}.
1. Load and summarize the data
2. Analyze by category (which category sells best?)
3. Analyze by city (which city is our top market?)
4. Identify key insights""")],
    "report": "",
})

print(f"\n--- 最终报告 ---\n{result['report'][:500]}...")

print("\n" + "=" * 60)
print("数据分析 Pipeline 架构")
print("=" * 60)
print("""
Pipeline 三阶段设计：

阶段 1：数据摘要 (Agent + Tools)
  为什么：直接给 LLM 原始数据太大了（几万行）。
  怎么做：Agent 调用分析工具，工具内部用 Python 聚合数据，
          只返回摘要（总金额、分类统计、城市统计）。
  原则：能算的不要问 LLM，能用代码聚合的就不要用模型。

阶段 2：洞察生成 (Agent)
  为什么：聚合后的数据还需要"人"来解释含义。
  怎么做：Agent 看到摘要后，结合业务知识给出洞察。
  示例："深圳的显示器支架销量最低，建议加大推广"。

阶段 3：报告撰写 (Reporter Node)
  为什么：最终交付物需要结构化、可读性高。
  怎么做：用专门的 prompt 把分析结果写成业务报告。
  包含：执行摘要、关键发现、行动建议。

生产环境扩展建议：
- 数据源：从 CSV 升级到 SQL 数据库 / 数据仓库
- 可视化：集成 ECharts / Vega-Lite 生成图表
- 定时运行：配合 Cron 或 Airflow 定时生成日报/周报
- 多渠道：自动发送到企业微信 / 钉钉 / Slack

对比 24_text_to_sql.py：
  24：专注于"自然语言查数据库"（即席查询）
  本示例：专注于"完整的数据分析报告"（定期报告）
  两者可以组合：Text-to-SQL 查数据 + Pipeline 生成报告
""")
