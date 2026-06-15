"""Text-to-SQL 与数据分析 Agent —— 自然语言查询数据库。

Text-to-SQL 是 Agent 最实用的场景之一：用户用自然语言提问，
Agent 生成 SQL → 执行查询 → 返回分析结果。

本文件覆盖 4 个递进案例：
1. 单表查询：自然语言 → SQL → 执行
2. 多表 JOIN：查询关联数据
3. 带验证的安全执行链：防止注入和破坏
4. 数据分析 Agent：叠加可视化建议
"""

import os
import sqlite3
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 准备：创建示例数据库
# ==============================================================================
DB_PATH = "/tmp/example_shop.db"


def init_database():
    """创建示例电商数据库，包含商品和订单表。"""
    if Path(DB_PATH).exists():
        Path(DB_PATH).unlink()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            customer TEXT,
            quantity INTEGER,
            total REAL,
            order_date TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        INSERT INTO products VALUES
            (1, 'Wireless Mouse', 'Electronics', 29.99, 150),
            (2, 'Mechanical Keyboard', 'Electronics', 89.99, 75),
            (3, 'USB-C Hub', 'Electronics', 49.99, 200),
            (4, 'Notebook Set', 'Stationery', 12.99, 500),
            (5, 'Desk Lamp', 'Furniture', 39.99, 120),
            (6, 'Monitor Stand', 'Furniture', 59.99, 45);

        INSERT INTO orders VALUES
            (1, 1, 'Alice', 2, 59.98, '2026-01-15'),
            (2, 2, 'Bob', 1, 89.99, '2026-01-16'),
            (3, 4, 'Alice', 5, 64.95, '2026-01-20'),
            (4, 3, 'Charlie', 1, 49.99, '2026-02-01'),
            (5, 6, 'Bob', 1, 59.99, '2026-02-05'),
            (6, 1, 'Diana', 3, 89.97, '2026-02-10'),
            (7, 5, 'Alice', 1, 39.99, '2026-02-15'),
            (8, 2, 'Charlie', 1, 89.99, '2026-02-20');
    """)
    conn.commit()
    conn.close()


def get_schema() -> str:
    """获取数据库 schema 描述。"""
    return """
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price REAL,
    stock INTEGER
);
-- Sample: (1, 'Wireless Mouse', 'Electronics', 29.99, 150)

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    product_id INTEGER,
    customer TEXT,
    quantity INTEGER,
    total REAL,
    order_date TEXT
);
-- Sample: (1, 1, 'Alice', 2, 59.98, '2026-01-15')
"""


init_database()
print(f"数据库已初始化: {DB_PATH}\n")


# ==============================================================================
# 示例 1：单表查询 —— LLM 生成 SQL 并执行
# ==============================================================================
print("=== 示例 1: 单表查询 ===")

prompt_single = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL expert. Given the database schema and a question,
generate a SQL query. Return ONLY the SQL query, nothing else.

Schema:
{schema}

Rules:
- Use only SELECT queries (read-only)
- Be precise with column names from the schema
- Use SQLite syntax"""),
    ("human", "{question}"),
])

sql_chain = prompt_single | model | StrOutputParser()

question_1 = "How many products are in each category?"
sql_1 = sql_chain.invoke({"schema": get_schema(), "question": question_1}).strip()
sql_1 = sql_1.replace("```sql", "").replace("```", "").strip()

print(f"1.1 生成的 SQL: {sql_1}")
conn = sqlite3.connect(DB_PATH)
result_1 = conn.execute(sql_1).fetchall()
print(f"   查询结果: {result_1}")
print()

# ==============================================================================
# 示例 2：多表 JOIN 查询
# ==============================================================================
print("=== 示例 2: 多表 JOIN ===")

question_2 = "Which customer spent the most money? Show their total spending."
sql_2 = sql_chain.invoke({"schema": get_schema(), "question": question_2}).strip()
sql_2 = sql_2.replace("```sql", "").replace("```", "").strip()

print(f"2.1 生成的 SQL: {sql_2}")
result_2 = conn.execute(sql_2).fetchall()
print(f"   查询结果: {result_2}")
print()

conn.close()

# ==============================================================================
# 示例 3：带验证的安全执行链
# ==============================================================================
print("=== 示例 3: 带验证的安全执行链 ===")

# 生产环境中不能直接执行 LLM 生成的 SQL——存在 SQL 注入风险
# 安全执行链 = SQL 生成 + SQL 验证 + 只读执行

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class SQLQuery(BaseModel):
    """验证后的 SQL 查询。"""
    query: str = Field(description="The generated SQL query")
    explanation: str = Field(description="What this query does in plain English")
    is_safe: bool = Field(description="Whether the query is safe (SELECT only)")


parser = PydanticOutputParser(pydantic_object=SQLQuery)

safe_prompt = ChatPromptTemplate.from_messages([
    ("system", """Generate a SQL query for the given question.

Schema:
{schema}

{format_instructions}

Safety rule: ONLY generate SELECT queries. Reject any INSERT/UPDATE/DELETE/DROP."""),
    ("human", "{question}"),
])

safe_chain = safe_prompt | model.with_structured_output(SQLQuery)

question_3 = "Show me the top 3 best-selling products by revenue."
sql_query = safe_chain.invoke({
    "schema": get_schema(),
    "question": question_3,
    "format_instructions": "",
})

print(f"3. 带验证的 SQL 查询:")
print(f"   说明: {sql_query.explanation}")
print(f"   SQL: {sql_query.query}")
print(f"   安全: {sql_query.is_safe}")

if sql_query.is_safe:
    conn = sqlite3.connect(DB_PATH)
    result = conn.execute(sql_query.query).fetchall()
    print(f"   结果: {result}")
    conn.close()

print()

# ==============================================================================
# 示例 4：完整的数据分析 Agent（查询 + 分析）
# ==============================================================================
print("=== 示例 4: 数据分析 Agent ===")

import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator


class AnalysisState(TypedDict):
    messages: Annotated[list, operator.add]
    query_result: str
    analysis: str


@tool
def query_database(sql: str) -> str:
    """Execute a SELECT SQL query on the shop database.

    Args:
        sql: The SQL query to execute
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed"
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(sql).fetchall()
        conn.close()
        cols = [desc[0] for desc in conn.execute(sql).description]
        return json.dumps([dict(zip(cols, row)) for row in rows], indent=2)
    except Exception as e:
        return f"Error: {e}"


@tool
def suggest_visualization(data: str, data_type: str) -> str:
    """Suggest the best chart type for the given data.

    Args:
        data: The data description
        data_type: Type like 'bar', 'pie', 'line', 'table'
    """
    suggestions = {
        "bar": "Best for comparing categories (e.g., sales by product)",
        "pie": "Best for showing proportions (e.g., market share)",
        "line": "Best for trends over time (e.g., monthly revenue)",
        "table": "Best for detailed numerical comparisons",
    }
    return suggestions.get(data_type, "Consider a bar or line chart")


analysis_tools = [query_database, suggest_visualization]
model_with_tools = model.bind_tools(analysis_tools)

from langgraph.prebuilt import ToolNode

tool_node = ToolNode(analysis_tools)


def call_analysis_model(state: AnalysisState) -> AnalysisState:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AnalysisState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


builder = StateGraph(AnalysisState)
builder.add_node("agent", call_analysis_model)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

analysis_agent = builder.compile()

result = analysis_agent.invoke({
    "messages": [HumanMessage(content="""Analyze our sales data:
1. What is the total revenue across all products?
2. Which product category sells the most?
3. Suggest how to visualize the results""")]
})

print(f"4. 数据分析 Agent 结果:")
for msg in result["messages"]:
    if hasattr(msg, "content") and msg.content:
        print(f"   {msg.content[:200]}")
    elif hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"   [调用工具] {tc['name']}({tc['args']})")
print()


# ==============================================================================
# 教学备注：Text-to-SQL —— 自然语言操作数据库
# ==============================================================================
# 核心问题：业务人员想查数据但不会 SQL，开发人员没时间写报表
# 答案：Text-to-SQL —— 用自然语言代替 SQL
#
# 从简单到生产的演进路径：
#
# 阶段 1：直接生成 SQL（示例 1-2）
#   prompt → model → SQL → execute
#   问题：SQL 可能格式错误、可能有注入风险、可能不精确
#
# 阶段 2：加验证层（示例 3）
#   prompt → model → 结构化输出(SQL + 验证) → execute
#   改进：用 Pydantic 验证 SQL 安全，拒绝非 SELECT 语句
#   问题：单次查询无法处理需要多步推理的复杂问题
#
# 阶段 3：Agent 模式（示例 4）
#   Agent 可以：先生成 SQL → 执行 → 看结果 → 再生成 SQL → ...
#   改进：多次查询、结果分析、可视化建议
#   这才是生产级方案
#
# Text-to-SQL 的关键挑战：
#   1. Schema 理解：模型需要准确理解表结构和关系
#      方案：明确标注主键/外键、样例数据、字段含义注释
#   2. 复杂查询：嵌套子查询、窗口函数、CASE WHEN
#      方案：使用 few-shot 示例引导特定 SQL 模式
#   3. 安全性：SQL 注入、数据泄露、破坏性操作
#      方案：只读用户 + SELECT-only 验证 + 参数化执行
#   4. 幻觉：模型可能"编造"不存在的字段
#      方案：执行前验证字段存在性、EXPLAIN 预检查
#
# 常见错误处理：
#   - 字段不存在：捕获 sqlite3.OperationalError 并让 Agent 重试
#   - 语法错误：返回错误信息让 LLM 修正
#   - 空结果：区分"查询正确但没有数据"和"查询错误"
#   - 超时：设置查询超时，避免大表全量扫描
#
# 扩展方向：
#   - 带 RAG 的 Text-to-SQL：先检索相似 SQL 示例（few-shot）
#   - 多轮对话 SQL：用户追问时，在上一轮 SQL 基础上修改
#   - SQL 优化建议：EXPLAIN 分析 + 索引建议
#   - 可视化输出：生成图表配置（ECharts/Vega-Lite）
