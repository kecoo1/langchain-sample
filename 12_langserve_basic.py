"""LangServe API 部署示例 —— 将 LCEL 链和 Agent 发布为 REST API。

LangServe 让你用一行代码将 LangChain 应用部署为生产级 API。
它自动生成 OpenAPI 文档、WebSocket 端点和流式支持。

运行方式：
  1. 将此文件保存为 app.py
  2. 运行: langserve serve app:app
  3. 访问 http://localhost:8000/docs 查看交互式 API 文档
  4. 客户端 SDK 调用: python -m langserve playclient localhost:8000

注意: 本文件设计为被 langserve serve 导入，也可以单独运行演示链的逻辑。
"""

from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
import operator

# ==============================================================================
# 链 1：简单问答链
# ==============================================================================

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    ("human", "{question}"),
])

qa_chain = qa_prompt | model | StrOutputParser()


# ==============================================================================
# 链 2：代码解释链（带结构化输出）
# ==============================================================================

from pydantic import BaseModel, Field


class CodeExplanation(BaseModel):
    summary: str = Field(description="One-sentence summary of what the code does")
    complexity: str = Field(description="Code complexity: 'simple', 'moderate', or 'complex'")
    suggested_improvements: list[str] = Field(description="List of suggested improvements")


code_chain = model.with_structured_output(CodeExplanation)


# ==============================================================================
# 链 3：带记忆的 Agent（LangGraph 构建）
# ==============================================================================

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Sunny, 25°C in {location}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def call_model(state: AgentState) -> AgentState:
    model_with_tools = model.bind_tools([get_weather, calculate])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


agent_builder = StateGraph(AgentState)
agent_builder.add_node("agent", call_model)
agent_builder.add_node("tools", ToolNode([get_weather, calculate]))
agent_builder.set_entry_point("agent")
agent_builder.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "end": END}
)
agent_builder.add_edge("tools", "agent")

memory = MemorySaver()
agent_graph = agent_builder.compile(checkpointer=memory)


# ==============================================================================
# 入口点定义 —— LangServe 从这里暴露 API
# ==============================================================================

from langserve import add_routes

# 所有 add_routes 调用将自动注册为 API 端点
# 例如 add_routes(app, qa_chain, path="/qa") 会暴露 /qa/invoke, /qa/stream, /qa/batch

# 以下为演示：在不启动服务器时直接测试这些链
if __name__ == "__main__":
    import os
    os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")

    print("=== 链 1: 简单 QA ===")
    result = qa_chain.invoke({"question": "What is LangServe?"})
    print(f"Answer: {result}\n")

    print("=== 链 2: 代码解释 ===")
    code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    result = code_chain.invoke(code)
    print(f"Explanation: {result}\n")

    print("=== 链 3: Agent 对话 ===")
    result = agent_graph.invoke(
        {"messages": [HumanMessage(content="What is 12 * 8?")]},
        config={"configurable": {"thread_id": "demo"}}
    )
    print(f"Agent response: {result['messages'][-1].content}\n")

    print("\n=== 部署说明 ===")
    print("1. 安装: pip install langserve[all] fastapi uvicorn")
    print("2. 创建 server.py:")
    print("""
    from fastapi import FastAPI
    from app import qa_chain, code_chain, agent_graph
    from langserve import add_routes

    app = FastAPI(title="My LangServe App")

    # 暴露 QA 链为 REST API
    add_routes(app, qa_chain, path="/qa")

    # 暴露代码解释链
    add_routes(app, code_chain, path="/explain")

    # 暴露 Agent 图
    add_routes(app, agent_graph, path="/agent")
    """)
    print("3. 运行: uvicorn server:app --host 0.0.0.0 --port 8000")
    print("4. 访问: http://localhost:8000/docs")
    print("5. 客户端调用: curl -X POST http://localhost:8000/qa/invoke -H 'Content-Type: application/json' -d '{{\"input\": \"Hello\"}}'")
