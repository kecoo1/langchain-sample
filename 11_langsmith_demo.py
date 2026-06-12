"""LangSmith 追踪示例——展示如何追踪 LCEL 链、Agent 和 LangGraph。

运行前需要设置 LANGCHAIN_TRACING_V2=true，可选设置 LANGCHAIN_API_KEY。
追踪结果可在 https://smith.langchain.com 查看。

无需 API Key 也能使用本地追踪（tracing 写入内存），
设置 API Key 后可在 LangSmith 云端面板查看完整可视化。
"""

import os
import operator
from typing import Annotated, TypedDict

# LangSmith 追踪设置
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
# 可选：设置 API Key 后数据会上传到 LangSmith 云端
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_API_KEY_HERE"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langsmith import traceable, RunTree
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

model = ChatOllama(model="llama3.2:1b", temperature=0)

# ==============================================================================
# 示例 1：LCEL 链追踪
# ==============================================================================
print("=== 示例 1: LCEL 链追踪 ===")

prompt = ChatPromptTemplate.from_template("What is {thing}? Explain in one sentence.")
chain = prompt | model | StrOutputParser()

# 使用 @traceable 装饰器追踪这个链
@traceable(run_type="chain")
def simple_chain(item: str) -> str:
    return chain.invoke({"thing": item})

result1 = simple_chain("LangChain")
print(f"1. 链结果: {result1}\n")

# ==============================================================================
# 示例 2：Agent 追踪
# ==============================================================================
print("=== 示例 2: Agent 追踪 ===")


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Current weather in {city}: Sunny, 25°C"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def search_info(query: str) -> str:
    """Search for information online."""
    return f"Results for '{query}': LangChain is a framework for LLM applications."


tools = [get_weather, calculate, search_info]
model_with_tools = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def call_model(state: AgentState) -> AgentState:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


agent_builder = StateGraph(AgentState)
agent_builder.add_node("agent", call_model)
agent_builder.add_node("tools", ToolNode(tools))
agent_builder.set_entry_point("agent")
agent_builder.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "end": END}
)
agent_builder.add_edge("tools", "agent")

agent_graph = agent_builder.compile()


@traceable(run_type="chain")
def agent_demo(question: str) -> dict:
    return agent_graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": "smith_demo_1"}}
    )


result2 = agent_demo("What is 3 * 12 + 5? Also tell me the weather in Tokyo.")
print(f"2. Agent 最终响应: {result2['messages'][-1].content}\n")

# ==============================================================================
# 示例 3：LangGraph 步骤追踪
# ==============================================================================
print("=== 示例 3: LangGraph 步骤追踪 ===")


class StepState(TypedDict):
    messages: Annotated[list, operator.add]
    step: int


def add_greeting(state: StepState) -> StepState:
    return {"step": state["step"] + 1, "messages": [AIMessage(content="Greeting added")]}


def add_question(state: StepState) -> StepState:
    return {"step": state["step"] + 1, "messages": [AIMessage(content="Question added")]}


def add_response(state: StepState) -> StepState:
    return {"step": state["step"] + 1, "messages": [AIMessage(content="Response generated")]}


step_builder = StateGraph(StepState)
step_builder.add_node("greeting", add_greeting)
step_builder.add_node("question", add_question)
step_builder.add_node("response", add_response)
step_builder.set_entry_point("greeting")
step_builder.add_edge("greeting", "question")
step_builder.add_edge("question", "response")
step_builder.add_edge("response", END)

step_graph = step_builder.compile()


@traceable(run_type="chain")
def run_step_graph() -> dict:
    return step_graph.invoke(
        {"messages": [HumanMessage(content="Start")], "step": 0}
    )


result3 = run_step_graph()
print(
    f"3. 步骤图完成, 共 {result3['step']} 步, 消息数: {len(result3['messages'])}\n"
)

# ==============================================================================
# 示例 4：手动 RunTree 追踪——自定义追踪粒度
# ==============================================================================
print("=== 示例 4: 手动 RunTree 追踪 ===")


@traceable(run_type="llm")
def call_llm_with_manual_trace(text: str) -> str:
    """手动追踪 LLM 调用，可以捕获输入/输出详情。"""
    result = model.invoke(text).content
    return result


result4 = call_llm_with_manual_trace("Say hello in 5 words.")
print(f"4. LLM 追踪结果: {result4}\n")

print("=== 所有追踪完成 ===")
print("如需在云端面板查看，请访问 https://smith.langchain.com")
print("（需要设置 LANGCHAIN_API_KEY 环境变量）")
