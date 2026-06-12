"""LangGraph 基础示例 - 有状态代理工作流。"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 模型实例
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===== 示例 1：简单的顺序图 =====
class SimpleState(TypedDict):
    messages: Annotated[list, operator.add]
    step: int


def step_a(state: SimpleState) -> SimpleState:
    return {"step": state["step"] + 1, "messages": [AIMessage(content="Step A done")]}


def step_b(state: SimpleState) -> SimpleState:
    return {"step": state["step"] + 1, "messages": [AIMessage(content="Step B done")]}


def step_c(state: SimpleState) -> SimpleState:
    return {"step": state["step"] + 1, "messages": [AIMessage(content="Step C done")]}


def should_continue(state: SimpleState) -> str:
    if state["step"] < 3:
        return "continue"
    return "end"

# 构建 SimpleState 的 StateGraph（必须先实例化 builder，再添加节点）
builder = StateGraph(SimpleState)

builder.add_node("a", step_a)
builder.add_node("b", step_b)
builder.add_node("c", step_c)
builder.set_entry_point("a")
builder.add_conditional_edges(
    "a", should_continue, {"continue": "b", "end": END}
)
builder.add_conditional_edges(
    "b", should_continue, {"continue": "c", "end": END}
)
builder.add_edge("c", END)

graph = builder.compile()
result = graph.invoke({"messages": [HumanMessage(content="Start")], "step": 0})
print(f"1. 简单图结果: {result}\n")


# ===== 示例 2：在 LangGraph 中使用 LLM 代理 =====
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny, 25°C in {location}"


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"


tools = [get_weather, search]
tool_node = ToolNode(tools)
model_with_tools = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def call_model(state: AgentState) -> AgentState:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue_fn(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

# 为 AgentState 创建 StateGraph
agent_builder = StateGraph(AgentState)
agent_builder.add_node("agent", call_model)
agent_builder.add_node("tools", tool_node)
agent_builder.set_entry_point("agent")
agent_builder.add_conditional_edges(
    "agent", should_continue_fn, {"tools": "tools", "end": END}
)
agent_builder.add_edge("tools", "agent")

# 添加持久化记忆
memory = MemorySaver()
agent_graph = agent_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
result = agent_graph.invoke(
    {"messages": [HumanMessage(content="What's the weather in Tokyo?")]},
    config,
)
print(f"2. 代理响应: {result['messages'][-1].content}\n")

# 使用记忆进行后续对话
result = agent_graph.invoke(
    {"messages": [HumanMessage(content="What about Paris?")]},
    config,
)
print(f"3. 后续: {result['messages'][-1].content}\n")


# ===== 示例 3：简单聊天机器人 =====
class ChatState(TypedDict):
    messages: Annotated[list, operator.add]


def chatbot(state: ChatState) -> ChatState:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 为聊天机器人创建 StateGraph
chat_builder = StateGraph(ChatState)
chat_builder.add_node("chatbot", chatbot)
chat_builder.set_entry_point("chatbot")
chat_builder.add_edge("chatbot", END)

chat_memory = MemorySaver()
chat_graph = chat_builder.compile(checkpointer=chat_memory)

print("=== 聊天机器人 (输入 'quit' 退出) ===")
user_messages = ["Hi!", "What is LangChain?", "quit"]
config = {"configurable": {"thread_id": "chat_1"}}

for msg in user_messages:
    if msg == "quit":
        break
    result = chat_graph.invoke(
        {"messages": [HumanMessage(content=msg)]}, config
    )
    print(f"User: {msg}")
    print(f"Bot: {result['messages'][-1].content}\n")

# =============================================================================
# 教学备注：LangGraph——从"链"到"图"，从"线性"到"循环"
# =============================================================================
# 核心问题：学完了前 9 个例子，你已经能用链(Chain)做很多事情了——
#   03 学了一步步线性传递（prompt -> model -> parser）
#   05 学了 Agent 的"思考-行动-观察"循环
#   但有没有想过一个问题：Agent 的"循环"是怎么实现的？
#   答案就是 LangGraph——一个用"图"来表达工作流的框架
#
# 为什么需要图（Graph），链（Chain）不够吗？
#   链式（Chain）：线性、无分支、无循环、无状态 persistence
#     适合：固定流程的 ETL、简单的模板->模型->解析
#   图式（Graph）：有节点、有边、有条件分支、有循环、有状态管理
#     适合：Agent 思考循环、多步工具调用、人工审核流程、状态机式对话
#   本质区别：Chain 是"静态管道"；Graph 是"动态状态机"
#
# LangGraph 的三大核心概念：
#   State（状态）：整个工作流的"共享数据面板"，所有节点都可以读写
#   Node（节点）：处理逻辑的单元，接收 state -> 处理 -> 返回更新后的 state
#   Edge（边）：决定执行路径——固定边(线性的)或条件边(分支/循环)
#
# 概念 1：State（状态）—— 工作流的数据契约
#   是什么：一个 TypedDict，定义了整个图共享的数据结构
#   class SimpleState(TypedDict):
#       messages: Annotated[list, operator.add]  # 消息列表，自动追加
#       step: int                                  # 步骤计数器，后写覆盖前写
#   Annotated[list, operator.add] 是什么：reducer（归约器）注解
#     没有 reducer 的字段：后一个节点的写入会覆盖前一个（如 step）
#     有 operator.add 的字段：所有节点的写入会自动合并（列表拼接）
#   为什么需要 reducer：
#     图可能有多个节点"同时"往 state 写数据（并行节点）
#     如果没有 reducer，最后一个写入的会覆盖之前的——丢失数据
#     operator.add 让"写入"变成"追加"，几个节点各追加各的，最终合并
#   设计原则：state 要包含所有节点需要"共享"的数据
#     比如聊天机器人的 state 就是消息列表——所有节点都能看到全部对话
#
# 概念 2：Node（节点）—— 工作的基本单元
#   是什么：一个接收 state、返回 partial state 的函数
#   def step_a(state: SimpleState) -> SimpleState:
#       return {"step": state["step"] + 1, "messages": [AIMessage(content="Step A done")]}
#   关键理解：节点不需要返回"完整"state，只返回要修改的字段
#     框架会把返回值合并到 state 中（字段级合并，不是替换整个 state）
#   节点的类型：
#     - 纯函数节点：同步函数，做计算/规则处理（示例 1 的 step_a）
#     - LLM 节点：调用模型，返回模型输出（示例 2 的 call_model）
#     - ToolNode：预置的"工具执行节点"，自动处理 tool_calls
#     - 异步节点：async def，返回 awaitable
#   节点的返回 = 对 state 的"delta"更新：你只写"改了哪些字段"
#
# 概念 3：Edge（边）—— 执行路线图
#   两种边：
#     固定边：add_edge("a", "b") —— a 执行完就一定走 b
#     条件边：add_conditional_edges("a", router_fn, mapping)
#       router_fn(state) 返回字符串
#       mapping = {"continue": "b", "end": END}
#       根据 router_fn 的返回值，选择走哪条边
#   循环是怎么实现的？条件边！
#     示例 2 中 agent -> tools -> agent（回到自己）就是循环
#     因为 agent 节点执行后，如果返回了 tool_calls -> 走 tools 边
#     tools 执行后 -> 固定边回到 agent -> agent 再次判断是否还要调工具
#   这就是 Agent 的"思考-行动-循环"在 LangGraph 中的实现
#
# 示例 1 解读：简单顺序图
#   做什么：三个步骤顺序执行：A -> 判断是否继续 -> B -> 判断 -> C
#   为什么做这个例子：展示最基础的"状态传递 + 条件路由"
#   问题：step < 3 时继续，但 A 和 B 只把 step +1，所以：
#     start: step=0 -> A: step=1 -> condition < 3 -> B: step=2 -> condition < 3 -> C: step=3 -> condition >= 3 -> END
#   关键学习点：
#     - StateGraph 初始化时传入状态类型
#     - add_node 注册处理函数
#     - add_conditional_edges 用返回字符串做路由
#     - END 是内置终止标记
#
# 示例 2 解读：Agent 图
#   做什么：模型收到消息 -> 判断是否需要调工具 -> 调工具或直接回答
#   与示例 1 的关键区别：有循环（agent -> tools -> agent）
#   call_model 函数：
#     response = model_with_tools.invoke(state["messages"])
#     return {"messages": [response]}
#     state["messages"] 包含了整个对话历史（因为 operator.add 自动追加）
#   should_continue_fn 函数：
#     判断最后一条消息是否包含 tool_calls
#     有 -> "tools" -> 走向 ToolNode 执行工具
#     无 -> "end" -> 结束
#   ToolNode 做了什么：
#     tool_node = ToolNode([get_weather, search])
#     自动解析 tool_calls -> 调用对应函数 -> 构造 ToolMessage -> 追加到 messages
#     然后边 edge("tools", "agent") 让流程回到 agent
#   checkpointer（MemorySaver）做了什么：
#     在一次 invoke 之间保存状态（不是跨对话保存）
#     使得第二次 invoke("What about Paris?") 时能看到第一次的对话
#   thread_id="1" 的作用：隔离不同的对话线程
#     不同 thread_id 的状态互不影响
#     同一 thread_id 的多次 invoke 共享历史
#
# 示例 3 解读：最简单的聊天机器人
#   做什么：输入 -> 模型 -> 输出，但带记忆持久化
#   为什么还要做这个例子：展示"如果没有循环、没有工具，Graph 也能用"
#     chatbot 节点就是 model.invoke(state["messages"])
#     然后固定边到 END——没有循环，一次就结束
#   带记忆的关键：checkpointer = MemorySaver() + thread_id
#     每次 invoke 时，框架自动从 checkpointer 读取历史
#     invoke 完成后，框架自动把新消息写入 checkpointer
#   类比：checkpointer 就像是数据库，thread_id 是主键
#
# 从 05_agent.py 到 10_langgraph_basics.py 的认知升级：
#   05 中的 create_tool_calling_agent + AgentExecutor：
#     是一个"封装好的黑盒"——你不知道循环是怎么实现的
#     AgentExecutor 内部帮你做了：调用模型 -> 判断 -> 执行工具 -> 循环
#   10 中的 LangGraph Agent：
#     是"拆开的白盒"——每个节点、每条边、每次循环你都看得到
#     你可以自由修改：在 agent 和 tools 之间加一个"审批节点"
#     或者在循环计数超过 3 次时走降级路径
#     这就是 LangGraph 的价值：AgentExecutor 不够灵活时，用 Graph 构建自定义循环
#
# 进阶概念预览（理解存在，不必深究）：
#   - interrupt()：在节点中暂停图执行，等待外部输入 resume
#     用途：人工审批、多轮确认、停等外部事件
#   - Subgraph：一个图作为另一个图的节点
#     用途：复杂任务拆分子流程，比如"分析订单"子图 + "发送通知"子图
#   - .get_graph().draw_mermaid()：可视化图结构，导出为流程图
#     用途：调试时看图的拓扑结构是否正确
#   - .stream() / .astream()：逐节点产出状态快照
#     用途：实时监控图执行进度，展示中间结果
#
# 教学建议顺序：
#   1. 先理解"为什么需要图"——Agent 的循环是怎么实现的？
#   2. 再理解三个核心概念：State、Node、Edge
#   3. 跑通示例 1（简单顺序）理解"状态 + 条件路由"
#   4. 再跑通示例 2（Agent 图）理解"循环"——这是最常用的模式
#   5. 最后理解 checkpointer 的作用——"持久化让对话有记忆"
#   6. 示例 3 是最简版，理解"没有循环的图"的边界情况
