"""LangChain 代理（Agent）示例。"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 模型实例
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 定义工具
@tool
def get_time(city: str) -> str:
    """Get current time for a city.

    Args:
        city: City name
    """
    return f"Current time in {city} is 14:30"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression like '2 + 2'
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def search(query: str) -> str:
    """Search for information online.

    Args:
        query: Search query
    """
    return f"Here are search results for '{query}': LangChain is a framework for building LLM applications."


tools = [get_time, calculate, search]


# 1. 工具调用代理（推荐用于 OpenAI/Anthropic）
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("=== 工具调用代理 ===")
result = agent_executor.invoke({
    "input": "What is 25 * 4 + 10? Also, what time is it in Tokyo?"
})
print(f"\nFinal answer: {result['output']}\n")


# 2. ReAct 代理
react_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools to answer questions."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

react_agent = create_react_agent(model, tools, react_prompt)
react_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

print("=== ReAct 代理 ===")
result = react_executor.invoke({
    "input": "Search for LangChain and tell me what it is."
})
print(f"\nFinal answer: {result['output']}\n")


# 3. 带会话记忆的代理
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent

memory = ConversationSummaryBufferMemory(
    llm=model,
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=1000,
)

agent_with_memory = create_tool_calling_agent(model, tools, prompt)
executor_with_memory = AgentExecutor(
    agent=agent_with_memory,
    tools=tools,
    memory=memory,
    verbose=True,
)

print("=== 带记忆的代理 ===")
executor_with_memory.invoke({"input": "Hi, my name is Alice"})
executor_with_memory.invoke({"input": "What's my name?"})

# =============================================================================
# 教学备注：Agent（代理）的核心概念——从"单次工具调用"到"自主推理循环"
# =============================================================================
# 核心问题：01_basics/04_tools.py 已经学了 bind_tools + 手动执行，为什么还要 Agent？
#   关键区别：01_basics/04_tools.py 是"模型说用什么工具，开发者在代码里调一次"
#     Agent 是"模型自己决定：是否要调用->调用哪个->看结果->再决定下一步"
#   类比：01_basics/04_tools.py 相当于"计算器"——你按按钮它给你结果；
#     Agent 相当于"数学家教"——它会自己思考"这一步该用加法还是公式"，调完工具再看结果决定下一步
#   Agent 的本质：Observe -> Think -> Act -> Observe 的自主循环（ReAct 范式）
#
# 为什么需要 Agent？
#   有些问题不能"一步解决"——比如"东京天气怎么样？推荐个活动，再查活动地址？"
#   第一步：查天气 -> 第二步：根据天气推荐活动 -> 第三步：查活动地址
#   这种多步推理需要"中间结果"参与下一步决策，这就是 Agent 的核心价值
#
# Agent 类型 1：create_tool_calling_agent（推荐，原生工具调用）
#   是什么：利用模型原生的 function/tool calling 能力，模型输出结构化工具调用请求
#   为什么是"推荐"：模型原生理解工具调用，不需要"教"它格式
#     模型直接返回 tool_calls=[{name, args}]，无需文本解析，准确率最高
#   工作流程：
#     用户输入 -> 模型推理 -> 有工具调用? -> 执行工具 -> 返回结果 -> 模型整合 -> 输出
#                                          \-> 无工具调用? -> 直接输出
#   限制：要求模型必须支持 function calling（OpenAI, Anthropic, Gemini, Mistral, Groq...）
#     如果模型不支持（如多数纯文本开源模型），会报错或行为异常
#   适用场景：所有支持 function calling 的模型，生产环境首选
#
# Agent 类型 2：create_react_agent（兼容方案，文本推理）
#   是什么：ReAct（Reasoning + Acting）范式——模型在文本中写"Thought/Action/Action Input/Observation"
#   为什么存在：解决"模型不支持 tool calling"的问题
#     模型只需要会生成文本，不需要任何特殊能力
#   工作流程：
#     模型输出: "Thought: 需要查天气\nAction: get_weather\nAction Input: Tokyo"
#     AgentExecutor 解析文本 -> 提取 Action 和 Action Input -> 执行工具 ->
#     构造 Observation: "25°C" -> 拼回提示 -> 模型继续推理
#   代价：文本解析脆弱（格式错一点就乱了）；Token 消耗大（每次写 Thought 和 Action）；
#     不能并行调用多个工具（除非自己扩展）
#   适用场景：不支持 tool calling 的模型、教学理解 ReAct 范式、本地小模型
#
# 为什么需要 AgentExecutor？
#   它解决 Agent 的核心问题："循环"——不能无限调用，不能死循环，不能因为解析失败就崩溃
#   AgentExecutor 的工作：
#     1. 调用 agent（模型）
#     2. 判断输出是"最终答案"还是"工具调用"
#     3. 如果是工具调用 -> 执行 -> 结果写回 scratchpad -> 回到第 1 步
#     4. 如果是最终答案 -> 返回
#     5. 超出 max_iterations -> 强制终止
#     6. 解析失败 -> handle_parsing_errors 决定重试还是报错
#
# Agent 增强：带会话记忆
#   问题：Agent 默认没有记忆——第二次对话不知道第一次说过什么
#   解决方案：在 prompt 中加入 MessagesPlaceholder(variable_name="chat_history") +
#     AgentExecutor 传入 memory 参数
#   记忆类型：
#     - ConversationBufferMemory：完整保留所有对话，最简单直接
#     - ConversationSummaryBufferMemory：最近对话完整保留，早期对话自动压缩摘要
#       这是最常用的"平衡方案"，通过 max_token_limit 控制切换阈值
#   关键：memory_key 必须与 prompt 中的 MessagesPlaceholder.variable_name 一致
#
# AgentExecutor 的关键配置参数（生产环境必须理解）：
#   verbose=True：打印完整中间过程，开发调试时开启，生产关掉
#   max_iterations=N：最大推理步数，防止无限循环（某模型就是死活不给出答案）
#     OpenAI 通常 3-5 步，复杂任务可设 10-15
#   handle_parsing_errors=True：ReAct 格式错误时自动重试，而不是崩溃
#   early_stopping_method="generate"：达到 max_iterations 时，让模型再生成一次最终答案；
#     "force"：直接返回最后输出，可能不完整
#   return_intermediate_steps=True：保留中间步骤，用于调试和展示思考过程
#
# 从 01_basics/04_tools.py 到 01_basics/05_agent.py 的思维转变：
#   04：模型 -> tool_calls -> 开发者手动执行 -> 手动构造 ToolMessage -> 模型整合
#   05：AgentExecutor 自动完成上述所有步骤，开发者只需 invoke 一次
#   Agent 并不是"新东西"，而是"工具调用 + 循环 + 记忆 + 错误处理"的封装
#
# 教学建议顺序：
#   1. 先理解"模型决策 -> 工具执行 -> 结果反馈"这个循环，用 04 手动做过一遍
#   2. 再用 create_tool_calling_agent，让 AgentExecutor 替你做循环
#   3. 再理解 ReAct 原理（虽然不常用，但它是 Agent 的思想起源）
#   4. 最后添加记忆，理解"无状态 vs 有状态"的区别
