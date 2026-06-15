"""Deep Agents 示例 —— LangChain 官方高层 Agent 框架。

Deep Agents 是 LangChain 最新推出的高层 Agent 框架（2026 年主推），
相比 LangGraph 更底层，内置规划、子 Agent、文件系统访问、代码执行等能力。

本文件展示 Deep Agents 的六大核心能力：
1. 创建第一个 Deep Agent（基础）
2. 任务规划（write_todos 内置能力）
3. 虚拟文件系统（读写文件、搜索）
4. 子 Agent 委托（task 工具）
5. 人机协同（interrupt_on 审批）
6. 多模型切换（Harness Profile）

运行前准备：
  pip install deepagents
  # 确保 Ollama 正在运行: ollama serve
  # 已安装模型: ollama pull llama3.2:1b

⚠️  重要提示：llama3.2:1b 模型较小，tool calling 能力有限，
    部分复杂功能可能需要更强模型（GPT-4o / Claude）才能良好工作。
    每个示例都标注了替代写法。
"""

import os

# ==============================================================================
# 示例 1: 创建第一个 Deep Agent（基础）
# ==============================================================================
print("=" * 60)
print("示例 1: 创建第一个 Deep Agent")
print("=" * 60)

import os

try:
    from deepagents import create_deep_agent
    from langchain_ollama import ChatOllama
    _DEEPAGENTS_AVAILABLE = True
except ImportError:
    _DEEPAGENTS_AVAILABLE = False
    print("⚠️  需要安装 deepagents: pip install deepagents\n")
    create_deep_agent = None


def _check_available():
    if not _DEEPAGENTS_AVAILABLE:
        print("   跳过: need pip install deepagents")


# ==============================================================================
# 示例 1: 创建第一个 Deep Agent（基础）
# ==============================================================================
print("=" * 60)
print("示例 1: 创建第一个 Deep Agent")
print("=" * 60)

if _DEEPAGENTS_AVAILABLE:
    # 自定义工具
    def get_weather(city: str) -> str:
        """Get weather for a given city.

        Args:
            city: City name like 'Tokyo' or 'Beijing'
        """
        weathers = {
            "tokyo": "Sunny, 25°C",
            "beijing": "Cloudy, 18°C",
            "new york": "Rainy, 20°C",
        }
        return weathers.get(city.lower(), f"Weather data not available for {city}")


    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: Math expression like '2 + 2'
        """
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Calculation error: {e}"


    # 使用 Ollama 本地模型创建 Deep Agent
    # 替代写法：
    #   - OpenAI: create_deep_agent(model="openai:gpt-4o-mini", tools=[...])
    #   - Gemini: create_deep_agent(model="google_genai:gemini-2.0-flash", tools=[...])
    #   - Anthropic: create_deep_agent(model="anthropic:claude-sonnet-4-6", tools=[...])

    try:
        agent = create_deep_agent(
            model="ollama:llama3.2:1b",
            tools=[get_weather, calculate],
            system_prompt="You are a helpful assistant. Use tools to answer questions.",
        )

        # 简单调用
        result = agent.invoke({
            "messages": [{"role": "user", "content": "What is 15 * 3 + 20? Also what's the weather in Tokyo?"}]
        })
        print(f"\n1.1 基础调用结果: {result['messages'][-1].content[:200]}\n")

    except Exception as e:
        print(f"   ⚠️  llama3.2:1b 工具调用能力有限，可能失败。建议改用: openai:gpt-4o-mini\n")
        print(f"   错误: {str(e)[:100]}\n")

    print("1. Deep Agent 核心优势:")
    print("   - 内置任务规划 (write_todos)")
    print("   - 内置虚拟文件系统 (write_file/read_file/grep/glob)")
    print("   - 内置子 Agent 委托 (task 工具)")
    print("   - 内置记忆和摘要")
    print("   - 一行代码创建，无需手动编排")
    print()
else:
    _check_available()


# ==============================================================================
# 示例 2: 任务规划（write_todos）
# ==============================================================================
print("=" * 60)
print("示例 2: 任务规划 — write_todos 内置能力")
print("=" * 60)

# Deep Agents 默认内置 TodoListMiddleware
# 当 Agent 接收到多步骤任务时，会自动生成待办列表
# 每个步骤完成后，Agent 自动调用 write_todos 标记完成

# 替代写法：
# agent = create_deep_agent(
#     model="openai:gpt-4o",
#     system_prompt="You are a project manager. Break down tasks clearly.",
# )

if _DEEPAGENTS_AVAILABLE:
    try:
        agent = create_deep_agent(
            model="ollama:llama3.2:1b",
            system_prompt="""You are a project manager. When given a complex task:
1. Break it down into clear, actionable steps
2. Use write_todos to create a todo list
3. Track progress on each step
4. Provide a summary when done""",
        )

        # 复杂多步骤任务 —— Agent 会自动规划
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": """Plan a 3-day trip to Tokyo:
1. Research top attractions
2. Find the best transportation options
3. Recommend restaurants
4. Create a day-by-day itinerary
5. Write the plan to a file""",
            }]
        })
        print(f"\n2. 多步骤任务规划: {result['messages'][-1].content[:300]}\n")

    except Exception as e:
        print(f"   跳过/错误: {str(e)[:80]}\n")
else:
    _check_available()

print("2. 任务规划的核心价值:")
print("   - Agent 自动分解复杂任务为子步骤")
print("   - write_todos 提供进度追踪")
print("   - 长运行任务不会迷失方向")
print("   - 适合研究、分析、规划类任务")
print()


# ==============================================================================
# 示例 3: 虚拟文件系统
# ==============================================================================
print("=" * 60)
print("示例 3: 虚拟文件系统 — write_file/read_file/grep/glob")
print("=" * 60)

# Deep Agents 内置 FilesystemMiddleware
# 提供以下工具给 Agent：
#   write_file: 创建/覆盖文件
#   read_file:  读取文件内容
#   edit_file:  编辑现有文件
#   grep:       搜索文件内容
#   glob:       按模式查找文件
#   directory_listing: 列出目录内容
#
# 上下文管理：当输出过大时，Agent 自动将结果写入文件
# 以避免占用过多的上下文窗口

# 替代写法（指定 root_dir）:
# from deepagents.backends.filesystem import FilesystemBackend
# backend = FilesystemBackend(root_dir="./project")
# agent = create_deep_agent(
#     model="openai:gpt-4o-mini",
#     backend=backend,
# )

if _DEEPAGENTS_AVAILABLE:
    try:
        agent = create_deep_agent(
            model="ollama:llama3.2:1b",
            system_prompt="""You are a code assistant. You can read, write, and search files.
When the user asks about files, use the filesystem tools.
Write results to files when appropriate to avoid context overflow.""",
        )

        # 文件操作 —— Agent 会自动选择 write_file/read_file/grep
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": """1. Create a file called /tmp/projects.txt with these projects:
   - LangChain
   - LangGraph
   - DeepAgents
   - LangSmith
2. Read the file back and count how many projects are listed
3. Append a new project 'LangServe' to the file""",
            }]
        })
        print(f"\n3. 文件操作结果: {result['messages'][-1].content[:300]}\n")

    except Exception as e:
        print(f"   跳过/错误: {str(e)[:80]}\n")
else:
    _check_available()

print("3. 虚拟文件系统的核心价值:")
print("   - Agent 自动管理上下文（大结果写入文件）")
print("   - 持久化存储：文件跨会话保留")
print("   - 代码搜索：grep/glob 替代 subprocess")
print("   - 编辑文件：edit_file 支持精确替换")
print()


# ==============================================================================
# 示例 4: 子 Agent 委托
# ==============================================================================
print("=" * 60)
print("示例 4: 子 Agent 委托 — task 工具")
print("=" * 60)

# Deep Agents 内置 SubAgentMiddleware
# 主 Agent 可以使用 task 工具将子任务委派给子 Agent
# 子 Agent 上下文隔离、可并行执行
#
# 自定义子 Agent 写法：
# research_subagent = {
#     "name": "researcher",
#     "description": "Research assistant with web search",
#     "system_prompt": "You are a researcher. Search the web and summarize.",
#     "tools": [web_search],
# }
# agent = create_deep_agent(
#     model="openai:gpt-4o",
#     subagents=[research_subagent],
# )

# 通用子 Agent（默认启用）：
# Deep Agents 会自动创建一个通用子 Agent，
# 主 Agent 通过 task 工具自动委派，无需额外配置

if _DEEPAGENTS_AVAILABLE:
    try:
        agent = create_deep_agent(
            model="ollama:llama3.2:1b",
            system_prompt="""You are a research coordinator.
When given a complex research task, use the task tool to delegate
sub-questions to specialized sub-agents.""",
        )

        # 多任务委派 —— Agent 使用 task 工具分发
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": """Compare the three LLM frameworks. For each one,
research:
1. What it does best
2. Key strengths and weaknesses
3. When to choose it over the others

Use sub-agents to research each framework in parallel.""",
            }]
        })
        print(f"\n4. 子 Agent 委派结果: {result['messages'][-1].content[:300]}\n")

    except Exception as e:
        print(f"   跳过/错误: {str(e)[:80]}\n")
else:
    _check_available()

print("4. 子 Agent 委托的核心价值:")
print("   - 并行执行多个子任务")
print("   - 上下文隔离：每个子 Agent 独立上下文")
print("   - 主 Agent 综合子任务结果")
print("   - 可自定义子 Agent（指定工具、模型、prompt）")
print()


# ==============================================================================
# 示例 5: 人机协同（interrupt_on）
# ==============================================================================
print("=" * 60)
print("示例 5: 人机协同 — interrupt_on 审批流程")
print("=" * 60)

# Deep Agents 支持 Human-in-the-loop 模式
# 当配置 interrupt_on 时，Agent 在特定操作前会暂停等待人工批准
#
# 关键配置：
#   interrupt_on={"write_file": True}
#     Agent 每次写入文件前暂停，等人工批准
#   interrupt_on={"edit_file": True}
#     Agent 每次编辑文件前暂停
#   interrupt_on={"all": True}
#     所有关键操作前暂停
#
# 必须配合 MemorySaver checkpointer 使用
#
# 恢复流程：
#   1. Agent invoke 到 interrupt 点暂停
#   2. 外部系统查看 interrupt 信息
#   3. 调用 invoke(None, config, resume={"approved": True}) 恢复

if _DEEPAGENTS_AVAILABLE:
    try:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

        agent = create_deep_agent(
            model="ollama:llama3.2:1b",
            interrupt_on={"write_file": True},
            checkpointer=checkpointer,
            system_prompt="""You are a coding assistant.
When you need to write files, pause for human approval.""",
        )

        # 演示：Agent 尝试写入文件时会暂停
        # 实际使用时：
        #   result = agent.invoke(
        #       {"messages": [{"role": "user", "content": "Write a report about LangChain to /tmp/report.txt"}]},
        #       config={"configurable": {"thread_id": "demo_1"}},
        #   )
        #   # 此时会暂停在 write_file 操作前
        #
        #   # 人工审核后恢复：
        #   result = agent.invoke(
        #       None,
        #       config={"configurable": {"thread_id": "demo_1"}},
        #       resume={"approved": True},  # 或 {"approved": False}
        #   )

        print("   演示配置（不实际运行 interrupt）:")
        print(f"   - interrupt_on: {{'write_file': True}}")
        print(f"   - checkpointer: {type(checkpointer).__name__}")
        print(f"   - thread_id: 'demo_1'")
        print()
        print("   使用流程:")
        print("   1. Agent invoke -> 遇到 write_file -> 暂停")
        print("   2. 人工查看 Agent 的计划")
        print("   3. resume={'approved': True} -> 继续执行")
        print("   4. resume={'approved': False} -> 拒绝操作")
        print()

    except Exception as e:
        print(f"   跳过: {e}\n")
else:
    _check_available()

print("5. 人机协同的核心价值:")
print("   - 关键操作（写文件、删除数据、公开发布）需要人工审批")
print("   - interrupt 让图变成'事件驱动'而非'一次性运行'")
print("   - 类比：Git merge 冲突需要手动解决；CI/CD 需要人工审批")
print("   - 金融交易、医疗处方、代码合并等高风险场景")
print()


# ==============================================================================
# 示例 6: 多模型切换（Harness Profile）
# ==============================================================================
print("=" * 60)
print("示例 6: 多模型切换 — Harness Profile")
print("=" * 60)

# Harness Profile 允许为不同 provider 注册默认配置
# 注册后，使用 provider:model 字符串创建 agent 时自动应用配置
#
# 注册层级：
#   1. Provider 级别：如 "openai" 对所有 OpenAI 模型生效
#   2. Model 级别：如 "openai:gpt-4o" 仅对该模型生效
#   3. Model 级别优先级高于 Provider 级别

if _DEEPAGENTS_AVAILABLE:
    try:
        from deepagents import ProviderProfile, register_provider_profile

        # 为 OpenAI 注册默认配置
        register_provider_profile(
            "openai",
            ProviderProfile(init_kwargs={"temperature": 0}),
        )

        # 为特定模型注册额外配置
        register_provider_profile(
            "openai:gpt-4o-mini",
            ProviderProfile(init_kwargs={"max_tokens": 2048}),
        )

        # 为 Ollama 注册默认配置
        register_provider_profile(
            "ollama",
            ProviderProfile(init_kwargs={"temperature": 0}),
        )

        print("   已注册的 Profile:")
        print("   - openai -> temperature=0")
        print("   - openai:gpt-4o-mini -> max_tokens=2048 (覆盖)")
        print("   - ollama -> temperature=0")
        print()

        # 使用 provider:model 字符串，自动应用 Profile 配置
        # 无需显式传入 temperature 参数
        agent_openai = create_deep_agent(
            model="openai:gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )
        print("   1. 使用 OpenAI 创建 agent:")
        print(f"   - model: openai:gpt-4o-mini")
        print(f"   - Profile 自动应用 temperature=0, max_tokens=2048")
        print()

        agent_ollama = create_deep_agent(
            model="ollama:llama3.2:1b",
            system_prompt="You are a helpful assistant.",
        )
        print("   2. 使用 Ollama 创建 agent:")
        print(f"   - model: ollama:llama3.2:1b")
        print(f"   - Profile 自动应用 temperature=0")
        print()

        print("   运行时切换模型：")
        # 同一 agent 实例，通过 runtime context 切换模型
        # from langchain.agents.middleware import wrap_model_call
        # from langchain.agents.middleware.types import ModelRequest, ModelResponse
        #
        # @wrap_model_call
        # def configurable_model(request, handler):
        #     model_name = request.runtime.context.model
        #     model = init_chat_model(model_name)
        #     return handler(request.override(model=model))
        #
        # agent = create_deep_agent(
        #     model="openai:gpt-4o-mini",
        #     middleware=[configurable_model],
        #     context_schema=UserContext,
        # )
        #
        # # 同一 agent，运行时使用不同模型
        # result = agent.invoke(
        #     {"messages": [...]},
        #     context=UserContext(model="anthropic:claude-sonnet-4-6"),
        # )
        print("   - 同一 agent 实例，运行时 context 指定不同 model")
        print("   - 示例: context=UserContext(model='anthropic:claude-sonnet-4-6')")
        print()

    except Exception as e:
        print(f"   跳过: {e}\n")
else:
    _check_available()

print("6. Harness Profile 的核心价值:")
print("   - 集中管理模型配置，无需每个 agent 重复设置")
print("   - Provider 级别配置对所有模型生效")
print("   - Model 级别配置覆盖 Provider 级别")
print("   - 运行时动态切换模型，无需重建 agent")
print()


# ==============================================================================
# 附录：Ollama 模型能力评估
# ==============================================================================
print("=" * 60)
print("附录: Ollama llama3.2:1b 能力评估")
print("=" * 60)

print("""
llama3.2:1b 是一个 13 亿参数的轻量级模型，适合学习但能力有限：

能力              | 评估          | 建议
-----------------|---------------|--------------------------------
简单对话          | ✅ 良好       | 可以流畅回答基础问题
工具调用          | ⚠️  不稳定    | 可能偶尔格式错误
文件系统操作      | ⚠️  可用       | 基本操作可以，复杂编辑会出错
多步骤规划        | ❌ 较弱       | 建议改用 GPT-4o / Claude
子 Agent 委派     | ❌ 较弱       | 需要较强 reasoning 能力
人机协同审批      | ✅ 可用       | interrupt_on 逻辑不依赖模型

推荐模型升级路径:
  1. ollama:llama3.2:1b    — 学习概念，理解 API
  2. ollama:qwen2.5:7b     — 更好的工具调用
  3. ollama:devstral-2     — Deep Agents eval 82% 通过率
  4. openai:gpt-4o-mini    — 生产环境首选
  5. openai:gpt-5.5        — 最强 Agent 模型（80% 总评分）
  6. google_genai:gemini-3.5-flash — 文件/检索满分，82% 工具调用

参考：https://github.com/langchain-ai/deepagents/tree/main/libs/evals
""")

print("=" * 60)
print("示例全部完成")
print("=" * 60)
