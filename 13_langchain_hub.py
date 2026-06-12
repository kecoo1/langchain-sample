"""LangChain Hub Prompt 管理示例 —— 从 Hub 拉取、推送和管理 Prompt 模板。

LangChain Hub 是一个 Prompt 模板共享和管理平台，类似"Prompt 的 GitHub"。
你可以在 Hub 上存储、版本化、协作编辑 Prompt，并在代码中通过 short name 引用。

网址: https://smith.langchain.com/hub

运行前需要设置:
  export LANGCHAIN_API_KEY="your-api-key"
"""

import os
os.environ.setdefault("LANGCHAIN_API_KEY", "YOUR_LANGCHAIN_API_KEY_HERE")

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain import hub

# ==============================================================================
# 示例 1：从 Hub 拉取 Prompt
# ==============================================================================
print("=== 示例 1: 从 Hub 拉取 Prompt ===")

# 拉取官方示例 Prompt —— 经典的 ReAct Agent Prompt
react_prompt = hub.pull("rlm/rag-prompt")
print(f"1. 拉取的 Prompt 模板:\n{react_prompt}\n")

# 拉取后可以查看其结构
print(f"   模板消息数: {len(react_prompt.messages)}")
for msg in react_prompt.messages:
    print(f"   - {type(msg).__name__}: {getattr(msg, 'prompt', None)}\n")


# ==============================================================================
# 示例 2：使用 Hub Prompt 构建 RAG 链
# ==============================================================================
print("=== 示例 2: 使用 Hub Prompt 构建 RAG 链 ===")

llm = ChatOllama(model="llama3.2:1b", temperature=0)

# 拉取官方推荐的 RAG Prompt
rag_prompt = hub.pull("rlm/rag-prompt")

# 模拟检索到的文档
from langchain_core.documents import Document
docs = [
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
    Document(page_content="It supports retrieval-augmented generation (RAG) out of the box."),
    Document(page_content="LangChain integrates with 50+ vector databases and embedding providers."),
]


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# 构建链
rag_chain = (
    {
        "context": lambda x: format_docs(x["context"]),
        "question": lambda x: x["question"],
    }
    | rag_prompt
    | llm
)

result = rag_chain.invoke({
    "context": docs,
    "question": "What is LangChain and what does it support?"
})
print(f"2. RAG 回答: {result.content}\n")


# ==============================================================================
# 示例 3：创建并发布 Prompt 到 Hub
# ==============================================================================
print("=== 示例 3: 创建并发布 Prompt 到 Hub ===")

# 创建一个自定义 Prompt
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {tone} assistant that specializes in {domain}."),
    ("human", "{question}"),
])

# 推送到 Hub（需要 LANGCHAIN_API_KEY）
# 推送后可以在 https://smith.langchain.com/hub 查看
#
# 使用方法:
#   from langchain import hub
#   prompt = hub.pull("your-username/my-custom-prompt")
#
# 推送命令（取消注释以实际推送）:
#   hub.push("your-username/my-custom-prompt", custom_prompt)
#
# 推送后的版本管理:
#   hub.pull("your-username/my-custom-prompt:main")    # 最新版本
#   hub.pull("your-username/my-custom-prompt:v0.1.0")  # 特定版本
#   hub.pull("your-username/my-custom-prompt:abc1234")  # 特定 commit hash
#
print("   自定义 Prompt 已创建:")
print(f"   模板: {custom_prompt}")
print("   取消注释 hub.push(...) 以推送到 Hub\n")


# ==============================================================================
# 示例 4：Hub 中常用的公开 Prompt
# ==============================================================================
print("=== 示例 4: 常用公开 Prompt 参考 ===")

# 以下是 LangChain Hub 上一些常用 Prompt 的 short name:
#
# RAG 相关:
#   "rlm/rag-prompt"              - 经典 RAG 提示模板
#   "langchain-ai/tavily-agent"   - Tavily 搜索 Agent Prompt
#
# 分类/抽取:
#   "langchain-ai/openai-function-format"  - OpenAI 函数调用格式
#   "langchain-ai/json-output"             - JSON 输出格式
#
# Agent:
#   "rlm/rag-agent-sales"          - 销售场景 Agent
#   "hwchase17/react"              - 经典 ReAct Agent Prompt
#
# 翻译:
#   "langchain-ai/translating-system-prompt"  - 翻译系统提示
#
# 代码:
#   "langchain-ai/code-question-answer"  - 代码问答
#
print("   查看完整列表: https://smith.langchain.com/hub")
print("   搜索: https://smith.langchain.com/hub?q=\n")


# ==============================================================================
# 示例 5：版本管理与 A/B 测试
# ==============================================================================
print("=== 示例 5: 版本管理与 A/B 测试 ===")

# 假设你已经推送了两个版本的 Prompt:
v1_prompt = hub.pull("your-username/translation-prompt:v1")
v2_prompt = hub.pull("your-username/translation-prompt:main")

# 用不同版本做 A/B 测试
test_cases = [
    {"source": "Hello, how are you?", "target_language": "French"},
    {"source": "The weather is nice today", "target_language": "Spanish"},
]

for case in test_cases:
    # v1 翻译
    result_v1 = v1_prompt | llm | lambda x: x.content
    output_v1 = result_v1.invoke(case)

    # v2 翻译
    result_v2 = v2_prompt | llm | lambda x: x.content
    output_v2 = result_v2.invoke(case)

    print(f"   原文: {case['source']}")
    print(f"   v1:   {output_v1}")
    print(f"   v2:   {output_v2}")
    print()

# ==============================================================================
# 教学备注：LangChain Hub —— Prompt 的版本控制和团队协作
# ==============================================================================
# 核心问题：Prompt 应该存在哪里？为什么不能只放在代码里？
#   1. Prompt 是"代码的一部分"，也需要版本控制
#   2. 非技术人员（产品/运营）需要参与 Prompt 编写和调优
#   3. Prompt 需要 A/B 测试和灰度发布
#   4. 不同环境（dev/staging/prod）需要不同 Prompt 版本
#
# Hub 的核心功能:
#   1. hub.pull("owner/name") - 从 Hub 拉取最新 Prompt
#   2. hub.pull("owner/name:v1.0") - 拉取特定版本
#   3. hub.push("owner/name", prompt) - 推送 Prompt 到 Hub
#   4. Hub 提供 Web UI 编辑 Prompt，支持评论和版本对比
#
# 为什么这很重要:
#   传统方式: 改 Prompt -> 改代码 -> 重新部署 -> 上线
#   Hub 方式: 改 Hub 上的 Prompt -> 应用下次 pull 时生效 -> 无需重新部署代码
#   这意味着 Prompt 的热更新能力——改完立刻生效，回滚也只需切换版本号
#
# 版本管理实践:
#   main 分支 = 生产环境当前使用的 Prompt
#   v1.0, v1.1, v2.0 = 历史版本，随时可回滚
#   新功能先在实验分支测试: experiment/prompt-v3 -> 测试通过后再合并到 main
#
# 与 LangSmith 结合:
#   Hub 的 Prompt 可以直接在 LangSmith 上做数据集评估
#   你可以看到"哪个版本的 Prompt 在哪个数据集上表现最好"
#   这是 Prompt Engineering 从"玄学"变成"科学"的关键
#
# 教学建议顺序:
#   1. 先理解"为什么需要 Hub"——Prompt 热更新、版本管理、团队协作
#   2. 再跑通 pull 示例，感受"一行代码从网络获取 Prompt"
#   3. 再尝试 push 自己的 Prompt，体验版本管理
#   4. 最后理解与 LangSmith 的结合——Prompt 的量化评估
