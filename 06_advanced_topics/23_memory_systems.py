"""LangChain 长期记忆系统 —— 滑动窗口、摘要、向量语义、实体记忆。

Agent 的"记忆"是区分"玩具"和"生产系统"的关键能力之一。
本文件覆盖 5 种记忆系统及组合使用策略：
1. 滑动窗口记忆（BufferWindowMemory）：保留最近 N 轮对话
2. 摘要记忆（SummaryBufferMemory）：旧对话自动压缩为摘要
3. 向量检索记忆（VectorStoreRetrieverMemory）：按语义搜索历史对话
4. 实体记忆（EntityMemory）：提取和追踪对话中的实体
5. 组合记忆（CombinedMemory）：多种记忆协同工作
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    ConversationEntityMemory,
    CombinedMemory,
)
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 示例 1：滑动窗口记忆 —— 只记住最近 N 轮
# ==============================================================================
print("=== 示例 1: 滑动窗口记忆 ===")

# ConversationBufferWindowMemory 只保留最近的 K 轮对话
# 超出窗口的旧对话自动丢弃 —— 控制 Token 消耗的"简单粗暴"方案
window_memory = ConversationBufferWindowMemory(
    k=2,  # 只保留最近 2 轮（4 条消息：2 Human + 2 AI）
    return_messages=True,
)

# 写入 3 轮对话
for i in range(3):
    window_memory.chat_memory.add_user_message(f"Message {i + 1}")
    window_memory.chat_memory.add_ai_message(f"Reply {i + 1}")

# 第 1 轮对话（最早）已被丢弃
messages = window_memory.load_memory_variables({})["history"]
print(f"1. 滑动窗口 (k=2) 当前记忆: {len(messages)} 条消息")
print(f"   内容: {[m.content for m in messages]}")
print(f"   可以看到，最早的 Message 1 已被丢弃\n")

print("   适用场景：短对话、客服快速轮转、Token 预算严格")
print()

# ==============================================================================
# 示例 2：摘要记忆 —— 旧对话自动压缩
# ==============================================================================
print("=== 示例 2: 摘要记忆 ===")

# ConversationSummaryBufferMemory 在窗口内保留完整对话，超出窗口时自动用 LLM 压缩为摘要
# 切换阈值由 max_token_limit 控制
summary_memory = ConversationSummaryBufferMemory(
    llm=model,
    max_token_limit=200,  # 超过 200 token 时触发摘要
    return_messages=True,
)

# 模拟较长对话
long_story = """I visited Tokyo last week. The weather was amazing, around 25°C.
I went to Shibuya crossing and it was incredibly busy.
Then I tried sushi at Tsukiji market - best sushi I've ever had.
I also visited the Meiji Shrine which was very peaceful."""

summary_memory.save_context(
    {"input": "Tell me about your trip to Japan."},
    {"output": long_story},
)

# 第二条消息（尚未触发摘要，总 token 未超过 200）
summary_memory.save_context(
    {"input": "What about the food?"},
    {"output": "The ramen was excellent too! I had it in Shinjuku."},
)

variables = summary_memory.load_memory_variables({})
print(f"2. 摘要记忆:")
print(f"   历史长度: {len(variables['history'])} 条消息")
print(f"   ...继续追加消息会触发自动摘要\n")

print("   适用场景：长对话、需要保留上下文细节的复杂任务")
print()

# ==============================================================================
# 示例 3：向量检索记忆 —— 按语义搜索历史对话
# ==============================================================================
print("=== 示例 3: 向量检索记忆 ===")

try:
    from langchain.memory import VectorStoreRetrieverMemory
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document

    # 带语义搜索的类人记忆：不像滑动窗口只记得"最近的"，而是记得"最相关的"
    # 原理：每次对话存入向量库，检索时用语义相似度查找
    vector_store = Chroma(
        collection_name="memory_store",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    vector_memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="history",
        return_messages=True,
    )

    # 存储一些对话
    vector_memory.save_context(
        {"input": "My favorite programming language is Python"},
        {"output": "Great choice! Python is versatile and has a great ecosystem."},
    )
    vector_memory.save_context(
        {"input": "I work at a fintech company"},
        {"output": "Fintech is fascinating. Are you working on payment systems?"},
    )
    vector_memory.save_context(
        {"input": "I love hiking on weekends"},
        {"output": "Hiking is a great way to stay active. Do you have favorite trails?"},
    )

    # 搜索相关的记忆 —— 查询"coding"会匹配"Python"相关的记忆
    relevant = vector_memory.load_memory_variables(
        {"input": "What do you know about my coding preferences?"}
    )
    print(f"3. 向量检索记忆:")
    print(f"   查询: 'What do you know about my coding preferences?'")
    print(f"   检索到: {relevant['history'][:200]}...\n")

    # 清理
    vector_store.delete_collection()

    print("   适用场景：长周期对话（数月）、个性化服务、学习型 Agent")
    print()

except ImportError:
    print("   跳过: 需要 pip install langchain-chroma\n")

# ==============================================================================
# 示例 4：实体记忆 —— 提取和追踪实体信息
# ==============================================================================
print("=== 示例 4: 实体记忆 ===")

try:
    # ConversationEntityMemory 自动从对话中提取实体（人名、地点、喜好等）
    # 并维护一个"实体 -> 信息"的映射表
    entity_memory = ConversationEntityMemory(
        llm=model,
        return_messages=True,
    )

    # 第一次对话 —— 提取用户姓名和城市
    entity_memory.save_context(
        {"input": "Hi, I'm Alice from New York"},
        {"output": "Nice to meet you Alice! How's New York?"},
    )

    # 第二次对话 —— 提取用户的宠物信息
    entity_memory.save_context(
        {"input": "I have a golden retriever named Max"},
        {"output": "Golden retrievers are lovely! Max sounds adorable."},
    )

    entities = entity_memory.load_memory_variables({})
    print(f"4. 实体记忆:")
    print(f"   已提取实体: {entities.get('entities', 'None')}")
    print(f"   实体映射: {entities.get('entity_store', 'None')}")
    print()

    # 现在我们可以直接通过实体名查询用户信息（无需语义搜索）
    # entity_memory.entity_store.get("Alice") -> "Alice lives in New York"
    # entity_memory.entity_store.get("Max") -> "Max is Alice's golden retriever"

    print("   适用场景：CRM 系统、个人助理、需要长期记住用户画像的场景")
    print()

except Exception as e:
    print(f"   跳过: {str(e)[:80]}\n")

# ==============================================================================
# 示例 5：组合记忆 —— 多种记忆协同工作
# ==============================================================================
print("=== 示例 5: 组合记忆 ===")

try:
    from langchain.memory import ConversationBufferMemory

    # 生产中，单一记忆类型往往不够
    # 组合记忆 = 滑动窗口（近期细节） + 摘要（长程摘要） + 实体（关键信息）
    conv_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    combined = CombinedMemory(
        memories=[conv_memory, summary_memory]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use conversation history for context."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    print(f"5. 组合记忆:")
    print(f"   组合 {len(combined.memories)} 种记忆:")
    print(f"   - ConversationBufferMemory: 当前轮对话")
    print(f"   - ConversationSummaryBufferMemory: 长程摘要")
    print(f"   (生产中可以加入 EntityMemory/VectorStoreMemory)")
    print()

    # 完整组合记忆链
    from langchain.chains import LLMChain

    chain = LLMChain(
        llm=model,
        prompt=prompt,
        memory=combined,
        verbose=False,
    )

    # 测试
    result = chain.invoke({"input": "What do you remember about me?"})
    print(f"   链式调用结果: {result['text'][:150]}...\n")

    print("   适用场景：需要同时保留细节+摘要+实体的复杂 Agent")
    print()

except Exception as e:
    print(f"   跳过: {str(e)[:80]}\n")

# ==============================================================================
# 教学备注：记忆系统 —— 给 Agent 一个"大脑皮层"
# ==============================================================================
# 核心问题：Agent 默认没有记忆——每次对话都是全新的，就像失忆症患者
#
# LangChain 记忆系统的分层结构：
#
#   ChatMessageHistory（最底层）
#     → 原始消息存储（内存/Redis/PostgreSQL/SQLite）
#     → 不支持任何"处理"，只是存和取
#     → 类似：数据库表
#
#   Memory（中间层）
#     → 在 ChatMessageHistory 之上增加处理逻辑
#     → 滑动窗口：截断（BufferWindow）
#     → 摘要：压缩（BufferSummary）
#     → 语义：检索（VectorStoreRetriever）
#     → 实体：提取（Entity）
#     → 类似：视图/物化视图
#
#   CombinedMemory（最上层）
#     → 组合多种 Memory，每个负责一个维度
#     → 类似：分库分表 + 汇总查询
#
# 选型决策树：
#
#   对话轮次 < 10 轮？
#   ├── 是 → ConversationBufferMemory (不丢任何消息)
#   └── 否
#       ├── Token 预算紧张？→ BufferWindowMemory (固定大小)
#       ├── 需要长程理解？→ SummaryBufferMemory (自动摘要)
#       ├── 需要语义搜索？→ VectorStoreRetrieverMemory (类人记忆)
#       └── 需要实体追踪？→ EntityMemory (CRM 场景)
#
# 生产级记忆架构建议:
#   - 短期记忆：ConversationBufferMemory（当前对话，窗口 10-20 轮）
#   - 中期记忆：SummaryBufferMemory（压缩摘要，LLM 定期生成）
#   - 长期记忆：VectorStoreRetrieverMemory（语义搜索，持久化到向量库）
#   - 关键事实：EntityMemory（姓名、偏好、重要信息）
#
# 持久的存储后端（ChatMessageHistory 的实现）：
#   - InMemoryChatMessageHistory：最快，进程重启丢失（开发用）
#   - RedisChatMessageHistory：跨进程共享，适合分布式（推荐）
#   - PostgresChatMessageHistory：持久化，可审计（推荐）
#   - SQLChatMessageHistory：SQLite 最简单，PostgreSQL 最可靠
#   - MongoDBChatMessageHistory：文档数据库，灵活
#
# 与 01_basics/05_agent.py 的关系：
#   05 中的 memory=ConversationSummaryBufferMemory 是"开箱即用"方案
#   本文件是"选型指南"——理解每种记忆的优劣，按需选择组合
#   生产系统中，AgentExecutor + memory 是最常用模式
#
# 教学建议顺序：
#   1. 先理解"为什么需要各种记忆"——不同场景需要不同记忆策略
#   2. 跑通示例 1-4，理解每种记忆的原理
#   3. 理解示例 5 的组合模式——生产中的真实用法
#   4. 考虑记忆的后端存储（Redis/PostgreSQL 替代 InMemory）
