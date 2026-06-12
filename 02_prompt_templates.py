"""提示模板与消息组合示例。"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
# 模型实例
# model = ChatOpenAI(model="gpt-4o-mini")

model = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
# 1. 基本字符串提示
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Tell me a {adjective} joke about {topic}."),
])
chain = prompt | model
result = chain.invoke({"adjective": "funny", "topic": "programming"})
print(f"1. Basic prompt: {result.content}\n")

# 2. Few-shot 提示示例
few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Translate English to French."),
    ("human", "Hello"),
    ("ai", "Bonjour"),
    ("human", "Good morning"),
    ("ai", "Bonjour"),
    ("human", "How are you?"),
    ("ai", "Comment allez-vous ?"),
    ("human", "I love programming"),
])
chain2 = few_shot_prompt | model
result2 = chain2.invoke({})
print(f"2. Few-shot: {result2.content}\n")

# 3. 带历史的动态对话示例
history_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain3 = history_prompt | model
result3 = chain3.invoke({
    "history": [
        HumanMessage(content="My name is Alice"),
        AIMessage(content="Nice to meet you, Alice!"),
    ],
    "input": "What is my name?",
})
print(f"3. With history: {result3.content}\n")

# 4. 带部分变量的提示模板
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "Explain {concept} in simple terms."),
])
partial_prompt = template.partial(domain="machine learning")
chain4 = partial_prompt | model
result4 = chain4.invoke({"concept": "transformers"})
print(f"4. Partial variables: {result4.content}")

# =============================================================================
# 教学备注：提示模板的四种核心构建模式——从"是什么"到"为什么这么设计"
# =============================================================================
# 核心问题：为什么不直接拼字符串？为什么要引入模板系统？
# 答案：字符串拼接会导致——注入攻击风险、格式不规范、难以复用、无法序列化/版本控制
# 模板系统的核心价值：结构化 + 安全插值 + 可组合 + 可序列化(LangSmith追踪)
#
# 模式 1：ChatPromptTemplate.from_messages([...]) - 结构化消息列表（最通用）
#   是什么：显式声明每条消息的角色，支持 system/human/ai/tool/function 四大角色
#   为什么需要：LLM 只理解"角色化对话"格式；system 设定人设、human 提问、ai 示例、tool 返回结果
#   核心设计：元组列表，每个元组 = (角色, 内容模板)，内容模板内可用 {variable} 插值
#   关键能力：支持 MessagesPlaceholder 动态注入历史消息列表（这是实现记忆的关键）
#   适用场景：所有多轮对话、Few-shot 示例、需要 system prompt 的场景、工具调用代理
#   教学重点：这是"正统"用法，其他方式本质都是它的语法糖或特化
#
# 模式 2：ChatPromptTemplate.from_template("string") - 单轮快捷方式
#   是什么：接收单个字符串，自动包装为 [("human", template)] 的简化形式
#   为什么需要：单轮问答太常见了，不想每次写 system、元组、列表
#   本质局限：只能产生一条 human 消息，无法设定 system、无法 Few-shot、无法注入历史
#   适用场景：原型验证、简单分类/抽取任务、无上下文的一次性调用
#   教学建议：教学/原型用它；生产多轮对话务必升级为 from_messages
#
# 模式 3：.partial(**kwargs) - 模板复用与上下文固化
#   是什么：预先绑定部分变量，返回新模板对象，调用时只需传剩余变量
#   为什么需要：同一模板在不同场景复用（如"领域专家"system prompt固定，只换问题）
#   核心价值：模板即配置，可序列化存储、版本管理、A/B 测试不同 system prompt
#   陷阱：partial 是惰性的，生成新对象；调试时打印模板要注意已绑定/未绑定变量
#   适用场景：多租户系统（租户级 system prompt）、多语言模板、风格迁移模板
#
# 模式 4：MessagesPlaceholder(variable_name="history") - 动态历史注入槽位
#   是什么：在模板中预留"消息列表插槽"，运行时传入 List[BaseMessage]
#   为什么需要：对话历史长度不固定，不能硬编码在模板里；需配合 RunnableWithMessageHistory 自动读写存储
#   关键协作：Placeholder 只负责"位置"，不负责"存储"；存储由 RunnableWithMessageHistory + BaseChatMessageHistory 实现
#   适用场景：所有需要上下文感知的对话、代理记忆、RAG 对话式检索
#
# 进阶理解：模板即数据结构
#   - 模板可 .to_json() 序列化，存数据库、配置中心、Git 版本控制
#   - 模板可 .partial() 衍生，形成模板族谱（基础模板 -> 专用模板 -> 实例化模板）
#   - 模板可 | 组合：prompt1 + prompt2 形成管道，支持复杂预处理链
#
# 选型决策树：
#   ┌─ 单轮无历史、无 system？ → from_template
#   ├─ 多轮/需 system/需 Few-shot？ → from_messages
#   ├─ 同一结构不同固定上下文？ → from_messages + partial()
#   └─ 需要注入动态历史？ → from_messages + MessagesPlaceholder
