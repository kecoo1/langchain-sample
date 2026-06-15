"""多 LLM 提供商集成示例 —— Google Gemini、HuggingFace、Cohere。

LangChain 支持 50+ LLM 提供商。本文件展示三个重要且独特的提供商：
1. Google Gemini: 1M token 上下文窗口、原生多模态
2. HuggingFace: 开源模型推理、自定义微调模型
3. Cohere: 企业级 RAG、内置 citations 和 reranking

运行前设置:
  export OPENAI_API_KEY="sk-..." (OpenAI)
  export GOOGLE_API_KEY="your-google-key" (Gemini)
  export HF_TOKEN="your-hf-token" (HuggingFace)
  export COHERE_API_KEY="your-cohere-key" (Cohere)
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 对比基准：OpenAI
openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 示例 1：Google Gemini —— 超长上下文 + 多模态
# ==============================================================================
print("=== 示例 1: Google Gemini ===")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    # 初始化 Gemini 模型
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    )

    # 1.1 超长上下文支持 (1M token)
    print("1.1 超长上下文:")
    long_context = "This is a test. " * 5000  # 约 15K tokens
    result = gemini_model.invoke(f"Summarize this text: {long_context}")
    print(f"   摘要: {result.content[:100]}...\n")

    # 1.2 多模态（图像理解）
    print("1.2 多模态:")
    # gemini_multimodal = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash-exp",
    #     temperature=0,
    #     google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    # )
    #
    # from langchain_core.messages import HumanMessage
    # message = HumanMessage(content=[
    #     {"type": "text", "text": "What is in this image?"},
    #     {
    #         "type": "image_url",
    #         "image_url": "https://example.com/image.jpg",
    #     },
    # ])
    # result = gemini_multimodal.invoke([message])

    # 1.3 函数调用
    print("1.3 函数调用:")
    from langchain_core.tools import tool

    @tool
    def get_current_price(symbol: str) -> str:
        """Get the current stock price for a given symbol."""
        prices = {"AAPL": "$178.50", "GOOGL": "$141.80", "MSFT": "$378.90"}
        return prices.get(symbol, "Price not available")

    gemini_with_tools = gemini_model.bind_tools([get_current_price])
    result = gemini_with_tools.invoke("What is the price of AAPL?")
    if hasattr(result, "tool_calls") and result.tool_calls:
        print(f"   工具调用: {result.tool_calls}\n")
    else:
        print(f"   响应: {result.content}\n")

    print("   Gemini 特点:")
    print("   - 1M token 上下文窗口（可处理整本书）")
    print("   - 原生多模态（图像、音频、视频）")
    print("   - 原生函数调用支持")
    print("   - 免费额度充足")
    print()
except ImportError:
    print("   跳过: 需要安装 pip install langchain-google-genai\n")


# ==============================================================================
# 示例 2：HuggingFace —— 开源模型推理
# ==============================================================================
print("=== 示例 2: HuggingFace ===")

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    # 方式 1: HuggingFace Inference API（托管）
    # 需要 HF_TOKEN 环境变量
    # hf_model = HuggingFaceEndpoint(
    #     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    #     huggingfacehub_api_token=os.environ.get("HF_TOKEN", ""),
    #     temperature=0,
    # )
    # result = hf_model.invoke("What is LangChain?")
    # print(f"   HF Inference API: {result}\n")

    # 方式 2: 通过 ChatHuggingFace 包装
    # hf_chat = ChatHuggingFace(
    #     llm=hf_model,
    #     model_id="mistralai/Mistral-7B-Instruct-v0.3",
    # )
    # result = hf_chat.invoke("Translate to French: Hello world")
    # print(f"   ChatHuggingFace: {result.content}\n")

    # 方式 3: 本地 Ollama 运行开源模型（推荐本地方案）
    from langchain_ollama import ChatOllama
    ollama_model = ChatOllama(model="llama3.2:1b", temperature=0)
    result = ollama_model.invoke("What is AI?")
    print(f"   Ollama (本地 Llama3): {result.content[:80]}...\n")

    print("   HuggingFace 特点:")
    print("   - 5000+ 开源模型可用")
    print("   - 可微调自定义模型")
    print("   - Inference API 免费额度")
    print("   - 本地部署需 GPU 资源")
    print()
except ImportError:
    print("   跳过: 需要安装 pip install langchain-huggingface\n")


# ==============================================================================
# 示例 3：Cohere —— 企业级 RAG 和 Reranking
# ==============================================================================
print("=== 示例 3: Cohere ===")

try:
    from langchain_cohere import ChatCohere
    from langchain_cohere.retrievers import CohereRerank

    cohere_model = ChatCohere(
        model="command-r-plus",
        temperature=0,
    )

    # 3.1 聊天
    result = cohere_model.invoke("What is the difference between RAG and fine-tuning?")
    print(f"   1. 聊天: {result.content[:100]}...\n")

    # 3.2 Cohere Rerank（重排序）
    print("3.2 Cohere Rerank:")
    from langchain_core.documents import Document

    docs = [
        Document(page_content="LangChain is a framework for LLM applications."),
        Document(page_content="Python is a programming language."),
        Document(page_content="LangGraph is for building stateful agent workflows."),
        Document(page_content="The weather is sunny today."),
        Document(page_content="Machine learning uses statistical techniques to give computers learning capability."),
    ]

    # 先检索（假设返回了 Top-5）
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # retrieved = retriever.invoke("What is LangChain?")

    # 再用 Cohere Rerank 精排
    # reranker = CohereRerank(top_n=3)
    # reranked = reranker.compress_documents(
    #     documents=docs,
    #     query="What is LangChain?"
    # )
    # print(f"   重排序后:")
    # for i, doc in enumerate(reranked, 1):
    #     print(f"   {i}. {doc.page_content[:60]}...")
    # print()

    # 3.3 Cohere 原生 citations
    print("3.3 Cohere 原生引用:")
    # 使用 citations=True 让 Cohere 返回引用来源
    # result = cohere_model.invoke(
    #     "What is LangChain?",
    #     citations=True,
    #     documents=[
    #         {"title": "LangChain Docs", "text": "LangChain is a framework..."},
    #         {"title": "AI Wiki", "text": "AI frameworks help build LLM apps..."},
    #     ],
    # )
    # print(f"   回答: {result.content}")
    # print(f"   引用: {result.citations}")
    # print()

    print("   Cohere 特点:")
    print("   - 企业级 RAG 优化")
    print("   - 内置 Rerank 模型")
    print("   - 原生 citations 支持")
    print("   - command-r-plus 模型专为对话优化")
    print()
except ImportError:
    print("   跳过: 需要安装 pip install langchain-cohere\n")


# ==============================================================================
# 示例 4：Provider 对比和统一接口
# ==============================================================================
print("=== 示例 4: 统一接口 ===")

# LangChain 的核心价值：统一不同 LLM 提供商的接口
providers = {
    "OpenAI": ChatOpenAI(model="gpt-4o-mini", temperature=0),
}

try:
    providers["Gemini"] = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ.get("GOOGLE_API_KEY", ""))
except:
    pass

try:
    providers["Ollama"] = ChatOllama(model="llama3.2:1b", temperature=0)
except:
    pass

try:
    providers["Cohere"] = ChatCohere(model="command-r-plus", temperature=0)
except:
    pass

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer in one sentence."),
    ("human", "{question}"),
])

common_question = "What is LangChain?"

print(f"   统一问题: {common_question}\n")

for name, model_instance in providers.items():
    chain = prompt | model_instance | StrOutputParser()
    try:
        result = chain.invoke({"question": common_question})
        print(f"   {name:12} | {result[:60]}")
    except Exception as e:
        print(f"   {name:12} | 不可用: {str(e)[:40]}")

print("\n   统一接口的好处:")
print("   - 相同代码调用不同模型")
print("   - 易于 A/B 测试和比较")
print("   - 模型切换只需改一行配置")
print()


# ==============================================================================
# 示例 5：模型回退策略
# ==============================================================================
print("=== 示例 5: 模型回退 ===")

# 整合多个提供商，构建健壮的多提供商回退链

# 主模型: OpenAI (高质量)
primary = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 备用1: Gemini (高质量 + 便宜)
try:
    fallback1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ.get("GOOGLE_API_KEY", ""))
except:
    fallback1 = None

# 备用2: Ollama (本地免费)
try:
    fallback2 = ChatOllama(model="llama3.2:1b", temperature=0)
except:
    fallback2 = None

# 构建回退链
robust_chain = (
    prompt
    | primary
    | StrOutputParser()
    .with_fallbacks(
        [fb for fb in [fallback1, fallback2] if fb is not None]
    )
)

try:
    result = robust_chain.invoke({"question": "What is LangChain in one sentence?"})
    print(f"   回退链结果: {result}\n")
except Exception as e:
    print(f"   回退链错误: {e}\n")


# ==============================================================================
# 教学备注：多提供商集成 —— 不要把所有鸡蛋放在一个篮子里
# ==============================================================================
# 核心问题：为什么需要多提供商支持？
#   1. 成本优化：不同模型价格差异大，简单任务用便宜模型
#   2. 可靠性：一个提供商 API 挂了，可以切换到另一个
#   3. 能力互补：不同模型在不同任务上有优势
#   4. 合规要求：某些数据可能要求使用本地/自有模型
#
# Google Gemini 独特优势:
#   - 1M token 上下文窗口：可以处理整本书、长代码库
#   - 原生多模态：原生支持图像、音频、视频输入
#   - 免费额度：每月 60 次/分钟请求（免费版）
#   - 函数调用：原生支持，与 OpenAI 兼容
#
# HuggingFace 独特优势:
#   - 5000+ 开源模型：从 Llama 到 Mistral 到 BERT
#   - 可微调：上传自己的数据训练定制模型
#   - 本地部署：用 Ollama 或 vLLM 本地运行
#   - 无 API 限制：自托管，无速率限制
#
# Cohere 独特优势:
#   - 企业级 RAG：专为 RAG 优化
#   - 原生 citations：回答自动标注引用来源
#   - Rerank 模型：业界最强的重排序之一
#   - command-r: 专为 RAG 和工具调用优化
#
# 提供商选型决策树:
#   高质量对话: OpenAI GPT-4o / Claude / Gemini
#   低成本批量: Ollama (本地) / Cohere (RAG)
#   超长上下文: Gemini (1M tokens)
#   多模态: Gemini (原生) / OpenAI (gpt-4o)
#   开源/本地: HuggingFace + Ollama
#   企业 RAG: Cohere + rerank
#
# 教学建议顺序:
#   1. 先理解"为什么需要多提供商"——成本、可靠性、能力
#   2. 跑通 OpenAI + Ollama 对比（本地免费）
#   3. 再理解各提供商的独特优势
#   4. 最后理解 .with_fallbacks() 多提供商回退模式
