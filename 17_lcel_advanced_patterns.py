"""高级 LCEL 模式示例 —— .map, .reduce, .with_fallbacks, .with_retry, .pipe 等生产级模式。

LCEL (LangChain Expression Language) 不仅是管道符 |，
还提供丰富的组合原语来构建健壮的生产级管道。

运行前设置: export OPENAI_API_KEY="sk-..."
"""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 示例 1：.map() —— 批量处理
# ==============================================================================
print("=== 示例 1: .map() 批量处理 ===")

summarize_chain = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in one sentence."),
    ("human", "{text}"),
]) | model | StrOutputParser()

articles = [
    "LangChain is a framework for building LLM applications. It provides a standard interface for chains, prompts, and agents.",
    "Machine learning models can be trained on large datasets to recognize patterns in images, text, and audio.",
    "Climate change is causing rising sea levels and more frequent extreme weather events worldwide.",
]

# .map() 将链应用到列表的每个元素
summaries = summarize_chain.map(articles)
print(f"1. 批量摘要 ({len(summaries)} 篇):")
for i, summary in enumerate(summaries, 1):
    print(f"   {i}. {summary}")
print()


# ==============================================================================
# 示例 2：.reduce() —— 聚合结果
# ==============================================================================
print("=== 示例 2: .reduce() 聚合结果 ===")


def merge_summaries(summaries: list[str]) -> str:
    """将多个摘要合并为最终摘要。"""
    if len(summaries) == 1:
        return summaries[0]
    combined = "\n\n".join(summaries)
    final_chain = ChatPromptTemplate.from_messages([
        ("system", "Synthesize the following summaries into one comprehensive summary."),
        ("human", "{text}"),
    ]) | model | StrOutputParser()
    return final_chain.invoke({"text": combined})


# .reduce() 在 .map() 之后聚合结果
final_summary = summarize_chain.map(articles).reduce(merge_summaries)
print(f"2. 最终汇总: {final_summary}\n")


# ==============================================================================
# 示例 3：.with_fallbacks() —— 模型容错/多提供者冗余
# ==============================================================================
print("=== 示例 3: .with_fallbacks() 容错 ===")

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# 主模型 + 备用模型
primary = ChatOpenAI(model="gpt-4o-mini", temperature=0)
fallback1 = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
fallback2 = ChatOllama(model="llama3.2:1b", temperature=0)

# 构建带降级链
robust_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ])
    | primary
    | StrOutputParser()
    .with_fallbacks([
        fallback1.with_structured_output(str),
        fallback2.with_structured_output(str),
    ])
)

# 使用（如果主模型失败，自动降级）
result = robust_chain.invoke({"question": "What is the capital of France?"})
print(f"3. 容错链结果: {result}\n")


# ==============================================================================
# 示例 4：.with_retry() —— 自动重试临时错误
# ==============================================================================
print("=== 示例 4: .with_retry() 自动重试 ===")

retry_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize concisely."),
        ("human", "{text}"),
    ])
    | model.with_retry(
        retry_if_retryable_generation_failure=True,
        max_retries=3,
    )
    | StrOutputParser()
)

# 对于速率限制或临时网络错误，会自动重试最多3次
result = retry_chain.invoke({"text": "AI is transforming every industry."})
print(f"4. 重试链结果: {result}\n")


# ==============================================================================
# 示例 5：.pipe() —— 通过任意 Python 函数传递
# ==============================================================================
print("=== 示例 5: .pipe() 自定义处理 ===")


def add_sentiment(text: str) -> dict:
    """模拟添加情感分析。"""
    positive_words = ["great", "excellent", "amazing", "love", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst"]

    text_lower = text.lower()
    if any(w in text_lower for w in positive_words):
        sentiment = "positive"
    elif any(w in text_lower for w in negative_words):
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"text": text, "sentiment": sentiment}


# .pipe() 将模型输出传递给自定义函数
pipe_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Analyze the sentiment of this review."),
        ("human", "Review: {review}"),
    ])
    | model
    | StrOutputParser()
    .pipe(add_sentiment)
)

reviews = [
    "This product is amazing and I love it!",
    "Terrible quality, worst purchase ever.",
    "The product works as described.",
]

print(f"5. .pipe() 情感分析:")
for review in reviews:
    result = pipe_chain.invoke({"review": review})
    print(f"   Review: {review[:40]}...")
    print(f"   -> Sentiment: {result['sentiment']}")
print()


# ==============================================================================
# 示例 6：.with_config() —— 每调用覆盖配置
# ==============================================================================
print("=== 示例 6: .with_config() 运行时配置 ===")

# 为每次调用注入元数据和标签
tagged_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a translator. Translate to {language}."),
        ("human", "{text}"),
    ])
    | model
    | StrOutputParser()
    .with_config(
        tags=["translation", "production"],
        metadata={"version": "v2.1", "team": "nlp"},
    )
)

result = tagged_chain.invoke({"text": "Hello world", "language": "Japanese"})
print(f"6. 带标签链结果: {result}\n")


# ==============================================================================
# 示例 7：.with_handlers() —— 自定义事件处理器
# ==============================================================================
print("=== 示例 7: .with_handlers() 事件处理 ===")


def handle_start(run_id, run):
    print(f"   [START] Run {run_id}: {run.name}")


def handle_end(run_id, run):
    output = run.outputs.get("output", "") if run.outputs else ""
    print(f"   [END]   Run {run_id}: {output[:50]}...")


handler_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Translate to {language}."),
        ("human", "{text}"),
    ])
    | model
    | StrOutputParser()
    .with_handlers({
        "on_chat_model_start": handle_start,
        "on_chat_model_end": handle_end,
    })
)

print("   事件流:")
result = handler_chain.invoke({"text": "Good morning", "language": "French"})
print(f"   最终: {result}\n")


# ==============================================================================
# 示例 8：组合模式 —— 生产级健壮管道
# ==============================================================================
print("=== 示例 8: 生产级健壮管道 ===")

# 整合所有模式：批处理 + 容错 + 重试 + 自定义后处理 + 配置


class TextProcessor:
    """生产级文本处理管道。"""

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def classify(self, texts: list[str]) -> list[dict]:
        """批量分类文本。"""

        # 定义分类链
        classify_chain = (
            ChatPromptTemplate.from_messages([
                ("system", """Classify the text into one of these categories:
- technology: software, AI, computers
- science: biology, physics, chemistry
- business: finance, marketing, management
- health: medicine, wellness, fitness
- other: everything else

Return JSON: {{"category": "...", "confidence": 0.0-1.0}}"""),
                ("human", "{text}"),
            ])
            | self.model.with_structured_output(dict)
            .with_retry(max_retries=2)
            .with_fallbacks([
                self.model.with_structured_output(dict)
                .with_retry(max_retries=1),
            ])
        )

        # 批处理
        results = classify_chain.map(texts)

        # 后处理
        def normalize(result):
            return {
                "category": result.get("category", "unknown"),
                "confidence": min(max(result.get("confidence", 0.0), 0.0), 1.0),
            }

        normalized = [normalize(r) for r in results]

        # 汇总
        def summarize_classification(classifications: list[dict]) -> dict:
            categories = {}
            for c in classifications:
                cat = c["category"]
                categories[cat] = categories.get(cat, 0) + 1
            return {"total": len(classifications), "breakdown": categories}

        summary = summarize_classification(normalized)

        return {
            "classifications": normalized,
            "summary": summary,
        }


# 使用
processor = TextProcessor()
test_texts = [
    "The new AI model achieves state-of-the-art results on benchmarks.",
    "Scientists discover a new species of deep-sea fish.",
    "Quarterly revenue increased by 15% year over year.",
    "New study shows meditation reduces stress by 40%.",
    "The latest smartphone has a great camera.",
]

result = processor.classify(test_texts)
print(f"8. 生产级管道结果:")
for c in result["classifications"]:
    print(f"   {c['category']:15} (confidence: {c['confidence']:.2f})")
print(f"   汇总: {result['summary']}\n")


# ==============================================================================
# 示例 9：动态并行 —— .with_context 共享上下文
# ==============================================================================
print("=== 示例 9: .with_config() 共享上下文 ===")


def shared_context(input_data):
    """在链执行前注入共享上下文。"""
    return {
        **input_data,
        "user_id": "user_123",
        "request_id": "req_456",
        "timestamp": "2025-01-01T00:00:00Z",
    }


context_chain = (
    RunnableLambda(shared_context)
    | RunnableParallel({
        "response": ChatPromptTemplate.from_messages([
            ("system", "Respond to user. Context: {user_id}, Request: {request_id}"),
            ("human", "{input}"),
        ]) | model | StrOutputParser(),
        "metadata": RunnablePassthrough.assign(),
    })
)

result = context_chain.invoke({"input": "Hello"})
print(f"9. 共享上下文: {result['response'][:60]}...\n")


# ==============================================================================
# 教学备注：LCEL 高级原语 —— 从"能用"到"生产级"
# ==============================================================================
# .map() - 批量处理
#   是什么：将链应用到列表/可迭代对象的每个元素
#   为什么需要：真实数据总是批量来的——100篇文章需要100个摘要
#   性能：.map() 内部使用线程池并发执行，比手动 for 循环快
#   等价于：[chain.invoke(x) for x in items] 但自动并发 + 错误隔离
#
# .reduce() - 聚合结果
#   是什么：在 .map() 之后，将多个结果合并为单一结果
#   为什么需要：批量处理后需要汇总——100个摘要合并为1个总摘要
#   典型模式：map().reduce() = 分治法：分（并行处理）-> 治（合并结果）
#
# .with_fallbacks() - 模型容错
#   是什么：定义降级链，当主模型失败时自动尝试备用模型
#   为什么需要：LLM API 可能超时、限流、崩溃——生产系统不能因为一个模型失败就停摆
#   执行顺序：按顺序尝试，直到成功或全部失败
#   生产建议：至少准备一个免费/低成本的备用（如 Ollama），主模型降级时可用
#
# .with_retry() - 自动重试
#   是什么：对临时错误（速率限制、网络超时）自动重试
#   为什么需要：临时错误不应该导致整个管道失败
#   参数：max_retries（最大重试次数）、retry_if_retryable_generation_failure
#   指数退避：重试间隔自动递增，避免雪崩
#
# .pipe() - 自定义函数
#   是什么：将上游输出通过任意 Python 函数处理
#   为什么需要：LCEL 覆盖了80%的场景，但剩下20%需要自定义逻辑
#   与 RunnableLambda 的区别：.pipe() 是方法调用，更链式友好
#
# .with_config() - 运行时配置
#   是什么：为每次调用注入标签、元数据、回调等
#   为什么需要：生产系统中每次调用需要可追踪的上下文
#   典型用途：用户ID、租户ID、A/B测试组、环境标签
#
# .with_handlers() - 事件处理
#   是什么：注册自定义回调处理各种生命周期事件
#   为什么需要：需要监控、日志、指标收集的场合
#   可用事件：on_chain_start/end, on_chat_model_start/end, on_tool_start/end...
#
# 选型决策树:
#   ┌─ 需要批量处理？ → .map()
#   ├─ 需要聚合结果？ → .map().reduce()
#   ├─ 需要容错？ → .with_fallbacks()
#   ├─ 需要重试？ → .with_retry()
#   ├─ 需要自定义处理？ → .pipe()
#   ├─ 需要注入配置？ → .with_config()
#   └─ 需要事件监控？ → .with_handlers()
#
# 教学建议顺序:
#   1. 先理解 .map() + .reduce() —— 最实用的批处理模式
#   2. 再理解 .with_fallbacks() —— 生产系统必备
#   3. 再理解 .pipe() —— 自定义逻辑的通用接口
#   4. 最后整合到示例 8 —— 看到所有模式如何组合
