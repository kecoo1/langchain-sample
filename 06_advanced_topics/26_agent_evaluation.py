"""Agent 评估指标 —— 量化 LLM 应用的效果。

你的 Agent 到底好不好？不能凭感觉，需要量化评估。
本文件覆盖 4 个核心评估维度：
1. 在环评估（LangSmith）：追踪 + 人工标注 + 统计
2. 离线评估（RAGAS）：Faithfulness / Answer Relevancy / Context Precision
3. 自定义评估器：针对业务场景编写专属评估函数
4. 对比评估：Prompt 版本对比 + 模型对比
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 示例 1：在环评估 —— 人工标注 + 统计
# ==============================================================================
print("=== 示例 1: 在环评估体系 ===")

# 在环评估（Human-in-the-loop Evaluation）是最可靠的评估方式
# 原理：用户每条回答给 👍/👎，汇集统计得出系统质量
# 适用于：客服、搜索、推荐等最终用户直接反馈的场景

# 模拟用户反馈数据
feedbacks = [
    {"query": "What is LangChain?", "response": "LangChain is a framework...", "rating": 5},
    {"query": "How to install it?", "response": "Run pip install...", "rating": 4},
    {"query": "Explain transformers", "response": "Transformers are...", "rating": 2},
    {"query": "What is RAG?", "response": "RAG stands for...", "rating": 5},
    {"query": "Compare GPT and Claude", "response": "Both are...", "rating": 3},
]


def calculate_online_metrics(feedbacks: list[dict]) -> dict:
    """计算在环评估指标。"""
    ratings = [f["rating"] for f in feedbacks]

    metrics = {
        "total_queries": len(feedbacks),
        "avg_rating": sum(ratings) / len(ratings),
        "satisfaction_rate": sum(1 for r in ratings if r >= 4) / len(ratings),
        "unsatisfaction_rate": sum(1 for r in ratings if r <= 2) / len(ratings),
        "rating_distribution": {
            "5 (very good)": ratings.count(5),
            "4 (good)": ratings.count(4),
            "3 (neutral)": ratings.count(3),
            "2 (bad)": ratings.count(2),
            "1 (very bad)": ratings.count(1),
        },
    }
    return metrics


metrics = calculate_online_metrics(feedbacks)
print(f"1. 在环评估统计:")
print(f"   - 总查询: {metrics['total_queries']}")
print(f"   - 平均评分: {metrics['avg_rating']:.2f} / 5")
print(f"   - 满意度: {metrics['satisfaction_rate']:.0%}")
print(f"   - 不满意度: {metrics['unsatisfaction_rate']:.0%}")
print(f"   - 评分分布: {metrics['rating_distribution']}")
print()

print("   在环评估的 3 个关键设计:")
print("   1. 收集时机：回答后立即收集（用户记忆最新鲜）")
print("   2. 收集便捷性：一个点击（👍/👎）不要复杂表单")
print("   3. 回访机制：不满意回答可以追问收集细节")
print()


# ==============================================================================
# 示例 2：RAGAS —— 自动化 RAG 评估
# ==============================================================================
print("=== 示例 2: RAGAS 离线评估 ===")

# RAGAS (Retrieval Augmented Generation Assessment) 是一套自动化 RAG 评估框架
# 不需要人工标注，只需要：question + answer + contexts + ground_truth

try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.dataset_schema import SingleTurnSample

    # 初始化 RAGAS
    evaluator_llm = LangchainLLMWrapper(model)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # 模拟评估数据
    test_question = "What is the main purpose of LangChain?"
    test_answer = "LangChain is a framework designed to simplify the development of applications using large language models."
    test_contexts = [
        Document(page_content="LangChain is a framework for building LLM applications. It provides tools for prompt management, model integration, and agent orchestration."),
        Document(page_content="LangChain helps developers connect LLMs to external data sources and create complex workflows."),
    ]
    test_ground_truth = "LangChain is a framework for building LLM-powered applications, providing tools for prompt engineering, model integration, and workflow orchestration."

    sample = SingleTurnSample(
        user_input=test_question,
        response=test_answer,
        retrieved_contexts=[doc.page_content for doc in test_contexts],
        reference=test_ground_truth,
    )

    print("2. RAGAS 评估指标:")

    # Faithfulness（忠实度）：回答是否基于提供的上下文，而非模型幻觉
    # 越高越好（0-1），表示回答中的每个事实都能从上下文中找到依据
    try:
        faithfulness_score = faithfulness.score(sample)
        print(f"   - Faithfulness (忠实度): {faithfulness_score:.2f}")
        print(f"     含义: 回答内容有多大比例来源于提供的文档")
    except Exception as e:
        print(f"   - Faithfulness: 计算错误 {str(e)[:50]}")

    # Answer Relevancy（回答相关性）：回答与问题的匹配程度
    # 越高越好（0-1），表示回答直接回应了问题，没有离题
    try:
        relevancy_score = answer_relevancy.score(sample)
        print(f"   - Answer Relevancy: {relevancy_score:.2f}")
        print(f"     含义: 回答内容与问题的相关程度")
    except Exception as e:
        print(f"   - Answer Relevancy: 计算错误 {str(e)[:50]}")

    # Context Precision（上下文精确度）：检索结果中有多少是相关的
    # 越高越好（0-1），表示检索结果噪声少，精准命中
    try:
        precision_score = context_precision.score(sample)
        print(f"   - Context Precision: {precision_score:.2f}")
        print(f"     含义: 检索到的文档中，真正有用的比例")
    except Exception as e:
        print(f"   - Context Precision: 计算错误 {str(e)[:50]}")

    # Context Recall（上下文召回率）：需要的信息是否都被检索到了
    # 越高越好（0-1），表示没有遗漏关键信息
    try:
        recall_score = context_recall.score(sample)
        print(f"   - Context Recall: {recall_score:.2f}")
        print(f"     含义: 回答问题所需的信息是否都被检索到了")
    except Exception as e:
        print(f"   - Context Recall: 计算错误 {str(e)[:50]}")

    print()

except ImportError:
    print("   跳过: 需要 pip install ragas\n")

# ==============================================================================
# 示例 3：自定义评估器 —— 针对业务场景
# ==============================================================================
print("=== 示例 3: 自定义评估器 ===")

# 标准指标（如 RAGAS）覆盖通用场景
# 业务场景需要自定义评估器：检查 Agent 行为的正确性

from pydantic import BaseModel, Field


class CustomEvaluation(BaseModel):
    score: int = Field(ge=1, le=5, description="Score from 1 (worst) to 5 (best)")
    pros: list[str] = Field(description="Positive aspects")
    cons: list[str] = Field(description="Areas for improvement")
    verdict: str = Field(description="pass / fail / needs_review")


# 构建自定义评估器 prompt
evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert evaluator of LLM responses.
Evaluate the given response based on these criteria:
1. Correctness: Is the answer technically accurate?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it well-structured and easy to understand?
4. Safety: Does it avoid harmful or misleading content?

Provide scores and detailed feedback."""),
    ("human", "Question: {question}\n\nResponse: {response}"),
])

custom_evaluator = evaluator_prompt | model.with_structured_output(CustomEvaluation)

# 测试 1：好的回答
eval_1 = custom_evaluator.invoke({
    "question": "What is the difference between TCP and UDP?",
    "response": """TCP is connection-oriented and guarantees reliable delivery through acknowledgment and retransmission.
UDP is connectionless and prioritizes speed over reliability, making it suitable for real-time applications like video streaming.
Key differences:
- TCP ensures packet order; UDP does not
- TCP has higher latency; UDP has lower latency
- TCP is used for web, email; UDP is used for DNS, VoIP""",
})
print(f"3a. 高质量回答:")
print(f"    评分: {eval_1.score}/5")
print(f"    优点: {eval_1.pros[:2]}")
print(f"    结论: {eval_1.verdict}")
print()

# 测试 2：差的回答（不安全 + 不准确）
eval_2 = custom_evaluator.invoke({
    "question": "How do I make my code run faster?",
    "response": "Just use threading, it will make everything faster. Don't worry about thread safety, it usually works fine.",
})
print(f"3b. 低质量回答（错误建议）:")
print(f"    评分: {eval_2.score}/5")
print(f"    缺点: {eval_2.cons[:2]}")
print(f"    结论: {eval_2.verdict}")
print()


# ==============================================================================
# 示例 4：对比评估 —— Prompt 版本对比 + 模型对比
# ==============================================================================
print("=== 示例 4: 对比评估 ===")

# A/B 测试是评估 Prompt/模型 效果的重要方法
# 在 LangSmith 中，你可以运行测试数据集，对比不同配置的评分

test_questions = [
    "What is RAG in LLM?",
    "Explain machine learning in simple terms.",
    "What's the best way to store passwords?",
]

# Prompt 版本 A：简洁版
prompt_a = ChatPromptTemplate.from_messages([
    ("system", "Answer concisely."),
    ("human", "{question}"),
])

# Prompt 版本 B：详细版
prompt_b = ChatPromptTemplate.from_messages([
    ("system", "Answer in detail with examples. Think step by step."),
    ("human", "{question}"),
])

chain_a = prompt_a | model | StrOutputParser()
chain_b = prompt_b | model | StrOutputParser()

print(f"4. Prompt A/B 对比评估:")
print(f"   测试问题数: {len(test_questions)}")
print()

# 模拟对比评估（每次运行并记录）
evaluator_short = evaluator_prompt | model.with_structured_output(CustomEvaluation)

results_comparison = []
for q in test_questions:
    response_a = chain_a.invoke({"question": q})
    response_b = chain_b.invoke({"question": q})

    eval_a = evaluator_short.invoke({"question": q, "response": response_a.content if hasattr(response_a, 'content') else str(response_a)})
    eval_b = evaluator_short.invoke({"question": q, "response": response_b.content if hasattr(response_b, 'content') else str(response_b)})

    results_comparison.append({
        "question": q,
        "prompt_a_score": eval_a.score,
        "prompt_b_score": eval_b.score,
        "winner": "A" if eval_a.score > eval_b.score else "B" if eval_b.score > eval_a.score else "tie",
    })
    print(f"   Q: {q[:40]}...")
    print(f"   Prompt A (简洁): {eval_a.score}/5 | Prompt B (详细): {eval_b.score}/5")
    print(f"   胜出: Prompt {results_comparison[-1]['winner']}")
    print()

# 汇总
a_wins = sum(1 for r in results_comparison if r["winner"] == "A")
b_wins = sum(1 for r in results_comparison if r["winner"] == "B")
ties = sum(1 for r in results_comparison if r["winner"] == "tie")

print(f"   汇总: A 胜 {a_wins} / B 胜 {b_wins} / 平局 {ties}")
print()

print("   对比评估的价值：")
print("   - Prompt 优化：哪个版本效果更好？不是靠感觉，是量化对比")
print("   - 模型对比：GPT-4o vs Claude vs Gemini，在同一任务上的表现")
print("   - 成本 vs 质量：便宜模型的评分差距是否可接受")
print("   - 回归检测：修改 prompt 后，其他能力是否有下降")
print()


# ==============================================================================
# 教学备注：Agent 评估 —— 量化你的 Agent 质量
# ==============================================================================
# 核心问题：Agent 是"随机"的——同样的输入可能得到完全不同的输出
# 所以不能靠"测试一次感觉还行"来判断质量
#
# 评估体系金字塔：
#
#                ┌──────────────┐
#                │ 业务指标     │ ← ROI、用户留存、转化率
#               ┌┴──────────────┴┐
#               │ 在环评估       │ ← 用户反馈 (👍/👎)、NPS
#              ┌┴────────────────┴┐
#              │ 离线评估         │ ← RAGAS (Faithfulness, Relevancy)
#             ┌┴──────────────────┴┐
#             │ 单元测试           │ ← 确定性测试用例
#            ┌┴────────────────────┴┐
#            │ 冒烟测试            │ ← 基础功能（模型可调用、响应非空）
#
# LangSmith 的评估体系：
#
#   数据集（Dataset）
#     └── 测试用例：question, ground_truth, contexts
#           │
#           ▼
#   运行（Run）—— 用目标 Chain/Agent 生成回答
#     └── 记录：question, response, latency, token_count
#           │
#           ▼
#   评估器（Evaluator）—— 对每个 Run 打分
#     ├── 内置：Faithfulness, Relevancy, Precision, Recall
#     ├── 自定义：你的业务评分函数
#     └── 人工：人类标注
#           │
#           ▼
#   对比表（Comparison View）
#     └── 不同配置（Prompt 版本 / 模型 / 参数）的评分对比
#
# RAGAS 四指标详解：
#
#   Faithfulness（忠实度）
#     概念：回答中的事实是否都能从上下文中找到
#     公式：支持的事实 / 总事实数
#     问题：回答编造了一个文档中没有的统计数据 → 扣分
#
#   Answer Relevancy（回答相关性）
#     概念：回答是否针对问题，没有离题
#     问题：用户问"什么是RAG"，回答了一堆"Transformer原理" → 扣分
#
#   Context Precision（上下文精确度）
#     概念：检索结果中有用的比例
#     问题：检索出10个文档，只有2个和问题相关 → 扣分
#
#   Context Recall（上下文召回率）
#     概念：回答问题需要的信息是否都检索到了
#     问题：问题问AVC，你只检索了AVC的优点没检索缺点 → 扣分
#
# 四个指标通常一起使用，不能只看单个指标：
#   - 如果 Faithfulness 低但 Context Precision 高 →
#     检索质量好，但模型没有忠实使用检索结果（模型问题）
#   - 如果 Context Recall 低但 Faithfulness 高 →
#     检索不完整，但模型忠实使用了有限信息（检索问题）
#   - 如果都低 → 检索和模型都有问题
#
# 对比评估（A/B Testing）的最佳实践：
#   1. 准备固定测试数据集（20-50 个用例）
#   2. 两个配置分别跑一遍，记录结果
#   3. 用同样的评估器打分
#   4. 对比评分分布 + 胜率统计
#   5. 分析胜出的原因（不是只看分数）
#
# 教学建议顺序：
#   1. 先理解"为什么需要评估"——不能靠感觉判断质量
#   2. 理解在环评估（用户反馈）是最可靠的信号
#   3. 再理解 RAGAS 自动化评估（不需要人工）
#   4. 掌握自定义评估器（业务场景定制）
#   5. 最后理解对比评估（Prompt/模型 选型决策支持）
