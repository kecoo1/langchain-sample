"""LangSmith 进阶示例 —— 数据集评估、自定义评估器、Prompt 版本对比。

LangSmith 不仅是追踪工具，更是 LLM 应用的"测试框架"。
你可以用它：
1. 创建评估数据集
2. 定义自定义评估器（LLM-as-judge / 规则评估 / 人工评估）
3. 运行批量评估
4. 对比不同 Prompt / 模型的表现

运行前需要设置:
  export LANGCHAIN_API_KEY="your-api-key"
  export LANGCHAIN_TRACING_V2=true
"""

import os
os.environ.setdefault("LANGCHAIN_API_KEY", "YOUR_LANGCHAIN_API_KEY_HERE")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "langchain-sample-eval")

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, EvaluationResult, RunEvaluator, EvaluationResult
from langsmith.evaluation import evaluate, evaluate_run, evaluator
from langchain_core.messages import HumanMessage, AIMessage
import time

# ==============================================================================
# 示例 1：创建评估数据集
# ==============================================================================
print("=== 示例 1: 创建评估数据集 ===")

client = Client()

# 定义评估数据集
dataset_name = "code-review-dataset"

# 删除旧数据集（如果需要重新创建）
try:
    existing = client.read_dataset(dataset_name=dataset_name)
    client.delete_dataset(dataset_id=existing.id)
    print(f"   已删除旧数据集: {dataset_name}")
except:
    pass

# 创建数据集
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Code review questions for evaluating LLM performance",
)

# 添加样本
samples = [
    {
        "input": "Review this code: def add(a, b): return a + b",
        "expected": "This is a simple addition function. It is correct but lacks type hints and docstrings.",
    },
    {
        "input": "Review this code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "expected": "This recursive factorial function is correct for n >= 0. Consider adding input validation for negative numbers.",
    },
    {
        "input": "Review this code: def greet(name): print('Hello, ' + name)",
        "expected": "Basic greeting function. Should use f-string for modern Python. No return statement.",
    },
    {
        "input": "Review this code: def is_even(n): return n % 2 == 0",
        "expected": "Correct and concise even-check function. Could add type hints for clarity.",
    },
    {
        "input": "Review this code: def reverse_string(s): return s[::-1]",
        "expected": "Pythonic string reversal using slice notation. Efficient and readable.",
    },
]

for i, sample in enumerate(samples):
    client.create_example(
        inputs={"input": sample["input"]},
        outputs={"output": sample["expected"]},
        dataset_id=dataset.id,
    )

print(f"   创建了 {len(samples)} 个样本到数据集: {dataset_name}\n")


# ==============================================================================
# 示例 2：定义自定义评估器
# ==============================================================================
print("=== 示例 2: 自定义评估器 ===")


# 评估器 1：基于规则的评估（简单关键词匹配）
class KeywordMatchEvaluator(RunEvaluator):
    """基于关键词匹配的简单评估器。"""

    def evaluate_run(self, run: object, example: object | None = None) -> EvaluationResult:
        if not isinstance(run, dict):
            return EvaluationResult(key="keyword_match", score=0)

        # 提取输入和输出
        inputs = run.get("inputs", {})
        output = ""
        if "outputs" in run:
            output = run["outputs"].get("output", "")
        elif run.get("events"):
            for event in run["events"]:
                if event.get("data", {}).get("output"):
                    output = str(event["data"]["output"])
                    break

        expected = ""
        if example and example.outputs:
            expected = example.outputs.get("output", "")

        # 简单关键词匹配
        if expected and output:
            expected_words = set(expected.lower().split())
            output_words = set(output.lower().split())
            overlap = expected_words & output_words
            precision = len(overlap) / len(output_words) if output_words else 0
            recall = len(overlap) / len(expected_words) if expected_words else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return EvaluationResult(
                key="keyword_f1",
                score=min(f1, 1.0),
            )

        return EvaluationResult(key="keyword_f1", score=0)


# 评估器 2：LLM-as-Judge（用 LLM 评估回答质量）
@evaluator
def code_quality(run, example):
    """用 LLM 评估代码审查回答的质量。"""
    # 获取原始输入和输出
    inputs = run.inputs if hasattr(run, "inputs") else {}
    run_input = inputs.get("input", "") if isinstance(inputs, dict) else ""

    outputs = run.outputs if hasattr(run, "outputs") else {}
    run_output = outputs.get("output", "") if isinstance(outputs, dict) else ""

    expected = example.outputs.get("output", "") if example and example.outputs else ""

    # 用 LLM 评判
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a code review quality evaluator. Rate the following code review response.

Criteria (score 0-1):
- Does it identify key aspects of the code?
- Is the feedback constructive and specific?
- Does it mention strengths AND weaknesses?

Return ONLY a JSON object: {"score": 0.0-1.0, "reason": "brief explanation"}"""),
        ("human", """Original Code Review Request: {input}

Expected (Reference) Answer: {expected}

Model's Answer: {actual}

Rate the model's answer:"""),
    ])

    try:
        result = judge_prompt | judge_llm | StrOutputParser()
        eval_result = result.invoke({
            "input": run_input,
            "expected": expected,
            "actual": run_output,
        })

        # 简单解析分数
        try:
            import json
            parsed = json.loads(eval_result)
            score = parsed.get("score", 0.0)
            reason = parsed.get("reason", "Could not parse reason")
        except:
            score = 0.5
            reason = eval_result[:100]

        return EvaluationResult(
            key="code_quality",
            score=score,
            comment=reason,
        )
    except Exception as e:
        return EvaluationResult(
            key="code_quality",
            score=0.0,
            comment=f"Evaluation failed: {str(e)}",
        )


print("   已定义两个评估器:")
print("   1. KeywordMatchEvaluator - 关键词匹配 F1")
print("   2. code_quality - LLM-as-Judge\n")


# ==============================================================================
# 示例 3：运行批量评估
# ==============================================================================
print("=== 示例 3: 运行批量评估 ===")


# 定义待评估的链
def code_review_chain(input_text: str) -> str:
    """代码审查链。"""
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert code reviewer. Review the provided code and give constructive feedback."),
        ("human", "{input}"),
    ])
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = review_prompt | model | StrOutputParser()
    return chain.invoke({"input": input_text})


# 运行评估
# 注意：完整的评估需要在 LangSmith 平台上运行
# 这里展示评估的配置和结构

evaluation_name = "code-review-chain-eval"

# 在 LangSmith 中运行的方式（取消注释以实际执行）:
#
# result = evaluate(
#     data=dataset_name,
#     fn=code_review_chain,
#     evaluators=[
#         KeywordMatchEvaluator(),
#         code_quality,
#     ],
#     experiment_prefix=evaluation_name,
#     metadata={
#         "version": "v1.0",
#         "model": "gpt-4o-mini",
#         "prompt": "default",
#     },
# )
#
# print(f"   评估完成: {evaluation_name}")
# print(f"   查看结果: https://smith.langchain.com")

print("   评估配置已就绪:")
print(f"   数据集: {dataset_name}")
print(f"   评估函数: code_review_chain")
print(f"   评估器: KeywordMatchEvaluator, code_quality")
print("   取消注释 evaluate() 调用以实际运行\n")


# ==============================================================================
# 示例 4：Prompt 版本 A/B 测试
# ==============================================================================
print("=== 示例 4: Prompt 版本 A/B 测试 ===")

# 定义两个版本的 Prompt
prompt_v1 = ChatPromptTemplate.from_messages([
    ("system", "You are a code reviewer."),
    ("human", "Review: {input}"),
])

prompt_v2 = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer with 10+ years of experience.
When reviewing code, consider:
1. Correctness
2. Readability and maintainability
3. Potential edge cases
4. Performance implications
Give specific, actionable feedback."""),
    ("human", "Please review this code:\n{input}"),
])

# 构建两个版本的链
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain_v1 = prompt_v1 | model | StrOutputParser()
chain_v2 = prompt_v2 | model | StrOutputParser()

# 在测试集上比较
test_inputs = [
    "def add(a, b): return a + b",
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    "def greet(name): print('Hello, ' + name)",
]

print("   A/B 测试结果:")
print(f"   {'代码':40} {'Prompt V1':30} {'Prompt V2':30}")
print("   " + "-" * 100)

for code in test_inputs:
    result_v1 = chain_v1.invoke({"input": code})
    result_v2 = chain_v2.invoke({"input": code})

    # 简单长度比较（生产环境用评估器）
    v1_len = len(result_v1)
    v2_len = len(result_v2)

    print(f"   {code[:40]:40} {str(v1_len)[:10]:10} chars | {str(v2_len)[:10]:10} chars")

print("\n   注意：正式比较应在 LangSmith 上运行完整评估，使用 code_quality 评估器打分\n")


# ==============================================================================
# 示例 5：RAG 评估
# ==============================================================================
print("=== 示例 5: RAG 评估 ===")

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 创建 RAG 链
rag_docs_text = """
LangChain is a framework for developing applications powered by large language models.
It provides a standard interface for chains, prompts, and agents.

LangGraph is a framework for building stateful, multi-actor applications.
It adds orchestration capabilities for complex agent workflows.

LangSmith is a platform for building production-grade LLM applications.
It provides tracing, debugging, and evaluation capabilities.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
docs = text_splitter.create_documents([rag_docs_text])

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

# RAG 链
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the context. Cite sources as [Doc N]."),
    ("human", "Context: {context}\n\nQuestion: {question}"),
])

rag_chain = (
    {
        "context": ensemble_retriever,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | model
    | StrOutputParser()
)

# RAG 评估数据集
rag_dataset_name = "rag-eval-dataset"

try:
    existing = client.read_dataset(dataset_name=rag_dataset_name)
    client.delete_dataset(dataset_id=existing.id)
except:
    pass

rag_dataset = client.create_dataset(
    dataset_name=rag_dataset_name,
    description="RAG evaluation dataset",
)

rag_samples = [
    {
        "input": "What is LangChain?",
        "expected": "LangChain is a framework for developing applications powered by large language models.",
    },
    {
        "input": "What does LangGraph do?",
        "expected": "LangGraph is a framework for building stateful, multi-actor applications with orchestration capabilities.",
    },
    {
        "input": "What is LangSmith used for?",
        "expected": "LangSmith is a platform for building production-grade LLM applications with tracing and debugging.",
    },
]

for sample in rag_samples:
    client.create_example(
        inputs={"input": sample["input"]},
        outputs={"output": sample["expected"]},
        dataset_id=rag_dataset.id,
    )

print(f"   创建了 RAG 评估数据集: {rag_dataset_name}")
print(f"   样本数: {len(rag_samples)}\n")

# ==============================================================================
# 示例 6：评估指标解读
# ==============================================================================
print("=== 示例 6: 评估指标解读 ===")

print("""
LangSmith 评估报告中的关键指标:

1. 准确率 (Accuracy):
   - 回答与预期答案的匹配程度
   - 关键词匹配: 简单但粗糙
   - LLM-as-Judge: 更准确但更慢

2. 召回率 (Recall):
   - 检索步骤：是否找到了包含正确答案的文档？
   - 低召回率 -> 改进检索策略（混合检索、HyDE）

3. 精确率 (Precision):
   - 检索步骤：返回的文档中有多少是相关的？
   - 低精确率 -> 改进重排序

4. F1 Score:
   - 精确率和召回率的调和平均
   - 综合衡量检索质量

5. 语义相似度 (Semantic Similarity):
   - 用嵌入模型计算回答与预期的余弦相似度
   - 比关键词匹配更语义化

6. 延迟 (Latency):
   - P50/P90/P99 延迟百分位
   - 帮助识别性能瓶颈

7. Token 消耗:
   - 输入/输出 token 数量
   - 成本控制的关键指标

评估最佳实践:
- 始终使用足够大的测试集（至少 30-50 个样本）
- 包含边缘情况和对抗性样本
- 使用多种评估器（规则 + LLM-as-Judge）
- 定期重新评估（每次改 Prompt/模型/检索策略后）
- 建立基线，跟踪改进趋势
""")

# Clean up
vectorstore.delete_collection()

# ==============================================================================
# 教学备注：LangSmith 评估 —— 让 LLM 应用从"玄学"变成"科学"
# ==============================================================================
# 核心问题：你怎么知道你的 LLM 应用"变好了"还是"变坏了"？
#   直觉： "这个 Prompt 感觉更好了"
#   数据： "这个 Prompt 在 100 个测试用例上 F1 提升了 15%"
#   答案：评估是唯一可靠的判断方式
#
# 评估的三个层次:
#   Level 1: 规则评估
#     关键词匹配、BLEU、ROUGE、BERTScore
#     优点：快、便宜、可复现
#     缺点：不理解语义，"猫"和"猫咪"可能被判为不匹配
#
#   Level 2: LLM-as-Judge
#     用另一个 LLM 评判生成的质量
#     优点：理解语义，可以做细粒度评分
#     缺点：慢、贵、judge 本身可能有偏见
#     最佳实践：用强模型做 judge（gpt-4o），温度设为 0
#
#   Level 3: 人工评估
#     真人标注员打分
#     优点：最准确
#     缺点：最贵、最慢、一致性差
#     最佳实践：用 LLM-as-Judge 做日常评估，定期抽样人工校准
#
# 评估数据集的构建:
#   1. 覆盖核心场景：用户最常问的问题
#   2. 覆盖边缘情况：模糊问题、多意图问题、错误输入
#   3. 包含"不知道"的情况：测试模型是否会幻觉
#   4. 持续扩充：每次发现 bad case 就加到数据集
#   5. 版本化管理：数据集本身也需要版本控制
#
# A/B 测试的正确姿势:
#   1. 固定测试集
#   2. 固定评估器
#   3. 运行足够多的样本（统计显著性）
#   4. 记录所有配置（模型、温度、Prompt 版本）
#   5. 在 LangSmith 上对比实验结果
#
# 教学建议顺序:
#   1. 先理解"为什么需要评估"——从直觉到数据驱动
#   2. 再理解三种评估方式及其权衡
#   3. 再动手创建数据集和评估器
#   4. 最后理解 A/B 测试和指标解读
