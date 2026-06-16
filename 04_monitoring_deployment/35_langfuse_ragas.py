"""LangFuse + RAGAS 组合使用 —— 追踪 + 评估一体化方案。

LangFuse 提供全链路追踪和可视化（类似 LangSmith 的开源替代），
RAGAS 提供自动化的 RAG 质量评估指标。

两者组合：LangFuse 记录每一次 RAG 请求的完整链路，
          RAGAS 对这些请求进行自动化质量评估。

适用场景：
  1. 生产环境中持续监控 RAG 质量
  2. 发现检索/生成质量问题并定位根因
  3. Prompt 优化前后对比 RAGAS 指标
  4. 建立 RAG 质量的基线和趋势

运行前设置:
  export LANGFUSE_PUBLIC_KEY="your-public-key"
  export LANGFUSE_SECRET_KEY="your-secret-key"
  export LANGFUSE_HOST="cloud.langfuse.com"  # 或自托管地址
  export OPENAI_API_KEY="sk-..."  # RAGAS 评估需要
"""

import os
import json
import time
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Optional

# ==============================================================================
# 示例 1：搭建 RAG Pipeline（带 LangFuse 追踪）
# ==============================================================================
print("=== 示例 1: RAG Pipeline + LangFuse 追踪 ===")

# 模拟文档库
DOCS_TEXT = """
LangChain is a framework for developing applications powered by large language models.
It provides a standard interface for chains, prompts, and agents.

LangChain supports many model providers including OpenAI, Anthropic, Ollama, and Google Gemini.

Key components of LangChain:
- Models: Chat models and LLMs from various providers
- Prompts: Template management and composition
- Chains: Composable building blocks for complex workflows
- Agents: Autonomous systems that use tools to accomplish tasks
- Tools: Integrations with external systems and APIs
- Memory: State persistence across interactions
- Retrieval: Document loading, splitting, and vector storage

LangGraph is a low-level framework built on top of LangChain for building stateful,
multi-actor applications. It adds orchestration capabilities for complex agent workflows.

LangSmith is a developer platform for building production-grade LLM applications.
It provides tracing, debugging, and evaluation capabilities.

LangServe enables you to deploy LangChain chains as REST APIs with one line of code.

RAG (Retrieval Augmented Generation) enhances LLM responses with external knowledge.
Instead of relying solely on training data, RAG retrieves relevant documents first.
"""

# 文档分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
documents = text_splitter.create_documents([DOCS_TEXT])

# 向量存储
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# BM25 关键词检索
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 自定义混合检索器（新版 LangChain 中 EnsembleRetriever 已被移除）
class HybridRetriever(BaseRetriever):
    """向量 + BM25 混合检索，使用倒数排名融合 (RRF) 合并结果。"""

    retrievers: List[BaseRetriever] = []
    weights: List[float] = []
    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 各检索器分别取 Top-K
        results = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs, 1):
                # RRF 融合: score = weight / (rank + 60)
                key = id(doc)
                results[key] = results.get(key, 0) + weight / (rank + 60)
                doc._id = key  # type: ignore[attr-defined]

        # 按融合分数排序
        sorted_docs = sorted(results.items(), key=lambda x: x[1], reverse=True)[:self.k]
        return [doc for _, doc in sorted_docs]


ensemble_retriever = HybridRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
    k=5,
)

# RAG 链
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question based ONLY on the
provided context. If you cannot answer from the context, say "I don't have enough information."

Context:
{context}

Question: {question}"""),
    ("human", "{question}"),
])

llm = ChatOllama(model="llama3.2:1b", temperature=0)

rag_chain = (
    {
        "context": ensemble_retriever,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# 带 LangFuse 追踪的 RAG 链
try:
    from langfuse.callback import LangfuseTraceLogger

    trace_logger = LangfuseTraceLogger()
    rag_chain_traced = rag_chain.with_types(
        input_type=str, output_type=str
    ).with_config(callbacks=[trace_logger])

    print("   ✓ LangFuse 追踪已启用")
except ImportError:
    print("   ⚠ LangFuse 未安装，使用普通链")
    rag_chain_traced = rag_chain
    print("   安装: pip install langfuse\n")

# 执行查询并追踪
questions = [
    "What is RAG?",
    "What are the key components of LangChain?",
    "How does LangGraph relate to LangChain?",
]

traces = []
for q in questions:
    start = time.time()
    result = rag_chain_traced.invoke(q)
    duration = time.time() - start
    traces.append({
        "question": q,
        "answer": result,
        "latency": round(duration, 3),
        "timestamp": datetime.now().isoformat(),
    })
    print(f"   Q: {q}")
    print(f"   A: {result[:80]}...")
    print(f"   耗时: {duration:.3f}s\n")


# ==============================================================================
# 示例 2：LangFuse 原生集成 —— Trace + Span + Generation
# ==============================================================================
print("=== 示例 2: LangFuse 原生 API 集成 ===")

try:
    from langfuse import Langfuse

    langfuse = Langfuse()

    # 创建 Trace（对应一次完整的 RAG 请求）
    trace = langfuse.trace(
        name="rag-pipeline",
        input={"query": "What is LangChain?"},
        metadata={"model": "llama3.2:1b", "retrieval": "ensemble"},
    )

    # 创建 Span（对应检索阶段）
    retrieval_span = trace.span(
        name="document-retrieval",
        input={"k": 5},
        metadata={"strategy": "vector+bm25"},
    )
    time.sleep(0.1)  # 模拟检索耗时
    retrieval_span.update(
        output={"num_docs": 5, "latency_ms": 100},
        status_message="Retrieved 5 documents",
    )

    # 创建 Generation（对应 LLM 生成阶段）
    generation = trace.generation(
        name="llm-generation",
        model="llama3.2:1b",
        input=[{"role": "user", "content": "What is LangChain?"}],
        output="LangChain is a framework...",
        metadata={"token_count": 150},
    )

    # 更新 Trace 状态
    trace.update(
        output={"answer": "LangChain is a framework..."},
        status_message="Completed successfully",
    )

    # 刷新同步
    langfuse.flush()

    print("   ✓ Trace 创建成功: rag-pipeline")
    print("   ✓ Span 记录: document-retrieval")
    print("   ✓ Generation 记录: llm-generation")
    print("   访问 LangFuse Dashboard 查看: http://localhost:3000\n")

except ImportError:
    print("   ⚠ LangFuse 未安装，展示 API 结构\n")
    print("""   LangFuse 原生 API 结构:
   
   langfuse = Langfuse()
   
   # 1. Trace - 对应一次完整的用户请求
   trace = langfuse.trace(
       name="rag-pipeline",
       input={"query": "..."},
       metadata={"model": "..."},
   )
   
   # 2. Span - 对应请求中的某个阶段（检索/生成）
   trace.span(
       name="document-retrieval",
       input={"k": 5},
   )
   
   # 3. Generation - 对应 LLM 调用
   trace.generation(
       name="llm-generation",
       model="llama3.2:1b",
       input=[...],
       output="...",
   )
   
   # 4. Event - 细粒度事件（可选）
   trace.event(name="cache-check", metadata={"hit": False})
   
   # 5. 会话（Session）- 关联多个 Trace
   trace.update(session_id="user-session-123")
   
   # 6. 资源（Resource）- 关联外部实体
   trace.update(user_id="user-456")
""")

# ==============================================================================
# 示例 3：LangFuse 与 LCEL 集成 —— 自动追踪
# ==============================================================================
print("=== 示例 3: LangFuse + LCEL 自动追踪 ===")

try:
    from langfuse.langchain import CallbackHandler

    # 创建 LangFuse 回调处理器
    langfuse_callback = CallbackHandler()

    # 通过 .with_config() 注入追踪
    chain_with_langfuse = (
        {
            "context": ensemble_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 执行带追踪的查询
    result = chain_with_langfuse.invoke(
        "What is LangServe?",
        config={"callbacks": [langfuse_callback]},
    )

    print(f"   问题: What is LangServe?")
    print(f"   回答: {result[:80]}...")
    print(f"   ✓ 追踪已通过 CallbackHandler 自动记录\n")

except ImportError:
    print("   ⚠ LangFuse 未安装\n")
except Exception as e:
    print(f"   注意: {e}\n")


# ==============================================================================
# 示例 4：RAGAS 评估 —— 自动化 RAG 质量指标
# ==============================================================================
print("=== 示例 4: RAGAS 自动化评估 ===")

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

    # 用 OpenAI 作为评估器（RAGAS 推荐）
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(eval_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # 准备测试数据
    test_cases = [
        {
            "question": "What is RAG?",
            "contexts": [
                "RAG (Retrieval Augmented Generation) enhances LLM responses with external knowledge.",
                "Instead of relying solely on training data, RAG retrieves relevant documents first.",
            ],
            "answer": "RAG stands for Retrieval Augmented Generation. It enhances LLM responses by retrieving relevant external documents before generating an answer, rather than relying solely on the model's training data.",
            "ground_truth": "RAG (Retrieval Augmented Generation) is a technique that combines retrieval with generation, allowing LLMs to access external knowledge before producing answers.",
        },
        {
            "question": "What are the key components of LangChain?",
            "contexts": [
                "Key components of LangChain: Models, Prompts, Chains, Agents, Tools, Memory, Retrieval.",
                "LangChain provides a standard interface for chains, prompts, and agents.",
            ],
            "answer": "The key components of LangChain are: Models (Chat models and LLMs), Prompts (template management), Chains (composable workflows), Agents (autonomous tool-using systems), Tools (external integrations), Memory (state persistence), and Retrieval (document loading and vector storage).",
            "ground_truth": "LangChain's key components include Models, Prompts, Chains, Agents, Tools, Memory, and Retrieval.",
        },
        {
            "question": "How does LangGraph relate to LangChain?",
            "contexts": [
                "LangGraph is a low-level framework built on top of LangChain.",
                "It adds orchestration capabilities for complex agent workflows.",
            ],
            "answer": "LangGraph is built on top of LangChain and adds state management and orchestration capabilities for building multi-actor, stateful applications with complex agent workflows.",
            "ground_truth": "LangGraph is a low-level framework built on top of LangChain for building stateful, multi-actor applications with orchestration capabilities.",
        },
    ]

    print("4. RAGAS 评估指标说明:")
    print("   - Faithfulness: 回答是否忠实于检索到的上下文")
    print("   - Answer Relevancy: 回答与问题的相关程度")
    print("   - Context Precision: 检索结果的精确度")
    print("   - Context Recall: 检索结果的召回率")
    print()

    all_scores = []
    for i, tc in enumerate(test_cases):
        sample = SingleTurnSample(
            user_input=tc["question"],
            response=tc["answer"],
            retrieved_contexts=tc["contexts"],
            reference=tc["ground_truth"],
        )

        print(f"   测试用例 {i+1}: {tc['question']}")

        metrics = {}
        for metric_fn, metric_name in [
            (faithfulness, "Faithfulness"),
            (answer_relevancy, "Answer Relevancy"),
            (context_precision, "Context Precision"),
            (context_recall, "Context Recall"),
        ]:
            try:
                score = metric_fn.score(sample)
                metrics[metric_name] = round(score, 3)
                print(f"     {metric_name}: {score:.3f}")
            except Exception as e:
                metrics[metric_name] = None
                print(f"     {metric_name}: 错误 ({str(e)[:40]})")

        all_scores.append(metrics)
        print()

    # 汇总
    if all_scores:
        print("   汇总平均:")
        avg_metrics = {}
        for name in ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]:
            vals = [s[name] for s in all_scores if s.get(name) is not None]
            if vals:
                avg = sum(vals) / len(vals)
                avg_metrics[name] = round(avg, 3)
                print(f"     {name}: {avg:.3f}")
        print()

except ImportError:
    print("   跳过: 需要 pip install ragas\n")
except Exception as e:
    print(f"   RAGAS 评估出错（可能需要 OpenAI API Key）: {e}\n")


# ==============================================================================
# 示例 5：LangFuse + RAGAS 组合 —— 追踪与评估闭环
# ==============================================================================
print("=== 示例 5: LangFuse + RAGAS 组合工作流 ===")

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context

    print("5. 组合架构设计:")
    print("""
   ┌─────────────────────────────────────────────────────┐
   │                  用户请求                              │
   └──────────────────┬──────────────────────────────────┘
                      │
                      ▼
   ┌─────────────────────────────────────────────────────┐
   │              LangFuse Trace                          │
   │  ┌─────────────┐    ┌──────────────┐                │
   │  │  Span: 检索  │───▶│  Span: 生成   │                │
   │  │  (k=5,       │    │  (model:     │                │
   │  │   latency)   │    │   llama3.2)  │                │
   │  └─────────────┘    └──────────────┘                │
   └──────────────────┬──────────────────────────────────┘
                      │
                      ▼
   ┌─────────────────────────────────────────────────────┐
   │              RAGAS 评估                               │
   │  ┌─────────────┐    ┌──────────────┐                │
   │  │ Faithfulness│    │ Relevancy    │                │
   │  │ Precision   │    │ Recall       │                │
   │  └─────────────┘    └──────────────┘                │
   └──────────────────┬──────────────────────────────────┘
                      │
                      ▼
   ┌─────────────────────────────────────────────────────┐
   │              LangFuse Evaluation                     │
   │  ┌─────────────┐    ┌──────────────┐                │
   │  │ Score: 0.85 │    │ Comment:     │                │
   │  │ (stored as  │    │ "Good but    │                │
   │  │  metric)    │    │  context     │                │
   │  └─────────────┘    └──────────────┘                │
   └─────────────────────────────────────────────────────┘
    """)

    print("5a. LangFuse Evaluation API 使用:")
    print("""
   # 在 LangFuse 中记录 RAGAS 评估结果
   langfuse_context.score(
       trace_id="<trace_id>",        # 关联到具体 Trace
       name="ragas_faithfulness",    # 指标名称
       value=0.85,                   # 分数
       comment="回答忠实于上下文",     # 可选说明
   )
   
   # 批量评估
   for trace in traces:
       langfuse_context.score(
           trace_id=trace["trace_id"],
           name="ragas_answer_relevancy",
           value=0.92,
       )
    """)

    print("\n5b. 完整组合流程示例:")
    print("""
   from langfuse import Langfuse
   from ragas.metrics import faithfulness
   from ragas.dataset_schema import SingleTurnSample
   
   langfuse = Langfuse()
   
   # 1. 用户发起请求
   trace = langfuse.trace(name="rag-request")
   
   # 2. 执行 RAG 链（自动记录 Span）
   answer = rag_chain.invoke("What is LangChain?")
   
   # 3. 用 RAGAS 评估
   sample = SingleTurnSample(
       user_input="What is LangChain?",
       response=answer,
       retrieved_contexts=["LangChain is..."],
       reference="LangChain is a framework...",
   )
   score = faithfulness.score(sample)
   
   # 4. 将评估结果回传到 LangFuse
   langfuse.score(
       trace_id=trace.id,
       name="faithfulness",
       value=score,
   )
   
   # 5. 在 LangFuse Dashboard 查看所有 Trace 的 RAGAS 分数趋势
    """)

    print("\n5c. 生产环境最佳实践:")
    print("""
   1. 采样评估：不是每次请求都做 RAGAS 评估（成本高），
              而是每天随机采样 50-100 条做评估
   
   2. 告警阈值：当 Faithfulness 低于 0.6 时触发告警
   
   3. 趋势监控：在 LangFuse 中查看 RAGAS 指标的周/月趋势
   
   4. 分段分析：按用户、时间段、问题类型分组查看评估结果
   
   5. 回归检测：每次更新 Prompt/模型/检索策略后，
              跑一次完整评估集，对比 RAGAS 指标变化
    """)

except ImportError:
    print("   ⚠ LangFuse 未安装，展示架构设计\n")


# ==============================================================================
# 示例 6：端到端 —— 模拟生产环境的 LangFuse + RAGAS 集成
# ==============================================================================
print("=== 示例 6: 端到端模拟集成 ===")


class RAGEvaluator:
    """模拟生产环境中的 RAG 质量评估器。"""

    def __init__(self):
        self.evaluation_history = []

    def evaluate_trace(self, trace: dict, ragas_scores: dict) -> dict:
        """将 RAGAS 评估结果与 LangFuse Trace 关联。"""
        evaluation = {
            "trace_id": trace.get("trace_id", "unknown"),
            "question": trace["question"],
            "answer": trace["answer"],
            "ragas_scores": ragas_scores,
            "evaluated_at": datetime.now().isoformat(),
        }
        self.evaluation_history.append(evaluation)
        return evaluation

    def get_dashboard_data(self) -> dict:
        """生成 LangFuse Dashboard 可用的数据。"""
        if not self.evaluation_history:
            return {"message": "暂无评估数据"}

        avg_scores = {}
        for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            vals = [
                e["ragas_scores"].get(key, 0)
                for e in self.evaluation_history
                if key in e["ragas_scores"]
            ]
            if vals:
                avg_scores[key] = round(sum(vals) / len(vals), 3)

        return {
            "total_evaluations": len(self.evaluation_history),
            "average_scores": avg_scores,
            "history": self.evaluation_history,
        }


# 模拟完整流程
print("6. 模拟生产流程:")

# Step 1: 用户查询（LangFuse 记录 Trace）
sample_trace = {
    "trace_id": "trace-001",
    "question": "What is LangChain?",
    "answer": "LangChain is a framework for developing applications powered by LLMs.",
    "latency": 0.5,
    "model": "llama3.2:1b",
}

print(f"   Step 1: 用户查询 -> Trace ID: {sample_trace['trace_id']}")
print(f"           Question: {sample_trace['question']}")

# Step 2: RAGAS 评估（离线批量）
mock_ragas_scores = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.90,
    "context_precision": 0.78,
    "context_recall": 0.82,
}

print(f"\n   Step 2: RAGAS 评估结果:")
for k, v in mock_ragas_scores.items():
    print(f"           {k}: {v}")

# Step 3: 关联评估结果到 Trace
evaluator = RAGEvaluator()
evaluation = evaluator.evaluate_trace(sample_trace, mock_ragas_scores)

print(f"\n   Step 3: 评估结果已关联到 Trace")
print(f"           Trace ID: {evaluation['trace_id']}")

# Step 4: 生成 Dashboard 数据
dashboard = evaluator.get_dashboard_data()

print(f"\n   Step 4: Dashboard 数据:")
print(f"           总评估数: {dashboard['total_evaluations']}")
print(f"           平均 Faithfulness: {dashboard['average_scores'].get('faithfulness', 'N/A')}")
print(f"           平均 Answer Relevancy: {dashboard['average_scores'].get('answer_relevancy', 'N/A')}")

# Clean up
vectorstore.delete_collection()

print()

# ==============================================================================
# 教学备注：LangFuse + RAGAS 组合 —— 让 RAG 可观测、可评估、可优化
# ==============================================================================
# 核心问题：RAG 系统上线后，你怎么知道它"好不好"？
#   单一追踪（LangFuse）：知道"发生了什么"，但不知道"做得怎么样"
#   单一评估（RAGAS）：知道"做得怎么样"，但不知道"什么时候变差的"
#   组合使用：既知道发生了什么，也知道质量如何，还能追踪质量变化趋势
#
# LangFuse 的角色（追踪层）:
#   1. Trace: 一次完整的用户请求（包含多个 Span）
#   2. Span: 请求中的子操作（检索、生成、后处理）
#   3. Generation: LLM 调用的详情（模型、token、延迟）
#   4. Session: 关联同一用户的多次请求
#   5. Events: 细粒度事件（缓存命中、重试等）
#   6. Metrics: 自定义指标（可以存 RAGAS 分数！）
#
# RAGAS 的角色（评估层）:
#   1. Faithfulness: 回答是否忠实于检索到的上下文
#      - 低分数 = 模型在"幻觉"，编造上下文里没有的内容
#      - 修复方向：加强 Prompt 约束，或用更强的模型
#   2. Answer Relevancy: 回答与问题的相关程度
#      - 低分数 = 回答跑题了
#      - 修复方向：优化 Prompt，减少噪声上下文
#   3. Context Precision: 检索结果中有多少是真正相关的
#      - 低分数 = 检索引入了太多噪声
#      - 修复方向：改进检索策略（混合检索、重排序）
#   4. Context Recall: 回答问题需要的信息是否都检索到了
#      - 低分数 = 检索不完整
#      - 修复方向：增大 k、用 HyDE、多查询检索
#
# 组合使用模式:
#   模式 1：实时评估（线上）
#     用户请求 -> RAG 链 -> LangFuse Trace -> 立即 RAGAS 评估 -> 分数写回 LangFuse
#     优点：即时发现质量问题
#     缺点：增加延迟，成本高
#     适用：关键业务场景
#
#   模式 2：离线评估（批量）
#     用户请求 -> RAG 链 -> LangFuse Trace -> 定期批量拉取 -> RAGAS 评估 -> 分数写回
#     优点：不影响线上延迟，成本低
#     缺点：发现问题有延迟
#     适用：日常质量监控
#
#   模式 3：采样评估（推荐）
#     每天随机采样 50-100 条 Trace -> RAGAS 评估 -> 分数写回
#     优点：平衡成本和及时性
#     缺点：覆盖率有限
#     适用：大多数生产场景
#
# LangFuse Dashboard 中的 RAGAS 可视化:
#   - 按时间查看 Faithfulness 趋势图
#   - 对比不同 Prompt 版本的 RAGAS 分数
#   - 按用户/Session 分组查看评估结果
#   - 设置告警：Faithfulness < 0.6 时通知
#   - 导出评估数据用于报告
#
# RAGAS 评估的数据准备:
#   需要: question + answer + contexts + ground_truth
#   - question: 用户原始问题（LangFuse Trace 中有）
#   - answer: RAG 链的输出（LangFuse Trace 中有）
#   - contexts: 检索到的文档（Span 中有）
#   - ground_truth: 标准答案（需要预先准备评估集）
#
# 评估数据集的准备:
#   1. 从生产数据中抽取典型问题
#   2. 人工编写标准答案（ground truth）
#   3. 至少 30-50 个测试用例
#   4. 覆盖核心场景 + 边缘情况
#   5. 定期更新评估数据集
#
# 生产部署建议:
#   1. 先用 LangFuse 接入追踪（低成本，立即可见）
#   2. 准备 RAGAS 评估数据集（30-50 个用例）
#   3. 每天运行一次采样评估，分数写回 LangFuse
#   4. 在 LangFuse Dashboard 中查看 RAGAS 指标趋势
#   5. 设定告警阈值，质量下降时自动通知
#   6. 每次优化 Prompt/检索策略后，跑完整评估集对比
#
# 与 LangSmith 的对比:
#   LangSmith: 一站式平台（追踪 + 评估 + 数据集管理），商业产品
#   LangFuse: 开源可自托管，追踪功能强大，需配合 RAGAS 做评估
#   选择建议:
#     - 预算充足 + 不想运维 -> LangSmith
#     - 需要自托管 + 控制成本 -> LangFuse + RAGAS
#     - 两者都集成 LangChain，接入方式类似
#
# 教学建议顺序:
#   1. 先跑通示例 1（RAG + LangFuse 追踪），感受追踪的力量
#   2. 再跑通示例 4（RAGAS 评估），理解四个指标的含义
#   3. 最后理解示例 5（组合架构），掌握"追踪 + 评估"的闭环
#   4. 在生产环境中用示例 6 的模式，每天采样评估 + 趋势监控
