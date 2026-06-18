"""Deep Agents RAG —— 用 deepagents 框架构建自主 RAG Agent。

Deep Agents 是 LangChain 官方高层 Agent 框架，内置 task planning、
虚拟文件系统、子 Agent 委托、human-in-the-loop 等能力。

本文件展示如何用 Deep Agents 构建自主 RAG 系统：
  1. 创建带 RAG 工具的 Deep Agent
  2. Agent 自主规划多步检索策略 (write_todos)
  3. 多检索器路由（向量/关键词/混合）
  4. 结果评估与迭代重检索
  5. 多源信息融合与溯源引用
  6. 人机协同 + 虚拟文件系统

运行前准备：
  pip install deepagents langchain-ollama langchain-community chromadb rank_bm25
  # 确保 Ollama 正在运行: ollama serve
  # 已安装模型: ollama pull llama3.2:1b

⚠️  llama3.2:1b 模型较小，tool calling 能力有限，
   部分功能可能需要更强模型（GPT-4o / Claude / Qwen2.5:7b）。
   每个示例标注了替代写法。
"""

import json
import re
from typing import Any

# ==============================================================================
# 全局配置
# ==============================================================================

try:
    from deepagents import create_deep_agent
    _DEEPAGENTS_AVAILABLE = True
except ImportError:
    _DEEPAGENTS_AVAILABLE = False
    create_deep_agent = None

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

LLM_MODEL = "llama3.2:1b"
TOP_K = 3
MAX_ITERATIONS = 3


def _check_available():
    if not _DEEPAGENTS_AVAILABLE:
        print("⚠️  需要安装 deepagents: pip install deepagents")


# ==============================================================================
# 结构化 LangChain FAQ 知识库
# ==============================================================================
# 8 个分类覆盖 LangChain 核心概念、组件、最佳实践和常见问题
# 与 08_rag.py 的原始文本不同，这个知识库有分类元数据，
# Agent 可以根据问题类型选择不同的检索策略

KNOWLEDGE_BASE: dict[str, dict[str, str]] = {
    "Basics": {
        "What is LangChain": "LangChain is a framework for developing applications powered by large language models (LLMs). It provides a standard interface for chains, prompts, agents, tools, and memory, allowing developers to compose complex LLM workflows.",
        "Key Components": "LangChain has six core components: Models (chat models and LLMs), Prompts (template management), Chains (composable building blocks), Agents (autonomous tool-using systems), Tools (external integrations), and Memory (state persistence).",
        "Installation": "Install LangChain with: pip install langchain. For specific integrations: pip install langchain-openai, langchain-anthropic, langchain-ollama, langchain-community.",
    },
    "Components": {
        "Chat Models": "Chat models are the primary interface to LLMs. Usage: from langchain_ollama import ChatOllama; llm = ChatOllama(model='llama3.2:1b'). Support sync invoke, async ainvoke, streaming, and batch operations.",
        "Prompt Templates": "PromptTemplates help manage LLM prompts with variables. ChatPromptTemplate.from_messages([('system', sys_msg), ('user', '{input}')]) creates structured multi-turn prompts with automatic formatting.",
        "Chains (LCEL)": "LangChain Expression Language (LCEL) uses the | operator to compose runnables: chain = prompt | model | StrOutputParser(). Supports parallel execution, branching, fallbacks, and streaming out of the box.",
        "Tools and Toolkits": "Tools are functions that LLMs can call. Defined with @tool decorator: @tool\ndef search(query: str) -> str: ... . Use bind_tools to attach tools to models, or create_tool_calling_agent for autonomous tool use.",
        "Memory Systems": "LangChain supports ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory, VectorStoreMemory, and EntityMemory for maintaining conversation state across interactions.",
    },
    "Agents": {
        "create_tool_calling_agent": "The recommended way to create agents. Uses the model's native tool calling API. agent = create_tool_calling_agent(llm, tools, prompt). Execute with AgentExecutor(agent=agent, tools=tools).",
        "create_react_agent": "An alternative that uses text-based ReAct (Thought/Action/Observation) format. Compatible with models that don't support native tool calling. Works with Ollama and other local models.",
        "Agent Executor": "AgentExecutor wraps an agent and manages the execution loop: call agent, execute tool, feed back result, repeat until final answer. Supports max_iterations, early_stopping_method, return_intermediate_steps.",
    },
    "LangGraph": {
        "StateGraph": "StateGraph is LangGraph's core graph structure. Define state schema with TypedDict, add nodes (processing functions), add edges (control flow), use conditional edges for branching. Compile with StateGraph.build().",
        "ToolNode": "ToolNode is a built-in node that executes tool calls from an agent message. Add to graph: graph.add_node('tools', ToolNode(tools)). Connect agent node to ToolNode for autonomous tool execution.",
        "Human-in-the-loop": "Use interrupt() to pause graph execution for human approval. resume parameter continues execution. Checkpointer (MemorySaver) required for state persistence across interrupts.",
    },
    "Best Practices": {
        "RAG Pipeline Design": "RAG follows: load documents → split into chunks → embed → store in vector DB → retrieve relevant chunks → format context → generate answer. Chunk size 500-1000 tokens, overlap 10-20% for best results.",
        "Model Selection": "Choose models by: task complexity, latency requirements, cost constraints, and privacy needs. Small models (1-3B) for simple tasks, large (70B+) for complex reasoning. Local models for privacy, cloud for quality.",
        "Error Handling": "Always wrap LLM/tool calls in try-except. Provide fallback responses. Use .with_fallbacks() for LCEL chains. Set max_retries on model init. Log errors with callbacks for debugging.",
    },
    "Troubleshooting": {
        "Model not loading": "Check Ollama is running: ollama serve. Verify model installed: ollama list. Ensure correct model name. For OpenAI: check OPENAI_API_KEY env var is set.",
        "Tool calling failing": "Small models (llama3.2:1b) have limited tool calling. Try: 1) Use a larger model (qwen2.5:7b), 2) Simplify tool schemas, 3) Check tool format matches model requirements.",
        "Slow retrieval": "Optimize by: 1) Reduce top_k, 2) Use approximate nearest neighbor (ANN) indexes, 3) Cache frequent queries, 4) Pre-filter by metadata before vector search.",
    },
    "FAQ": {
        "LangChain vs LlamaIndex": "LangChain is better for general LLM application building with agents and chains. LlamaIndex specializes in data indexing and retrieval. Use LangChain for complex workflows, LlamaIndex for pure RAG.",
        "When to use Agents vs Chains": "Use chains for predictable, fixed workflows (always: retrieve → generate). Use agents when the path is uncertain (needs tool selection, multi-step reasoning, adaptation to intermediate results).",
        "OpenAI vs Ollama": "OpenAI: better quality, faster, paid, requires internet. Ollama: free, private, local, slower, smaller models. Use OpenAI for production, Ollama for development and privacy-sensitive applications.",
    },
}


# ==============================================================================
# 构建检索索引
# ==============================================================================

class _SentenceTransformerEmbeddings:
    """Adapter: SentenceTransformer → ChromaDB embedding interface."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, show_progress_bar=False).tolist()


def build_knowledge_base() -> tuple:
    """Build ChromaDB vector store and BM25 keyword index from structured knowledge base."""
    all_docs = []
    for category, entries in KNOWLEDGE_BASE.items():
        for title, content in entries.items():
            doc = Document(
                page_content=f"[{category}] {title}: {content}",
                metadata={"category": category, "title": title},
            )
            all_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(all_docs)

    embeddings = _SentenceTransformerEmbeddings()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K

    return vectorstore, bm25, embeddings, chunks


# ==============================================================================
# RAG 工具——提供给 Deep Agent
# ==============================================================================
# 每个工具都有详细的 description，指导 Agent 根据问题类型自主选择
# description 中包含了工具的适用场景，Agent 会据此做出路由决策

_VECTORSTORE: Chroma | None = None
_BM25_RETRIEVER: BM25Retriever | None = None


def _format_results(docs: list[Document], method: str) -> str:
    """Format search results as JSON list with metadata for citation."""
    entries = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        content_escaped = doc.page_content.replace('"', '\\"')
        entries.append(
            f'{{"id": "Doc{i}", "title": "{meta.get("title", "")}", '
            f'"category": "{meta.get("category", "")}", '
            f'"content": "{content_escaped}", '
            f'"method": "{method}"}}'
        )
    return "[" + ", ".join(entries) + "]"


@tool
def vector_search(query: str, top_k: int = TOP_K) -> str:
    """Semantic search using vector embeddings. Use for conceptual questions,
    explanations, comparisons, and any query where meaning matters over exact words.

    Args:
        query: The search query
        top_k: Number of results to return (default: 3)
    """
    if _VECTORSTORE is None:
        return "[]"
    try:
        docs = _VECTORSTORE.similarity_search(query, k=top_k)
        return _format_results(docs, "vector")
    except Exception as e:
        return f'[{{"id": "Error", "content": "Vector search error: {e}"}}]'


@tool
def keyword_search(query: str, top_k: int = TOP_K) -> str:
    """Exact keyword matching using BM25. Use for specific technical terms,
    API names, function names, and when you need precise term matching.

    Args:
        query: The search query
        top_k: Number of results to return (default: 3)
    """
    if _BM25_RETRIEVER is None:
        return "[]"
    try:
        docs = _BM25_RETRIEVER.invoke(query)
        return _format_results(docs[:top_k], "keyword")
    except Exception as e:
        return f'[{{"id": "Error", "content": "Keyword search error: {e}"}}]'


@tool
def hybrid_search(query: str, top_k: int = TOP_K) -> str:
    """Combined semantic + keyword search. Use when the query contains
    both conceptual meaning and specific terms that need exact matching.
    Returns deduplicated results from both methods.

    Args:
        query: The search query
        top_k: Number of results per method (default: 3)
    """
    if _VECTORSTORE is None or _BM25_RETRIEVER is None:
        return "[]"
    try:
        vec_docs = _VECTORSTORE.similarity_search(query, k=top_k)
        kw_docs = _BM25_RETRIEVER.invoke(query)[:top_k]

        seen = set()
        merged = []
        for d in vec_docs + kw_docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                merged.append(d)

        return _format_results(merged, "hybrid")
    except Exception as e:
        return f'[{{"id": "Error", "content": "Hybrid search error: {e}"}}]'


@tool
def evaluate_relevance(query: str, results_json: str) -> str:
    """Evaluate whether search results can answer the user's question.
    Returns JSON with 'sufficient' (bool) and 'reason' (str).
    Call this after search to decide if more retrieval is needed.

    Args:
        query: The original user question
        results_json: Search results as JSON string (from vector/keyword/hybrid_search)
    """
    try:
        results = json.loads(results_json) if isinstance(results_json, str) else results_json
        if isinstance(results, dict):
            results = [results]

        texts = [r.get("content", "") for r in results]
        combined = " ".join(texts)

        query_terms = set(re.findall(r"[a-zA-Z]+", query.lower()))
        content_terms = set(re.findall(r"[a-zA-Z]+", combined.lower()))
        overlap = query_terms & content_terms
        coverage = len(overlap) / max(len(query_terms), 1)

        if coverage > 0.3 and len(combined) > 100:
            return json.dumps({
                "sufficient": True,
                "reason": f"Results cover {coverage:.0%} of query terms with adequate content length ({len(combined)} chars)",
            })
        return json.dumps({
            "sufficient": False,
            "reason": f"Only {coverage:.0%} query term coverage ({len(combined)} chars), need more targeted search",
        })
    except Exception as e:
        return json.dumps({"sufficient": True, "reason": f"Evaluation bypassed: {e}"})


RAG_TOOLS = [vector_search, keyword_search, hybrid_search, evaluate_relevance]


# ==============================================================================
# 在模块加载时初始化知识库
# ==============================================================================

_VECTORSTORE, _BM25_RETRIEVER, _EMBEDDINGS, _CHUNKS = build_knowledge_base()
print(f"知识库已初始化: {len(_CHUNKS)} 个文档块，{len(KNOWLEDGE_BASE)} 个分类\n")


# ==============================================================================
# 示例 1: 基础 RAG Agent
# ==============================================================================
print("=" * 60)
print("示例 1: 基础 RAG Agent — create_deep_agent + RAG 工具")
print("=" * 60)

# Deep Agents 的 create_deep_agent 一行创建 Agent
# 传入 RAG 工具后，Agent 自动获得检索能力
# 替代写法：
#   agent = create_deep_agent(model="openai:gpt-4o-mini", tools=RAG_TOOLS)

if _DEEPAGENTS_AVAILABLE:
    try:
        agent = create_deep_agent(
            model=f"ollama:{LLM_MODEL}",
            tools=RAG_TOOLS,
            system_prompt="""You are a RAG (Retrieval-Augmented Generation) assistant.
You have access to search tools. When asked a question:
1. Use vector_search for conceptual questions
2. Use keyword_search for specific terms
3. Use hybrid_search for mixed queries
4. ALWAYS cite sources in your answer using [Doc1], [Doc2] format
5. If results are insufficient, use evaluate_relevance and try again""",
        )

        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "What is LangChain and what are its key components?",
            }]
        })
        print(f"\n1.1 RAG Agent 回答:\n{result['messages'][-1].content[:400]}\n")

    except Exception as e:
        print(f"   ⚠️  llama3.2:1b 工具调用可能失败: {str(e)[:120]}\n")
else:
    _check_available()

print("1. 基础 RAG Agent 的核心价值:")
print("   - create_deep_agent 一行代码创建 Agent")
print("   - 工具自动注入，Agent 自主选择检索策略")
print("   - 无需手动编排检索流程")
print()


# ==============================================================================
# 示例 2: 查询分解 — write_todos
# ==============================================================================
print("=" * 60)
print("示例 2: 查询分解 — write_todos 多步规划")
print("=" * 60)

# Deep Agents 内置 TodoListMiddleware，自动处理多步骤任务
# 当遇到多部分问题时，Agent 会：
#   1. 分解问题为子查询
#   2. 为每个子查询创建 todo
#   3. 逐一检索并标记完成
#   4. 综合所有子查询结果

if _DEEPAGENTS_AVAILABLE:
    try:
        agent = create_deep_agent(
            model=f"ollama:{LLM_MODEL}",
            tools=RAG_TOOLS,
            system_prompt="""You are a RAG assistant. When you receive a multi-part question:
1. Use write_todos to break it down into sub-queries
2. Search for each part separately using the appropriate tool
3. Synthesize all results into a coherent answer
4. Cite sources with [DocN]""",
        )

        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Compare LangChain agents and chains: what each one is for, and when should I use which?",
            }]
        })
        print(f"\n2. 查询分解结果:\n{result['messages'][-1].content[:400]}\n")

    except Exception as e:
        print(f"   跳过/错误: {str(e)[:80]}\n")
else:
    _check_available()

print("2. 查询分解的核心价值:")
print("   - Agent 自动拆解复杂问题为子查询")
print("   - write_todos 提供进度追踪")
print("   - 综合多路检索结果为统一回答")
print()


# ==============================================================================
# 示例 3: 多检索器路由
# ==============================================================================
print("=" * 60)
print("示例 3: 多检索器路由 — Agent 自主选择检索策略")
print("=" * 60)

# Agent 通过工具 description 理解每个检索策略的用途：
#   - vector_search: "Use for conceptual questions" → 概念性问题
#   - keyword_search: "Use for specific technical terms" → 精确术语
#   - hybrid_search: "Use when query contains both" → 混合场景
# Agent 的 LLM 根据问题内容决定选哪个工具

if _DEEPAGENTS_AVAILABLE:
    try:
        agent = create_deep_agent(
            model=f"ollama:{LLM_MODEL}",
            tools=RAG_TOOLS,
            system_prompt="""You are a RAG assistant. Choose the right search tool:
- For conceptual/meaning questions → vector_search
- For specific API/function names → keyword_search
- For mixed queries → hybrid_search
Cite sources with [DocN].""",
        )

        # 概念性问题（应选 vector_search）
        result1 = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Explain how RAG pipeline design works in LangChain.",
            }]
        })
        print(f"\n3.1 概念性问题:\n{result1['messages'][-1].content[:300]}\n")

        # 精确术语问题（应选 keyword_search）
        result2 = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "What is create_tool_calling_agent and how is it different from create_react_agent?",
            }]
        })
        print(f"\n3.2 精确术语问题:\n{result2['messages'][-1].content[:300]}\n")

    except Exception as e:
        print(f"   跳过/错误: {str(e)[:80]}\n")
else:
    _check_available()

print("3. 多检索器路由的核心价值:")
print("   - Agent 通过 tool description 理解工具用途")
print("   - 不同问题类型自动选不同检索策略")
print("   - 与 37_search_agent.py 的条件边区别：")
print("     37 用硬编码路由，这里用 Agent 自主决策")
print()


# ==============================================================================
# 示例 4: 结果评估与迭代
# ==============================================================================
print("=" * 60)
print("示例 4: 结果评估与迭代 — evaluate_relevance + 重检索")
print("=" * 60)

# Agent 可以在检索后调用 evaluate_relevance 判断结果是否足够
# 如果不足，Agent 可以换关键词或换检索策略重新搜索
# 这是 Agent RAG 相比固定 RAG pipeline 的核心优势

if _DEEPAGENTS_AVAILABLE:
    try:
        agent = create_deep_agent(
            model=f"ollama:{LLM_MODEL}",
            tools=RAG_TOOLS,
            system_prompt="""You are a RAG assistant. For each question:
1. Search using the appropriate tool
2. Call evaluate_relevance to check if results are sufficient
3. If insufficient, try a different search tool or query terms
4. Repeat up to 3 times max
5. Answer with what you have, citing sources""",
        )

        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "What are the differences between OpenAI and Ollama for production use?",
            }]
        })
        print(f"\n4. 评估与迭代结果:\n{result['messages'][-1].content[:400]}\n")

    except Exception as e:
        print(f"   跳过/错误: {str(e)[:80]}\n")
else:
    _check_available()

print("4. 结果评估与迭代的核心价值:")
print("   - Agent 自主判断结果是否足够")
print("   - 不足则自动调整策略重搜")
print("   - 固定 RAG pipeline 不具备这种自适应能力")
print()


# ==============================================================================
# 示例 5: 多源信息融合与溯源引用
# ==============================================================================
print("=" * 60)
print("示例 5: 多源信息融合与溯源引用")
print("=" * 60)

# 当 Agent 从多个工具获取结果时，会自动合并去重
# 通过 metadata 中的 title/category 信息实现溯源
# 最终回答以 [Doc1][Doc2] 格式引用来源

if _DEEPAGENTS_AVAILABLE:
    try:
        agent = create_deep_agent(
            model=f"ollama:{LLM_MODEL}",
            tools=RAG_TOOLS,
            system_prompt="""You are a RAG assistant. When answering:
1. Search from MULTIPLE sources using different tools
2. Combine and deduplicate information
3. Cite each source as [DocN] in your answer
4. At the end, list all referenced sources with their titles
5. If sources conflict, mention both perspectives""",
        )

        # 使用 hybrid_search 自动获取多源结果
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "How do memory systems work in LangChain? What types are available?",
            }]
        })
        print(f"\n5. 多源融合回答:\n{result['messages'][-1].content[:400]}\n")

    except Exception as e:
        print(f"   跳过/错误: {str(e)[:80]}\n")
else:
    _check_available()

print("5. 多源融合与溯源引用的核心价值:")
print("   - 多检索器结果自动合并去重")
print("   - 引用溯源增加回答可信度")
print("   - 用户可以追溯信息来源")
print()


# ==============================================================================
# 示例 6: 人机协同 + 虚拟文件系统
# ==============================================================================
print("=" * 60)
print("示例 6: 人机协同 + 虚拟文件系统 — interrupt_on")
print("=" * 60)

# 使用 interrupt_on 配置，Agent 写文件前暂停等待人工审批
# 结合虚拟文件系统，Agent 可以将检索报告保存到文件系统
#
# 替代写法（用更强模型）：
#   agent = create_deep_agent(
#       model="openai:gpt-4o-mini",
#       interrupt_on={"write_file": True},
#       checkpointer=MemorySaver(),
#   )

if _DEEPAGENTS_AVAILABLE:
    try:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

        agent = create_deep_agent(
            model=f"ollama:{LLM_MODEL}",
            tools=RAG_TOOLS,
            interrupt_on={"write_file": True},
            checkpointer=checkpointer,
            system_prompt="""You are a RAG research assistant.
When asked to research a topic:
1. Search using your RAG tools
2. Write the findings to a file for persistence
3. Before writing, pause for human approval
4. Summarize what you found""",
        )

        print("   演示配置（不实际运行 interrupt）:")
        print(f"   - interrupt_on: {{'write_file': True}}")
        print(f"   - checkpointer: MemorySaver")
        print(f"   - 流程:")
        print("     1. Agent 检索信息")
        print("     2. 调用 write_file 保存报告")
        print("     3. ⏸ 暂停等待审批")
        print("     4. resume={'approved': True} → 继续")
        print("     5. resume={'approved': False} → 拒绝")
        print()

        # 实际运行时，invoke 会暂停在 write_file 处：
        # result = agent.invoke(
        #     {"messages": [{"role": "user",
        #       "content": "Research LangChain memory systems and save a report."}]},
        #     config={"configurable": {"thread_id": "rag_demo_1"}},
        # )
        # # Agent writes file → interrupt!
        # # 人工查看报告内容:
        # print(agent.get_state(...).values)
        # # 批准后恢复:
        # result = agent.invoke(
        #     None,
        #     config={"configurable": {"thread_id": "rag_demo_1"}},
        #     resume={"approved": True},
        # )

    except Exception as e:
        print(f"   跳过: {e}\n")
else:
    _check_available()

print("6. 人机协同的核心价值:")
print("   - 关键操作（写文件、保存报告）需人工审批")
print("   - 适合需要审核的场景：研究报告、分析报告")
print("   - interrupt_on 不依赖模型推理能力")
print()


# ==============================================================================
# 附录：模型能力评估（Agent RAG 场景）
# ==============================================================================
print("=" * 60)
print("附录: 模型能力评估 — Agent RAG 场景")
print("=" * 60)

print("""
        Agent RAG 不同于普通 RAG，对模型有更高的 tool calling 和 reasoning 要求：

        能力                      | llama3.2:1b    | qwen2.5:7b     | gpt-4o-mini
        -------------------------|---------------|---------------|---------------
        基础检索（vector_search）  | ✅ 可用       | ✅ 良好       | ✅ 优秀
        关键词检索（keyword）      | ✅ 可用       | ✅ 良好       | ✅ 优秀
        多检索器路由决策           | ⚠️  不稳定    | ✅ 良好       | ✅ 优秀
        查询分解（write_todos）    | ❌ 较弱       | ⚠️  可用      | ✅ 优秀
        结果评估（evaluate）       | ⚠️  不稳定    | ✅ 良好       | ✅ 优秀
        迭代重检索                 | ❌ 较弱       | ⚠️  可用      | ✅ 优秀
        溯源引用                   | ✅ 可用       | ✅ 良好       | ✅ 优秀
        人机协同（interrupt_on）   | ✅ 可用       | ✅ 可用       | ✅ 可用

        推荐模型升级路径:
          1. ollama:llama3.2:1b    — 理解 Agent RAG 概念
          2. ollama:qwen2.5:7b     — 更好的工具调用和路由
          3. openai:gpt-4o-mini    — 生产环境首选
          4. anthropic:claude-sonnet-4-6 — 最强 Agent 能力

        参考: https://github.com/langchain-ai/deepagents/tree/main/libs/evals
""")


# ==============================================================================
# 主入口
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Deep Agents RAG — 全部示例执行完成")
    print("=" * 60)
    print()
    print("各示例开关（注释/取消注释以控制）:")
    print("  示例 1: 基础 RAG Agent  ✓")
    print("  示例 2: 查询分解       ✓")
    print("  示例 3: 多检索器路由    ✓")
    print("  示例 4: 结果评估与迭代  ✓")
    print("  示例 5: 多源融合与引用  ✓")
    print("  示例 6: 人机协同+VFS   ✓")
