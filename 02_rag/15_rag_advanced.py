"""RAG 进阶示例 —— 混合检索、重排序、来源引用、HyDE。

进阶 RAG 解决初级 RAG 的痛点：
1. 检索不精准 -> 混合检索（关键词 + 向量）
2. 返回太多/太少 -> 重排序（Cross-encoder 精排）
3. 答案无来源 -> 引用溯源（在答案中标注引用）
4. 语义不匹配 -> HyDE（假设文档嵌入）
"""

from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="llama3.2:1b", temperature=0)

# 示例文本
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
"""

# 1. 创建文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
documents = text_splitter.create_documents([DOCS_TEXT])

# ==============================================================================
# 示例 1：混合检索（关键词 + 向量）
# ==============================================================================
print("=== 示例 1: 混合检索 ===")

# 向量嵌入
vector_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 向量存储
vectorstore = Chroma.from_documents(documents=documents, embedding=vector_embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# BM25 关键词检索（基于 Whoosh，无需外部服务）
from langchain_community.retrievers import BM25Retriever

# 从文档初始化 BM25
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 混合：向量 + BM25，各自取 Top-k，再用 Reciprocal Rank Fusion 合并
from langchain_core.retrievers import CompressorRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import CharacterTextSplitter

# 使用 EnsembleRetriever 做 RR（Reciprocal Rank Fusion）融合
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],  # 向量检索和关键词检索各占 50% 权重
)

print(f"   混合检索返回 {len(ensemble_retriever.invoke('What is LangGraph?'))} 个文档\n")


# ==============================================================================
# 示例 2：上下文重排序（Reranking）
# ==============================================================================
print("=== 示例 2: 重排序 ===")

# 第一步：召回（用向量检索取较多的候选）
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderCompressor
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 粗召回 Top-10
recall_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 精排：用 Cross-encoder 对召回结果重新打分排序
# 注意：需要下载 cross-encoder 模型，首次运行会下载约 250MB
# 这里用轻量方案：用 LLM 做简单重排序

def rerank_with_llm(query: str, docs: list[Document]) -> list[Document]:
    """用 LLM 对文档相关性打分并排序。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Rank these documents by relevance to the query.
Return ONLY a JSON array of indices sorted by relevance (most relevant first).
Example: [2, 0, 1]"""),
        ("human", "Query: {query}\n\nDocuments:\n{documents}"),
    ])

    doc_text = "\n---\n".join(
        f"[{i}] {doc.page_content}" for i, doc in enumerate(docs)
    )

    ranked = prompt | llm | StrOutputParser()
    result = ranked.invoke({"query": query, "documents": doc_text})

    # 简单解析（实际生产用 Pydantic 解析）
    try:
        import json
        indices = json.loads(result.strip())
        return [docs[i] for i in indices if i < len(docs)]
    except:
        return docs  # 解析失败返回原序


sample_docs = vector_retriever.invoke("What is LangSmith?")
reranked = rerank_with_llm("What is LangSmith?", sample_docs)
print(f"   重排序后返回 {len(reranked)} 个文档:")
for i, doc in enumerate(reranked[:3]):
    print(f"   {i+1}. {doc.page_content[:80]}...")
print()


# ==============================================================================
# 示例 3：来源引用 —— 在答案中标注引用
# ==============================================================================
print("=== 示例 3: 来源引用 ===")

# 增强 Prompt：要求在回答中引用来源
citation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question based ONLY on the
provided context. When referencing information, cite the source using [Doc 1], [Doc 2], etc.
If you cannot answer from the context, say "I don't have enough information."

Context:
{context}"""),
    ("human", "{question}"),
])


def format_docs_with_refs(docs: list[Document]) -> str:
    """格式化文档并添加引用标记。"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[Doc {i}] {doc.page_content}")
    return "\n\n".join(formatted)


# 使用重排序后的文档
rag_with_citations = (
    {
        "context": ensemble_retriever | format_docs_with_refs,
        "question": RunnablePassthrough(),
    }
    | citation_prompt
    | llm
    | StrOutputParser()
)

questions_with_refs = [
    "What is LangChain?",
    "What is LangGraph?",
    "What does LangServe do?",
]

for q in questions_with_refs:
    answer = rag_with_citations.invoke(q)
    print(f"Q: {q}")
    print(f"A: {answer}")
    print()


# ==============================================================================
# 示例 4：HyDE —— 假设文档嵌入
# ==============================================================================
print("=== 示例 4: HyDE (Hypothetical Document Embeddings) ===")


class HyDERetriever:
    """HyDE 检索器：先生成"假设答案"，用假设答案的向量去检索真实文档。"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a question, create a fictional document paragraph that
would contain the answer. Write it as if from a knowledge base article.
Do NOT answer the question directly — write a plausible explanatory paragraph."""),
            ("human", "{question}"),
        ])
        self.hyde_chain = self.hyde_prompt | llm | StrOutputParser()

    def invoke(self, query: str, k: int = 3):
        # 步骤 1: 生成假设文档
        hypothetical = self.hyde_chain.invoke({"question": query})

        # 步骤 2: 用假设文档的向量去检索
        docs = self.vectorstore.similarity_search(hypothetical, k=k)

        return docs, hypothetical


hyde = HyDERetriever(vectorstore, llm)
docs, hypothetical = hyde.invoke("What are the key components of LangChain?")

print(f"   假设文档: {hypothetical[:150]}...")
print(f"   检索到 {len(docs)} 个文档:")
for i, doc in enumerate(docs, 1):
    print(f"   {i}. {doc.page_content[:80]}...")
print()


# ==============================================================================
# 示例 5：多查询检索（MultiQueryRetriever）
# ==============================================================================
print("=== 示例 5: 多查询检索 ===")

from langchain.retrievers.multi_query import MultiQueryRetriever

# 为同一个问题生成多个查询，提高召回率
multi_query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=multi_query_llm,
)

# 多查询会生成不同的变体问题再分别检索
results = multi_retriever.invoke("What does LangChain do?")
unique_docs = list({id(d): d for d in results}.values())  # 去重
print(f"   多查询变体后检索到 {len(unique_docs)} 个唯一文档\n")


# ==============================================================================
# 示例 6：完整进阶 RAG 链
# ==============================================================================
print("=== 示例 6: 完整进阶 RAG 链 ===")


class AdvancedRAGChain:
    """完整的进阶 RAG 链，集成混合检索、重排序和引用。"""

    def __init__(self, vectorstore, llm, embeddings):
        self.llm = llm
        self.vectorstore = vectorstore

        # 混合检索组件
        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 8
        self.ensemble = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.6, 0.4],
        )

        # 重排序
        self.citation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer based ONLY on the context. Cite sources as [Doc N].
If you cannot answer, say "I don't have enough information."

Context:
{context}

Question: {question}"""),
        ])

    def invoke(self, question: str, max_chunks: int = 5) -> dict:
        # 1. 混合检索
        docs = self.ensemble.invoke(question)

        # 2. 取 Top-K
        docs = docs[:max_chunks]

        # 3. 格式化
        context = format_docs_with_refs(docs)

        # 4. 生成答案
        answer = (
            self.citation_prompt
            | self.llm
            | StrOutputParser()
        ).invoke({"context": context, "question": question})

        return {
            "question": question,
            "answer": answer,
            "sources": [doc.page_content for doc in docs],
            "num_sources": len(docs),
        }


# 使用
advanced_rag = AdvancedRAGChain(vectorstore, llm, vector_embeddings)

test_questions = [
    "What is LangChain and what are its components?",
    "How does LangGraph relate to LangChain?",
    "What deployment options exist for LangChain applications?",
]

for q in test_questions:
    result = advanced_rag.invoke(q)
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['num_sources']}")
    print()

# Clean up
vectorstore.delete_collection()

# ==============================================================================
# 教学备注：RAG 进阶 —— 从"能用"到"好用"
# ==============================================================================
# 核心痛点 1：检索不精准
#   初级 RAG 只用向量检索，问题是：
#     - 向量检索对语义相似敏感，但对精确关键词匹配不够
#       例：用户搜 "LangServe" 而文档中有 "LangChain"，语义相近但不精确
#     - 向量嵌入在高维空间会稀释"特有名词"的区分度
#   解决方案：混合检索（向量 + BM25）
#     BM25 擅长关键词匹配、专有名词匹配
#     向量擅长语义匹配、同义词匹配
#     两者互补：RR Fusion 融合结果 = 兼顾语义和精确
#     weights=[0.6, 0.4] = 向量占 60% 权重（语义优先）或调整到 [0.5, 0.5]
#
# 核心痛点 2：召回太多，信息噪音
#   问题：k=10 召回 10 个 chunk，模型"看不过来"，注意力被稀释
#   解决方案：两阶段检索
#     阶段 1（召回）：快速召回 Top-10~20（用向量检索，快）
#     阶段 2（精排）：用 Cross-encoder 或 LLM 对 Top-10 重新打分排序
#       Cross-encoder：将 query + doc 一起输入模型，计算精确相关性分数
#       精度高于 Bi-encoder（向量检索），但速度慢 10-50 倍
#       所以只在第二阶段用：召回快 -> 精排准
#
# 核心痛点 3：答案不可信
#   问题：用户问"你怎么知道的"，模型回答不出来
#   解决方案：引用溯源
#     在 Prompt 中要求模型输出 [Doc N] 引用标记
#     在 UI 中展示引用来源，用户可以点击查看原文
#     这是"信任建设"的关键——不是让模型说"我相信"，而是展示证据
#
# 核心痛点 4：语义鸿沟
#   问题：用户用日常语言提问，文档用专业术语描述
#     例：用户说"AI 怎么查资料"，文档写"检索增强生成（Retrieval Augmented Generation）"
#   解决方案：HyDE
#     先生成一个"假设性答案文档"，用这个文档的向量去检索
#     假设文档的措辞更接近真实用户提问 -> 向量空间距离更近 -> 召回更准
#     类比：翻译问题 -> 翻译成专业术语 -> 去专业文档库检索
#
# 核心痛点 5：一问题一答案太死板
#   问题：用户一个问题，可能隐含多个意图
#   解决方案：MultiQueryRetriever
#     用 LLM 生成 3-5 个变体问题
#     每个变体分别检索，合并结果后去重
#     类比：Google 搜索的同义词扩展
#
# RAG 进阶架构（生产级）:
#   用户问题 -> MultiQuery（生成变体）-> Ensemble（向量 + BM25）-> Top-20
#         -> CrossEncoder Rerank -> Top-5 -> LLM 生成 + 引用
#
# 选型决策树:
#   ┌─ 检索不准？ → 混合检索 (EnsembleRetriever)
#   ├─ 太多噪音？ → 重排序 (CrossEncoderCompressor)
#   ├─ 无来源？ → 引用模板 + format_docs_with_refs
#   ├─ 语义不匹配？ → HyDE
#   └─ 覆盖面不够？ → MultiQueryRetriever
#
# 教学建议顺序:
#   1. 先跑通示例 1（混合检索），感受 "关键词 + 语义" 互补
#   2. 再跑通示例 3（引用），理解"信任建设"
#   3. 再跑通示例 4（HyDE），理解"假设文档嵌入"
#   4. 最后整合到示例 6（完整进阶 RAG 链）
