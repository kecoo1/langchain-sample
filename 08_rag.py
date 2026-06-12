"""简单的 RAG（检索增强生成）示例。"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. 从原始文本创建文档
raw_text = """
LangChain is a framework for developing applications powered by large language models.
It provides a standard interface for chains, prompts, and agents.
LangChain supports many model providers including OpenAI, Anthropic, and Ollama.

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
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
)
documents = text_splitter.create_documents([raw_text])
print(f"Created {len(documents)} document chunks\n")

# 2. 创建向量存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. RAG 链
template = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

# 4. 查询示例
questions = [
    "What is LangChain?",
    "What are the key components of LangChain?",
    "What is LangGraph?",
]

for q in questions:
    result = rag_chain.invoke(q)
    print(f"Q: {q}")
    print(f"A: {result}\n")

# Clean up
vectorstore.delete_collection()

# =============================================================================
# 教学备注：RAG（检索增强生成）——解决 LLM 的"我不知道"问题
# =============================================================================
# 核心问题：LLM 的知识有"截止日期"（比如不知道 2024 年后的新闻），
#   而且它"不知道的就是不知道"（不会说"我不知道"，而是会"幻觉"——编造答案）
#   RAG 的思路：不依赖模型"记住"知识，而是让模型"读到"知识再回答
#   类比：考试时模型是"闭卷"还是"开卷"？RAG 就是给它开卷——先查资料再答题
#
# 第一步：文档分割（Chunking）——把书拆成便签
#   为什么需要分割：模型上下文有上限（128K token），整本书放不进去
#     而且检索时"整本书"太粗了，你需要的是"相关段落"
#   RecursiveCharacterTextSplitter 的智慧：按层级递归切分
#     先按段落切（\n\n）-> 段落太大再按句子切（.!?）-> 再按字符切
#     这样尽量保证每个 chunk 语义完整（不把一句话切成两半）
#   chunk_size 和 chunk_overlap 是什么：
#     chunk_size=150：每个 chunk 约 150 个字符
#     chunk_overlap=20：相邻 chunk 重叠 20 个字符，防止关键信息恰好被切在边界丢失
#   不同文本选不同分割器：
#     普通文本：RecursiveCharacterTextSplitter 通用推荐
#     代码文件：按函数/类分割（PythonCodeTextSplitter）
#     Markdown：MarkdownHeaderTextSplitter 保留标题层级
#     长文档（PDF）：TokenTextSplitter 按 token 精确控制
#
# 第二步：嵌入（Embedding）——把文字变成向量
#   为什么需要向量：计算机不理解"LangChain is a framework"，但它理解 [0.23, -0.45, 0.78, ...]
#     嵌入模型的任务：把任何文本映射到高维空间的一个点
#     语义相近的文本 -> 向量空间中的距离近
#   OpenAIEmbeddings(model="text-embedding-3-small")：
#     把 "LangChain is a framework" 变成一个 1536 维的向量
#     相似的问题 "What is LangChain?" 的向量会离它很近
#   选嵌入模型的原则：
#     质量 vs 成本：text-embedding-3-small 便宜但够用，3-large 更准但贵 5 倍
#     语言：中文场景选 text-embedding-3-small（支持多语言）或 BGE-M3（国产专门优化）
#     离线：sentence-transformers/all-MiniLM-L6-v2 只要几百 MB
#   Token 成本意识：文档被嵌入时是按 token 计费的，10 万条文档的 embedding 成本不可忽略
#
# 第三步：向量存储（Vector Store）——建索引
#   为什么需要专门的存储：不能每次检索都算一遍所有文档的向量
#     向量数据库做的事情：建索引（HNSW/IVF）-> 近似最近邻搜索（ANN）-> 毫秒级响应
#   Chroma 的角色：本地轻量向量数据库，全程在内存/Disk 中运行
#     适合：原型验证、中小规模（<100 万条）、单机部署
#     不适合：分布式、大规模、高可用生产环境
#   从 Chroma 到生产的迁移路径：
#     Chroma -> FAISS（性能更快，但内存）-> PGVector（复用 Postgres）-> Pinecone/Qdrant/Milvus（托管分布式）
#   search_kwargs={"k": 3}：返回最相似的 3 个 chunk
#     k 越大 -> 召回更多 -> 可能包含不相关信息 -> 影响回答质量
#     k 越小 -> 召回更少 -> 可能遗漏关键信息 -> 模型知识不足
#     经验值：k=3~5 适合大多数场景
#
# 第四步：构建 RAG 链——串联起来
#   rag_chain 的结构解读：
#     {
#         "context": retriever | format_docs,     # 先检索，再格式化
#         "question": RunnablePassthrough(),       # 原样传递问题
#     }
#     | prompt                                      # 填入模板：Context + Question
#     | model                                       # 模型根据 context 回答
#     | StrOutputParser()                           # 提取文本
#   为什么用 RunnablePassthrough()？
#     因为模板需要两个变量：context 和 question
#     输入只有 question，retriever 算出 context，question 原样传递
#     RunnablePassthrough 就是"原样传递"的意思
#
# 第五步：format_docs 做了什么？
#   问题：retriever 返回的是 List[Document]（Python 对象），不能直接塞进模板
#     需要把 [Document1, Document2, Document3] -> "文本1\n\n文本2\n\n文本3"
#   为什么用 "\n\n" 分隔：让模型清晰感知"这是三个独立段落"
#   进阶：这里可以加入元信息——"以下来自第 3 章第 2 节：\n{text}"
#
# RAG 的常见痛点（生产必读）：
#   1. 检索不到：问题用了"框架"而文档写的是"library"——需要同义词/多检索
#      方案：查询扩展（HyDE）、多查询（MultiQueryRetriever）
#   2. 检索太多：返回了 10 个 chunk，模型"看不过来"，注意力被稀释
#      方案：先检 Top-20，再用重排序（Cross-encoder）缩到 Top-3
#   3. 回答不准确：模型正确检索了，但自己发挥写错了
#      方案：加 prompt 约束"严格根据 context 回答，不知道就说不知道"
#   4. 来源不可追溯：用户问"你怎么知道的"——回答不出来
#      方案：保留 doc_id/metadata，在答案中引用来源 [1][2]
#
# 教学建议顺序：
#   1. 先理解"为什么要 RAG"——解决知识截止和幻觉
#   2. 再理解四个步骤：拆分 -> 嵌入 -> 存储 -> 检索+生成
#   3. 手动跑通这个例子，感受"没有 RAG 时模型的幻想 vs 有 RAG 时的基于事实回答"
#   4. 再学习进阶优化：重排序、查询扩展、HyDE、多模态 RAG
