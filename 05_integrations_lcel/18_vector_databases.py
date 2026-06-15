"""向量数据库对比示例 —— FAISS、Pinecone、Qdrant、PGVector。

展示如何在相同的 RAG 管道中切换不同的向量数据库后端。
每个向量数据库有不同的特点，适合不同的使用场景。

注意: 本文件展示了接口调用方式，实际使用前需要安装对应的包:
  pip install faiss-cpu          # FAISS
  pip install pinecone-client     # Pinecone (需 API Key)
  pip install qdrant-client       # Qdrant (需 API Key 或本地运行)
  pip install psycopg2-binary     # PGVector (需 PostgreSQL 数据库)
"""

from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 示例文本
SAMPLE_DOCS_TEXT = """
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

# 创建文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
documents = text_splitter.create_documents([SAMPLE_DOCS_TEXT])
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==============================================================================
# 示例 1：Chroma（本地轻量）
# ==============================================================================
print("=== 示例 1: Chroma (本地轻量) ===")

chroma_store = Chroma.from_documents(documents=documents, embedding=embeddings)
chroma_retriever = chroma_store.as_retriever(search_kwargs={"k": 3})

results = chroma_retriever.invoke("What is LangGraph?")
print(f"   返回 {len(results)} 个文档:")
for i, doc in enumerate(results, 1):
    print(f"   {i}. {doc.page_content[:80]}...")
print(f"   特点: 零配置、内存/Disk存储、适合原型验证\n")
chroma_store.delete_collection()


# ==============================================================================
# 示例 2：FAISS（快速内存索引）
# ==============================================================================
print("=== 示例 2: FAISS (快速内存索引) ===")

try:
    from langchain_community.vectorstores import FAISS

    faiss_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

    results = faiss_retriever.invoke("What is LangGraph?")
    print(f"   返回 {len(results)} 个文档:")
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content[:80]}...")
    print(f"   特点: 极快检索、内存驻留、适合小规模 (<100万) 数据")
    print(f"   持久化: faiss_store.save_directory('/path/to/dir')")
    print(f"   加载: FAISS.load_from_index('/path/to/dir')")
    print()
except ImportError:
    print("   跳过: 需要安装 pip install faiss-cpu\n")


# ==============================================================================
# 示例 3：Pinecone（托管云服务）
# ==============================================================================
print("=== 示例 3: Pinecone (托管云服务) ===")

try:
    from langchain_pinecone import PineconeVectorStore
    import pinecone

    # 首次使用需要: pip install pinecone-client pinecone-text
    # 并设置环境变量: export PINECONE_API_KEY="your-key"

    # 初始化 Pinecone 客户端
    # pc = pinecone.Pinecone(api_key="your-api-key")

    # 创建索引（首次）
    # pc.create_index(
    #     name="langchain-demo",
    #     dimension=384,  # HuggingFace all-MiniLM-L6-v2 的维度
    #     metric="cosine",
    # )

    # 连接已有索引
    # pinecone_store = PineconeVectorStore(
    #     index=pc.Index("langchain-demo"),
    #     embedding=embeddings,
    # )

    # 添加文档
    # pinecone_store.add_documents(documents)

    # 检索
    # results = pinecone_store.similarity_search("What is LangGraph?", k=3)

    print("   特点: 完全托管、自动扩展、高可用、适合大规模生产")
    print("   定价: 按存储量和查询量付费")
    print("   适用: 企业级应用、需要自动扩展和高可用")
    print("   注意: 取消注释上面的代码以实际使用\n")
except ImportError:
    print("   跳过: 需要安装 pip install pinecone-client langchain-pinecone\n")


# ==============================================================================
# 示例 4：Qdrant（高性能开源）
# ==============================================================================
print("=== 示例 4: Qdrant (高性能开源) ===")

try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    # 首次使用需要: pip install qdrant-client langchain-qdrant
    # 本地运行: docker run -p 6333:6333 qdrant/qdrant

    # 连接本地 Qdrant
    # client = QdrantClient(host="localhost", port=6333)

    # 创建向量存储
    # qdrant_store = QdrantVectorStore(
    #     client=client,
    #     collection_name="langchain_docs",
    #     embedding=embeddings,
    # )

    # 添加文档
    # qdrant_store.add_documents(documents)

    # 检索
    # results = qdrant_store.similarity_search("What is LangGraph?", k=3)

    # 带过滤的检索（例如按来源过滤）
    # results = qdrant_store.similarity_search(
    #     "What is LangGraph?",
    #     k=3,
    #     filter={"must": [{"key": "source", "match": {"value": "docs"}}]},
    # )

    print("   特点: Rust 编写、高性能、支持丰富过滤、可本地或云端部署")
    print("   适用: 需要高性能过滤检索的场景")
    print("   Docker: docker run -p 6333:6333 qdrant/qdrant")
    print("   注意: 取消注释上面的代码以实际使用\n")
except ImportError:
    print("   跳过: 需要安装 pip install qdrant-client langchain-qdrant\n")


# ==============================================================================
# 示例 5：PGVector（PostgreSQL 扩展）
# ==============================================================================
print("=== 示例 5: PGVector (PostgreSQL 扩展) ===")

try:
    from langchain_postgres import PGVector
    from sqlalchemy import create_engine

    # 首次使用需要: pip install psycopg2-binary langchain-postgres
    # 需要运行 PostgreSQL 并安装 pgvector 扩展

    # 连接 PostgreSQL
    # DATABASE_URL = "postgresql://user:password@localhost:5432/langchain_db"
    # engine = create_engine(DATABASE_URL)

    # 创建向量存储
    # pgvector_store = PGVector(
    #     collections_collection_name="langchain_docs",
    #     embedding_function=embeddings,
    #     connection=engine,
    # )

    # 添加文档
    # pgvector_store.add_documents(documents)

    # 检索
    # results = pgvector_store.similarity_search("What is LangGraph?", k=3)

    # 利用 PostgreSQL 的强大功能：SQL 查询 + 向量检索结合
    # SELECT * FROM documents WHERE content % 'LangGraph' ORDER BY embedding <-> %s LIMIT 3

    print("   特点: 复用现有 PostgreSQL 基础设施、ACID 事务、SQL + 向量联合查询")
    print("   适用: 已有 PostgreSQL 的团队、需要事务支持的场景")
    print("   安装: CREATE EXTENSION vector;")
    print("   注意: 取消注释上面的代码以实际使用\n")
except ImportError:
    print("   跳过: 需要安装 pip install psycopg2-binary langchain-postgres\n")


# ==============================================================================
# 示例 6：Weaviate（语义搜索 + 结构化数据）
# ==============================================================================
print("=== 示例 6: Weaviate (语义搜索 + 结构化数据) ===")

try:
    from langchain_weaviate import WeaviateVectorStore
    import weaviate

    # 首次使用需要: pip install weaviate-client langchain-weaviate
    # 本地运行: docker run -p 8080:8080 -p 5000:5000 weaviate/weaviate

    # 连接本地 Weaviate
    # client = weaviate.connect_to_local()

    # 创建向量存储
    # weaviate_store = WeaviateVectorStore(
    #     client=client,
    #     index_name="LangChainDocs",
    #     embedding=embeddings,
    # )

    # 添加文档
    # weaviate_store.add_documents(documents)

    # 检索
    # results = weaviate_store.similarity_search("What is LangGraph?", k=3)

    print("   特点: 内置向量化、GraphQL 查询、支持混合搜索、自动数据管理")
    print("   适用: 需要语义搜索 + 结构化数据结合的场景")
    print("   注意: 取消注释上面的代码以实际使用\n")
except ImportError:
    print("   跳过: 需要安装 pip install weaviate-client langchain-weaviate\n")


# ==============================================================================
# 示例 7：Milvus（超大规模分布式）
# ==============================================================================
print("=== 示例 7: Milvus (超大规模分布式) ===")

try:
    from langchain_milvus import Milvus

    # 首次使用需要: pip install pymilvus langchain-milvus
    # 本地运行: docker compose up -d (使用官方的 milvus-docker-compose)

    # 连接 Milvus
    # milvus_store = Milvus(
    #     connection_uri="http://localhost:19530",
    #     embedding_function=embeddings,
    #     collection_name="langchain_docs",
    # )

    # 添加文档
    # milvus_store.add_documents(documents)

    # 检索
    # results = milvus_store.similarity_search("What is LangGraph?", k=3)

    print("   特点: 分布式、支持十亿级向量、自动分片、高可用")
    print("   适用: 超大规模数据（>1亿条向量）")
    print("   注意: 取消注释上面的代码以实际使用\n")
except ImportError:
    print("   跳过: 需要安装 pip install pymilvus langchain-milvus\n")


# ==============================================================================
# 示例 8：向量数据库选型决策树
# ==============================================================================
print("=== 示例 8: 向量数据库选型 ===")

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        向量数据库选型指南                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  数据规模?                                                          │
│  ├── < 10 万条                                                     │
│  │   ├── 原型/开发 → Chroma                                        │
│  │   └── 需要快速检索 → FAISS                                      │
│  │                                                                  │
│  ├── 10 万 - 1000 万条                                             │
│  │   ├── 需要过滤 → Qdrant                                         │
│  │   ├── 已有 PostgreSQL → PGVector                                │
│  │   └── 需要语义+结构化 → Weaviate                                │
│  │                                                                  │
│  └── > 1000 万条                                                   │
│      ├── 需要分布式 → Milvus                                       │
│      └── 不想管运维 → Pinecone (托管)                              │
│                                                                     │
│  其他考虑因素:                                                      │
│  ├── 预算: 免费开源 (Chroma/FAISS/Qdrant/Milvus)                   │
│  │       vs 付费托管 (Pinecone/Weaviate Cloud)                     │
│  ├── 部署: 本地 (Chroma/FAISS/Qdrant) vs 云端 (Pinecone)           │
│  ├── 集成: 已有 PostgreSQL → PGVector                              │
│  └── 性能: 延迟敏感 → FAISS; 吞吐敏感 → Milvus                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# ==============================================================================
# 教学备注：向量数据库 —— RAG 的基础设施层
# ==============================================================================
# 核心问题：为什么不能直接用内存列表存储向量？
#   1. 性能：暴力搜索 O(N) 复杂度，百万级数据要几秒；向量数据库用 HNSW/IVF 做到 O(logN)
#   2. 索引：向量数据库维护 ANN（近似最近邻）索引，检索快 100-1000 倍
#   3. 持久化：内存数据断电丢失；向量数据库持久化到磁盘
#   4. 过滤：很多向量数据库支持元数据过滤，缩小搜索范围
#
# 各向量数据库核心对比:
#   Chroma:
#     优点: 零配置、pip install 即用、内存/Disk 两种模式
#     缺点: 不支持分布式、大规模性能差
#     适合: 原型验证、小规模 (<10万)、单机部署
#
#   FAISS (Facebook):
#     优点: 速度极快、内存驻留、C++ 底层
#     缺点: 无持久化、无元数据过滤、需自行管理
#     适合: 快速原型、小规模 (<100万)、延迟敏感
#
#   Pinecone:
#     优点: 完全托管、自动扩展、高可用、全球部署
#     缺点: 付费、数据不在本地
#     适合: 企业级、大规模、不需要运维的团队
#
#   Qdrant:
#     优点: Rust 编写、高性能、丰富的过滤语法、可本地或云端
#     缺点: 需要自行部署和维护
#     适合: 需要高性能 + 过滤的场景
#
#   PGVector:
#     优点: 复用现有 PostgreSQL、ACID 事务、SQL + 向量联合查询
#     缺点: 大规模性能不如专用向量数据库
#     适合: 已有 PostgreSQL 的团队、中小规模
#
#   Weaviate:
#     优点: 内置向量化、GraphQL 查询、混合搜索
#     缺点: 资源占用较高
#     适合: 需要语义搜索 + 结构化数据的场景
#
#   Milvus:
#     优点: 分布式、支持十亿级向量、自动分片
#     缺点: 运维复杂度高
#     适合: 超大规模 (>1亿条向量)
#
# 迁移路径:
#   开发: Chroma → 测试: FAISS → 生产: Pinecone/Qdrant/PGVector
#
# 教学建议顺序:
#   1. 先理解"为什么需要向量数据库"——性能、索引、持久化
#   2. 跑通 Chroma 示例（已安装，零配置）
#   3. 了解各数据库的特点和适用场景
#   4. 根据项目需求选择合适的向量数据库
