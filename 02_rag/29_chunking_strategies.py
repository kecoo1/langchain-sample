"""RAG 分块策略对比 —— 不同分块方式对检索效果的影响。

分块（Chunking）是 RAG 中最容易被忽视但影响最大的环节之一。
分块太大：信息混杂，检索精度低；分块太小：上下文不足，模型理解困难。

本文件对比 5 种常见分块策略：
1. 固定大小分块：最简单的字符数截断
2. 递归字符分块：按段落/句子/单词逐级分割
3. 语义分块：基于语义边界（LLM 判断）切分
4. 代码分块：按函数/类分割（代码场景专用）
5. 混合分块：多种策略组合
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
)
from langchain_chroma import Chroma

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 准备：测试文本
# ==============================================================================
LONG_TEXT = """
# LangChain 框架详解

## 概述
LangChain 是一个用于构建 LLM 应用的框架。
它提供了丰富的工具链，让开发者能够快速构建从简单到复杂的应用。

## 核心模块

### 1. 模型 I/O
模型 I/O 模块负责与 LLM 进行交互。
它包括三个子模块：
- 提示模板（Prompt Templates）：管理和格式化提示词
- 模型接口（Models）：统一调用不同 LLM 提供商
- 输出解析（Output Parsers）：结构化模型输出

### 2. 检索增强生成（RAG）
RAG 是 LangChain 最核心的应用模式之一。
它通过以下步骤工作：
1. 加载文档（Document Loaders）
2. 分割文档（Text Splitters）
3. 向量化存储（Vector Stores）
4. 检索（Retrievers）
5. 生成答案（LLM 结合检索结果）

### 3. 代理系统（Agents）
Agent 是 LangChain 的另一个核心能力。
- 工具调用：Agent 可以调用外部工具
- 记忆系统：维护对话历史
- 规划能力：分解复杂任务

## 最佳实践
在生产环境中，建议：
1. 使用 LangSmith 进行追踪和监控
2. 设置合理的重试机制
3. 实施安全防护措施
4. 进行充分的评估测试

## 性能优化
为了获得最佳性能：
- 选择合适的向量数据库
- 优化分块策略
- 使用缓存减少 API 调用
- 实施流式输出提升用户体验

## 部署方案
LangChain 应用可以通过 LangServe 部署为 REST API，
也可以集成到 FastAPI、Flask 等 Web 框架中。
支持容器化部署（Docker）和云原生部署（Kubernetes）。
""".strip()

print(f"测试文本长度: {len(LONG_TEXT)} 字符\n")

# ==============================================================================
# 策略 1：固定大小分块
# ==============================================================================
print("=== 策略 1: 固定大小分块 ===")

fixed_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator=" ",
    length_function=len,
)

fixed_chunks = fixed_splitter.split_text(LONG_TEXT)
print(f"   分块数: {len(fixed_chunks)}")
print(f"   示例块 1: {fixed_chunks[0][:80]}...")
print(f"   示例块 2: {fixed_chunks[1][:80]}...")
print(f"   问题: 可能在句子中间截断，语义不完整")
print()

# ==============================================================================
# 策略 2：递归字符分块
# ==============================================================================
print("=== 策略 2: 递归字符分块 ===")

# 按 [\n\n, \n, ., ?, !, 空格] 优先级递归分割
# 优先保持段落完整 → 句子完整 → 短语完整
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "],
)

recursive_chunks = recursive_splitter.split_text(LONG_TEXT)
print(f"   分块数: {len(recursive_chunks)}")
print(f"   示例块 1: {recursive_chunks[0][:100]}...")
print(f"   示例块 2: {recursive_chunks[1][:100]}...")
print(f"   优点: 优先在段落/句子边界分割，语义更完整")
print()

# ==============================================================================
# 策略 3：Markdown 结构分块（信息层级感知）
# ==============================================================================
print("=== 策略 3: Markdown 结构分块 ===")

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]
)

md_docs = markdown_splitter.split_text(LONG_TEXT)
print(f"   分块数: {len(md_docs)}")
for i, doc in enumerate(md_docs):
    headers = doc.metadata.get("H1", "") or doc.metadata.get("H2", "") or doc.metadata.get("H3", "")
    print(f"   块 {i+1}: [{headers}] {doc.page_content[:60]}...")
print(f"   优点: 保留文档层级结构，检索时理解上下文归属")
print()

# ==============================================================================
# 策略 4：语义分块（LLM 判断语义边界）
# ==============================================================================
print("=== 策略 4: 语义分块（LLM 判断）===")

# 人工标注的语义边界（真实场景中用 LLM 判断）
# LLM 分析文本后，在"语义完整的段落"边界处分割
semantic_boundaries = [
    "# LangChain 框架详解\n\n## 概述\nLangChain 是一个用于构建 LLM 应用的框架。",
    "## 核心模块\n\n### 1. 模型 I/O\n模型 I/O 模块负责与 LLM 进行交互。",
    "### 2. 检索增强生成（RAG）\nRAG 是 LangChain 最核心的应用模式之一。",
    "### 3. 代理系统（Agents）\nAgent 是 LangChain 的另一个核心能力。",
    "## 最佳实践\n在生产环境中，建议：",
    "## 性能优化\n为了获得最佳性能：",
    "## 部署方案\nLangChain 应用可以通过 LangServe 部署为 REST API",
]

print(f"   语义边界分块数: {len(semantic_boundaries)}")
for i, chunk in enumerate(semantic_boundaries):
    print(f"   块 {i+1}: {chunk[:60]}...")
print(f"   优点: 每个块语义完整，检索命中率最高")
print(f"   代价: LLM 判断语义边界需要额外 API 调用")
print()

# ==============================================================================
# 策略 5：分块效果对比
# ==============================================================================
print("=== 策略 5: 分块效果对比测试 ===")

# 使用实际检索场景对比不同策略的效果
test_queries = [
    "What is the RAG workflow in LangChain?",
    "How does the Agent system work?",
    "What are the best practices for production deployment?",
]

# 创建不同分块策略的向量库并检索
strategies = {
    "Fixed": fixed_splitter.split_text(LONG_TEXT),
    "Recursive": recursive_splitter.split_text(LONG_TEXT),
    "Markdown": [doc.page_content for doc in md_docs],
    "Semantic": semantic_boundaries,
}

for strategy_name, chunks in strategies.items():
    # 为每个分块创建带元数据的 Document
    docs = [Document(page_content=c, metadata={"source": strategy_name}) for c in chunks]

    # 创建临时向量库
    vs = Chroma.from_documents(
        docs,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=f"test_{strategy_name}",
    )
    retriever = vs.as_retriever(search_kwargs={"k": 1})

    print(f"   [{strategy_name}] 分块检索测试:")
    for query in test_queries:
        retrieved = retriever.invoke(query)
        score = len(retrieved[0].page_content) if retrieved else 0
        relevance = any(kw in retrieved[0].page_content.lower()
                        for kw in query.lower().split()) if retrieved else False
        print(f"     Q: {query[:30]}... → 命中长度: {score} | 相关: {'✅' if relevance else '❌'}")
    print()

    # 清理
    vs.delete_collection()

# ==============================================================================
# 教学备注：分块策略选择指南
# ==============================================================================
# 核心问题：没有"最好的"分块策略，只有"最适合的"策略
#
# 选型决策树：
#
#   文档类型?
#   ├── Markdown/RST/HTML
#   │   └── MarkdownHeaderTextSplitter（保留层级结构）
#   ├── 代码（Python/JS）
#   │   └── Language-specific splitter（按函数/类）
#   ├── 自然语言长文
#   │   ├── 一般场景 → RecursiveCharacterTextSplitter（推荐首选）
#   │   └── 高质量要求 → 语义分块（LLM 判断边界）
#   └── 短文本/FAQ
#       └── 不需要分块（每篇文档就是一个块）
#
#   业务场景要求?
#   ├── 精准检索 → 小块（200-300 token），语义独立
#   ├── 全面上下文 → 大块（500-1000 token），重叠 10-20%
#   ├── 代码补全 → 函数/类级别分割
#   └── 多步推理 → 保留关系（Markdown 层级/图谱）
#
# 关键参数调优:
#
#   chunk_size（块大小）:
#     小（100-200）：精确检索，但上下文不足
#     中（300-500）：平衡方案（推荐）
#     大（800+）：上下文丰富，但信息混杂
#
#   chunk_overlap（重叠大小）:
#     10-20%：标准配置
#     0%：可能丢失边界信息
#     30%+：上下文冗余但安全
#
#   分隔符优先级（separators）:
#     ["\n\n", "\n", ".", "?", "!", " "]：从段落级到单词级
#     自定义：根据文档结构定制
#
# 评估分块质量的方法:
#   1. 检索命中率：检索 Top-1 是否包含答案
#   2. 语义完整性：块内信息是否逻辑完整
#   3. 上下文丢失率：边界处信息是否被截断
#   4. 块数/文档：分块太多增加检索负担
#
# 生产建议:
#   1. 默认方案：RecursiveCharacterTextSplitter（chunk_size=500, overlap=50）
#   2. 先跑通，再优化（不要一开始就追求最优分块）
#   3. 用 A/B 测试验证不同分块策略的效果
#   4. 结构化文档保留 metadata（标题、层级、URL）
