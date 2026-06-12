"""文档加载示例 —— PDF、网页、API、JSON、CSV、目录批量加载。

文档加载是 RAG 管道的第一步。加载质量直接影响后续检索和生成效果。

运行前设置: export OPENAI_API_KEY="sk-..." (如果用到 OpenAI 模型)
"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    JSONLoader,
    CSVLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    GithubFileLoader,
)
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
import json
import os
from pathlib import Path

# ==============================================================================
# 示例 1：加载 PDF 文档
# ==============================================================================
print("=== 示例 1: 加载 PDF ===")

try:
    # PyPDFLoader: 使用 pypdf 解析 PDF
    # pip install pypdf

    # loader = PyPDFLoader("path/to/document.pdf")
    # pages = loader.load()
    # print(f"   加载了 {len(pages)} 页")

    # 注意: PyPDFLoader 每页一个 Document
    # 如果需要按内容分块，后续要用 RecursiveCharacterTextSplitter

    # 替代方案: UnstructuredPdfLoader (更精确的文本提取)
    # from langchain_community.document_loaders import UnstructuredPdfLoader
    # pip install unstructured PyPDF2
    # loader = UnstructuredPdfLoader("path/to/document.pdf")

    print("   PyPDFLoader: 每页一个 Document，适合扫描件少的 PDF")
    print("   UnstructuredPdfLoader: 更精确提取，适合复杂排版 PDF")
    print("   注意: 取消注释上面的代码以实际加载 PDF\n")
except Exception as e:
    print(f"   跳过: {e}\n")


# ==============================================================================
# 示例 2：加载网页内容
# ==============================================================================
print("=== 示例 2: 加载网页 ===")

try:
    # WebBaseLoader: 使用 BeautifulSoup 从网页提取文本
    # pip install beautifulsoup4

    urls = [
        "https://python.langchain.com/docs/get_started/integrations/",
        "https://smith.langchain.com/hub",
    ]

    # 加载单个网页
    # loader = WebBaseLoader(urls[0])
    # docs = loader.load()
    # print(f"   网页标题: {docs[0].metadata.get('title', 'N/A')}")
    # print(f"   文本长度: {len(docs[0].page_content)} 字符")

    # 批量加载多个网页
    # loaders = [WebBaseLoader(url) for url in urls]
    # all_docs = []
    # for loader in loaders:
    #     all_docs.extend(loader.load())

    # 提取特定内容（只取 <article> 标签）
    # loader = WebBaseLoader(
    #     web_path=urls[0],
    #     bs4_kwargs={"attrs": {"class": "markdown"}}  # 按 CSS 选择器过滤
    # )

    print("   特点: 自动提取文本、保留标题/作者/发布日期等元数据")
    print("   注意: 某些网站可能有反爬机制，需要处理")
    print("   注意: 取消注释上面的代码以实际加载\n")
except Exception as e:
    print(f"   跳过: {e}\n")


# ==============================================================================
# 示例 3：加载 JSON 文件
# ==============================================================================
print("=== 示例 3: 加载 JSON ===")

# 创建示例 JSON 文件
sample_json = {
    "products": [
        {"id": 1, "name": "LangChain", "type": "framework", "version": "0.3"},
        {"id": 2, "name": "LangGraph", "type": "framework", "version": "0.2"},
        {"id": 3, "name": "LangSmith", "type": "platform", "version": "1.0"},
    ]
}
json_path = "/tmp/sample_products.json"
with open(json_path, "w") as f:
    json.dump(sample_json, f, indent=2)

try:
    # JSONLoader: 结构化加载 JSON 数据
    # jq_schema 定义如何提取每个文档
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".products[]",
        text_content=False,  # 不使用 LLM 提取文本
    )

    docs = loader.load()
    print(f"   加载了 {len(docs)} 条 JSON 记录:")
    for doc in docs[:2]:
        print(f"   - {doc.page_content[:60]}...")
        print(f"     元数据: {doc.metadata}")
    print()
except ImportError:
    print("   跳过: 需要安装 pip install jq")
    print()


# ==============================================================================
# 示例 4：加载 CSV 文件
# ==============================================================================
print("=== 示例 4: 加载 CSV ===")

# 创建示例 CSV 文件
csv_content = """id,name,category,description
1,LangChain,framework,LLM application framework
2,LangGraph,framework,Stateful multi-actor applications
3,LangSmith,platform,LLM development platform
"""
csv_path = "/tmp/sample_docs.csv"
with open(csv_path, "w") as f:
    f.write(csv_content)

try:
    # CSVLoader: 将 CSV 每行转换为 Document
    # 第一行作为字段名
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()

    print(f"   加载了 {len(docs)} 条 CSV 记录:")
    for doc in docs:
        print(f"   - {doc.page_content[:60]}...")
        print(f"     元数据: {doc.metadata}")
    print()
except ImportError:
    print("   跳过: 需要安装 pip install arxiv (CSV 通常需要额外依赖)")
    print()


# ==============================================================================
# 示例 5：加载纯文本文件
# ==============================================================================
print("=== 示例 5: 加载纯文本 ===")

# 创建示例文本文件
txt_content = """LangChain is a framework for developing applications powered by large language models.

Key components:
- Models
- Prompts
- Chains
- Agents
- Tools
- Memory
- Retrieval
"""
txt_path = "/tmp/sample.txt"
with open(txt_path, "w") as f:
    f.write(txt_content)

try:
    # TextLoader: 最简单的加载器
    loader = TextLoader(txt_path, encoding="utf-8")
    docs = loader.load()
    print(f"   加载了 {len(docs)} 个文档:")
    print(f"   文本长度: {len(docs[0].page_content)} 字符")
    print(f"   元数据: {docs[0].metadata}\n")
except Exception as e:
    print(f"   跳过: {e}\n")


# ==============================================================================
# 示例 6：Markdown 文件加载
# ==============================================================================
print("=== 示例 6: 加载 Markdown ===")

md_content = """# LangChain Documentation

## Overview
LangChain is a framework for LLM applications.

## Components
- Models
- Prompts
- Chains

## Getting Started
Install with: pip install langchain
"""
md_path = "/tmp/sample.md"
with open(md_path, "w") as f:
    f.write(md_content)

try:
    # 方案 1: UnstructuredMarkdownLoader（保留更多格式）
    # from langchain_community.document_loaders import UnstructuredMarkdownLoader
    # loader = UnstructuredMarkdownLoader(md_path)

    # 方案 2: 手动解析
    loader = TextLoader(md_path, encoding="utf-8")
    docs = loader.load()
    print(f"   加载了 Markdown 文件:")
    print(f"   内容预览: {docs[0].page_content[:80]}...")
    print(f"   注意: 对于复杂 Markdown，推荐使用 UnstructuredMarkdownLoader\n")
except Exception as e:
    print(f"   跳过: {e}\n")


# ==============================================================================
# 示例 7：目录批量加载
# ==============================================================================
print("=== 示例 7: 目录批量加载 ===")

# 创建示例目录和文件
docs_dir = "/tmp/sample_docs"
os.makedirs(docs_dir, exist_ok=True)

for filename in ["doc1.txt", "doc2.txt", "doc3.md"]:
    with open(os.path.join(docs_dir, filename), "w") as f:
        f.write(f"Content of {filename}. This is sample text for RAG.\n")

try:
    # DirectoryLoader: 批量加载目录下的文件
    # glob 模式支持通配符

    # 加载所有 .txt 文件
    # loader = DirectoryLoader(
    #     docs_dir,
    #     glob="**/*.txt",
    #     loader_cls=TextLoader,
    #     loader_kwargs={"encoding": "utf-8"},
    # )

    # 同时加载多种格式
    # loader = DirectoryLoader(
    #     docs_dir,
    #     glob="**/*",
    #     use_multithreading=True,  # 并行加载
    #     max_concurrency=4,
    # )

    # 加载后自动分块
    # from langchain_community.document_loaders import DirectoryLoader
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    # loader = DirectoryLoader(
    #     docs_dir,
    #     glob="**/*.txt",
    #     loader_cls=TextLoader,
    #     recursive=True,
    # )
    # docs = loader.load()
    # splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    # chunks = splitter.split_documents(docs)
    # print(f"   加载并分块后: {len(chunks)} 个 chunk")

    print("   特点: 递归加载子目录、支持通配符过滤、并行加载")
    print("   glob 示例:")
    print("   - **/*.txt: 所有 txt 文件")
    print("   - **/*.pdf: 所有 PDF")
    print("   - **/*:   所有文件")
    print("   注意: 取消注释上面的代码以实际加载\n")
except Exception as e:
    print(f"   跳过: {e}\n")


# ==============================================================================
# 示例 8：GitHub 仓库加载
# ==============================================================================
print("=== 示例 8: GitHub 加载 ===")

try:
    # GithubFileLoader: 加载 GitHub 仓库中的文件
    # pip install PyGithub

    # loader = GithubFileLoader(
    #     repo="langchain-ai/langchain",
    #     file="README.md",
    #     github_token="your-token",  # 可选，但有 token 速率限制更高
    # )
    # docs = loader.load()

    # 批量加载多个文件
    # files = ["README.md", "pyproject.toml", "langchain/__init__.py"]
    # loaders = [
    #     GithubFileLoader(repo="langchain-ai/langchain", file=f, github_token="token")
    #     for f in files
    # ]
    # all_docs = [doc for loader in loaders for doc in loader.load()]

    print("   特点: 直接加载 GitHub 仓库文件")
    print("   适用: 代码分析、文档提取")
    print("   注意: 取消注释上面的代码以实际使用\n")
except ImportError:
    print("   跳过: 需要安装 pip install PyGithub\n")


# ==============================================================================
# 示例 9：文档加载后的分块策略
# ==============================================================================
print("=== 示例 9: 分块策略 ===")

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
)

sample_text = """
LangChain is a framework for developing LLM applications.

## Key Components
Models, Prompts, Chains, Agents, Tools, Memory, Retrieval

LangGraph adds orchestration for complex workflows.

LangSmith provides tracing and evaluation.

LangServe enables API deployment.
"""

# 策略 1: 按字符分割（基础）
char_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
char_chunks = char_splitter.split_text(sample_text)
print(f"1. Character 分割: {len(char_chunks)} 个 chunk")

# 策略 2: 递归字符分割（推荐，尽量保持语义完整）
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(sample_text)
print(f"2. Recursive 分割: {len(recursive_chunks)} 个 chunk")

# 策略 3: 按 Token 分割（精确控制 token 数）
token_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=5)
token_chunks = token_splitter.split_text(sample_text)
print(f"3. Token 分割: {len(token_chunks)} 个 chunk")

# 策略 4: Markdown 按标题分割（保留文档结构）
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header_Level_1"),
        ("##", "Header_Level_2"),
    ]
)
md_chunks = md_splitter.split_text(sample_text)
print(f"4. Markdown 标题分割: {len(md_chunks)} 个 chunk")
for chunk in md_chunks:
    print(f"   - {chunk.metadata} | {chunk.page_content[:50]}...")

# 策略 5: Python 代码按函数/类分割
code = """
def function_one():
    return 1

class MyClass:
    def method_one(self):
        pass

def function_two():
    return 2
"""
code_splitter = PythonCodeTextSplitter(chunk_size=200, chunk_overlap=0)
code_chunks = code_splitter.split_text(code)
print(f"\n5. Python 代码分割: {len(code_chunks)} 个 chunk")
print("   特点: 按函数定义和类定义分割，保持代码结构完整\n")


# ==============================================================================
# 教学备注：文档加载 —— RAG 管道的第一站
# ==============================================================================
# 核心问题：为什么不能直接用原始文件？为什么需要 Document 和分块？
#   1. LLM 有上下文窗口限制（通常 128K token，但实际检索时不会用满）
#   2. 向量嵌入有长度限制（太长嵌入质量下降）
#   3. 检索需要"按内容查找"，而不是检索整个文件
#
# Document 对象的结构:
#   Document(page_content="文本内容", metadata={"source": "file.pdf", "page": 1, ...})
#   metadata 是关键——它记录了来源信息，用于引用溯源和过滤
#
# 加载器分类:
#   文件类: PyPDFLoader, TextLoader, CSVLoader, JSONLoader, Unstructured*Loader
#   网络类: WebBaseLoader, GithubFileLoader
#   批量类: DirectoryLoader（递归加载整个目录）
#
# 分块策略选型:
#   普通文本: RecursiveCharacterTextSplitter（默认推荐）
#   精确 token 控制: TokenTextSplitter
#   Markdown: MarkdownHeaderTextSplitter（保留标题层级）
#   代码文件: PythonCodeTextSplitter（按函数/类分割）
#   PDF: 先用 PyPDFLoader 加载，再按段落分割
#
# 分块最佳实践:
#   1. chunk_size: 普通文本 150-300 字符，代码 500-1000 字符
#   2. chunk_overlap: 10-20%，保持上下文连续性
#   3. 不要太小（丢失语义）也不要太大（检索不精确）
#   4. 保留元数据（source, page, title）用于引用
#
# 教学建议顺序:
#   1. 先理解"为什么需要加载和分块"
#   2. 跑通 TextLoader + RecursiveCharacterTextSplitter（最常用组合）
#   3. 再了解其他加载器（PDF、网页、JSON、CSV）
#   4. 最后理解分块策略的选择
