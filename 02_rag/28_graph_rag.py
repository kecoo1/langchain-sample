"""知识图谱 RAG (GraphRAG) —— 用实体关系提升检索精度。

传统 RAG 的局限：检索时只考虑"语义相似度"，不理解实体之间的关系。
GraphRAG 通过知识图谱捕获实体关系，回答需要多步推理的问题时效果显著。

本文件覆盖 3 种 GraphRAG 实现方案：
1. 用 LLM 从文档中提取实体关系图
2. 用图谱增强检索（检索 + 关系扩展）
3. 社区检测 + 摘要（Microsoft GraphRAG 简化版）
"""

import os
from typing import Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 准备：示例文档
# ==============================================================================
SAMPLE_DOCS = [
    Document(page_content="Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company is headquartered in Cupertino, California. Apple is known for iPhone, iPad, and Mac computers."),
    Document(page_content="Tim Cook is the CEO of Apple since 2011. He succeeded Steve Jobs. Under Cook, Apple launched Apple Watch, AirPods, and the Apple Silicon M1 chip."),
    Document(page_content="OpenAI was founded in 2015 by Sam Altman, Elon Musk, Greg Brockman, and others. It's an AI research organization based in San Francisco. GPT-4 is their most famous product."),
    Document(page_content="Sam Altman is the CEO of OpenAI. He previously was president of Y Combinator. Elon Musk left OpenAI's board in 2018 but later founded xAI, which created Grok."),
    Document(page_content="Microsoft invested $10 billion in OpenAI in 2023. Azure provides cloud computing for OpenAI's models. Microsoft Copilot uses OpenAI's GPT-4 technology."),
    Document(page_content="Satya Nadella is the CEO of Microsoft since 2014. Microsoft is headquartered in Redmond, Washington. Their products include Windows, Azure, and Office 365."),
]

# ==============================================================================
# 示例 1：LLM 提取实体关系图
# ==============================================================================
print("=== 示例 1: LLM 提取实体关系图 ===")


class Entity(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type: Person/Organization/Product/Location")


class Relation(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: str = Field(description="Relation type (founded/CEO/headquartered/invested/...")


class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(description="Entities found in text")
    relations: list[Relation] = Field(description="Relations between entities")


# 从文档中提取知识图谱
extractor = model.with_structured_output(KnowledgeGraph)

all_entities = {}
all_relations = []

for doc in SAMPLE_DOCS:
    kg = extractor.invoke([
        SystemMessage(content="Extract entities and their relationships from the text."),
        HumanMessage(content=doc.page_content),
    ])
    for entity in kg.entities:
        key = (entity.name, entity.type)
        if key not in all_entities:
            all_entities[key] = entity
    all_relations.extend(kg.relations)

print(f"1. 从 {len(SAMPLE_DOCS)} 篇文档提取的图谱:")
print(f"   实体数: {len(all_entities)}")
for (name, etype), entity in all_entities.items():
    print(f"   - {name} ({etype})")

print(f"\n   关系数: {len(all_relations)}")
for r in all_relations[:8]:
    print(f"   {r.source} --[{r.relation}]--> {r.target}")
print()

# ==============================================================================
# 示例 2：图谱增强检索（检索 + 关系扩展）
# ==============================================================================
print("=== 示例 2: 图谱增强检索 ===")

# 传统 RAG 只检索相似的文档，
# 图谱增强检索：检索到实体 → 沿着关系扩展 → 找到间接相关的信息


class GraphRetriever:
    """基于知识图谱的增强检索器。"""

    def __init__(self, entities: dict, relations: list):
        self.entities = entities
        self.relations = relations
        # 构建邻接表
        self.adjacency = {}
        for r in relations:
            if r.source not in self.adjacency:
                self.adjacency[r.source] = []
            self.adjacency[r.source].append((r.target, r.relation))
            if r.target not in self.adjacency:
                self.adjacency[r.target] = []
            self.adjacency[r.target].append((r.source, r.relation))

    def expand_entity(self, entity_name: str, depth: int = 1) -> set:
        """从实体出发，沿关系扩展到 depth 层。"""
        related = {entity_name}
        current = {entity_name}
        for _ in range(depth):
            next_set = set()
            for name in current:
                if name in self.adjacency:
                    for target, _ in self.adjacency[name]:
                        next_set.add(target)
            current = next_set - related
            related.update(current)
        return related

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """检索与查询相关的实体及扩展实体。"""
        # 用 LLM 提取查询中的实体
        query_entities = extractor.invoke([
            SystemMessage(content="Extract entities from this query."),
            HumanMessage(content=query),
        ])

        # 收集所有相关实体
        related_entities = set()
        for entity in query_entities.entities:
            expanded = self.expand_entity(entity.name, depth=1)
            related_entities.update(expanded)

        # 根据相关实体的文档构建结果
        results = []
        for doc in SAMPLE_DOCS:
            for entity_name in related_entities:
                if entity_name.lower() in doc.page_content.lower():
                    results.append(doc.page_content)
                    break

        return results[:top_k]


graph_retriever = GraphRetriever(all_entities, all_relations)

# 测试：查询 OpenAI 的 CEO
query = "Who leads OpenAI and what company invested in them?"
results = graph_retriever.retrieve(query)
print(f"2. 图谱增强检索:")
print(f"   查询: '{query}'")
print(f"   检索到 {len(results)} 篇相关文档:")
for i, r in enumerate(results, 1):
    print(f"   {i}. {r[:80]}...")

# 对比：如果只用关键词检索
keyword_results = [doc.page_content for doc in SAMPLE_DOCS if "OpenAI" in doc.page_content]
print(f"\n   纯关键词检索: {len(keyword_results)} 篇")
print(f"   图谱增强额外找到: {len(results) - len(keyword_results)} 篇关联文档")
print()

print("   图谱增强检索的优势：")
print("   - 传统检索：只找到包含 'OpenAI' 的文档")
print("   - 图谱增强：还找到 Microsoft + Azure + Sam Altman 的文档")
print('   - 回答 "谁投资了 OpenAI" 时，这两篇都提供了有用信息')
print()


# ==============================================================================
# 示例 3：社区检测 + 摘要（简版 GraphRAG）
# ==============================================================================
print("=== 示例 3: 社区检测 + 摘要 ===")

# Microsoft GraphRAG 的核心创新：
#   将关联紧密的实体聚类成"社区" → 对每个社区生成摘要 →
#   回答问题时：搜索社区摘要（宏观）+ 搜索原始文档（微观）

# 简单的社区检测：根据关系密度聚类
communities = {
    "Apple Ecosystem": {
        "entities": ["Apple Inc.", "Steve Jobs", "Tim Cook", "Cupertino", "iPhone", "Apple Watch", "Apple Silicon M1"],
        "summary": "",
    },
    "AI Industry": {
        "entities": ["OpenAI", "Sam Altman", "Elon Musk", "GPT-4", "Microsoft", "Azure", "Satya Nadella"],
        "summary": "",
    },
}

# 对每个社区生成摘要
for community_name, community_data in communities.items():
    # 收集该社区的文档
    community_docs = [
        doc.page_content for doc in SAMPLE_DOCS
        if any(e.lower() in doc.page_content.lower() for e in community_data["entities"])
    ]
    combined_text = "\n\n".join(community_docs)

    # 生成社区摘要
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the key information about these entities and their relationships."),
        ("human", combined_text),
    ])
    summary = (summary_prompt | model | StrOutputParser()).invoke({})
    community_data["summary"] = summary
    print(f"3. 社区 [{community_name}] 摘要:")
    print(f"   {summary[:150]}...\n")


# GraphRAG 查询：先搜索社区摘要，再搜索原始文档
def graphrag_query(query: str) -> str:
    """GraphRAG 风格的双层检索。"""
    # Step 1: 找最相关的社区
    community_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the question, which community summary is most relevant? Pick one."),
        ("human", f"Question: {query}\n\nCommunities:\n" + "\n".join(
            f"- {name}: {data['summary'][:200]}"
            for name, data in communities.items()
        )),
    ])
    best_community = (community_prompt | model | StrOutputParser()).invoke({})

    # Step 2: 基于社区 + 原始文档回答
    relevant_docs = [
        doc.page_content for doc in SAMPLE_DOCS
        if any(e.lower() in doc.page_content.lower()
               for e in communities.get("AI Industry", {}).get("entities", []))
        if "AI" in best_community or "Apple" in best_community
    ]

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer based on the provided community summaries and documents.
Cite which entities and relationships support your answer."""),
        ("human", f"Question: {query}\n\nContext:\n{chr(10).join(relevant_docs[:3])}"),
    ])
    answer = (answer_prompt | model | StrOutputParser()).invoke({})
    return answer


query = "Who are the key people in the AI industry and how are they connected?"
answer = graphrag_query(query)
print(f"   GraphRAG 查询结果:")
print(f"   Q: {query}")
print(f"   A: {answer[:300]}...\n")

print("   GraphRAG 核心概念:")
print("   - 社区检测：将紧密关联的实体聚为社区")
print("   - 社区摘要：每个社区生成摘要（宏观知识）")
print("   - 双层检索：先搜社区摘要（快速定位），再搜原始文档（精确回答）")
print("   - 关系推理：利用实体关系回答需要多步推理的问题")
print()


# ==============================================================================
# 教学备注：GraphRAG —— 让 RAG 理解"关系"
# ==============================================================================
# 核心问题：传统 RAG 基于向量相似度，无法理解实体间的"关系"
#   问"Tim Cook 接替了谁？"→ 传统 RAG 需要文档同时提到 Tim Cook 和 Steve Jobs
#   问"微软投资的 AI 公司 CEO 是谁？"→ 需要多步推理（微软→OpenAI→Sam Altman）
#
# RAG 技术的演进路线：
#
#   朴素 RAG（02_rag/08_rag.py）
#     → 文档 → 分块 → 向量化 → 检索 → 回答
#     → 限制：只做语义匹配，不理解实体关系
#
#   进阶 RAG（02_rag/15_rag_advanced.py）
#     → 混合检索 + 重排序 + HyDE + 多查询
#     → 改进：检索精度大幅提升
#     → 限制：仍然不理解关系
#
#   GraphRAG（本文件）
#     → 文档 → 实体提取 → 关系构建 → 社区检测 → 双层检索
#     → 突破：理解实体关系，能回答多步推理问题
#
#   三种 GraphRAG 实现方案对比：
#
#   | 方案 | 实现复杂度 | 检索精度 | 适用场景 |
#   |------|-----------|---------|---------|
#   | LLM 提取（示例1） | 低 | 中 | 快速原型，小规模文档 |
#   | 关系扩展（示例2） | 中 | 高 | 需要"间接关联"的场景 |
#   | 社区摘要（示例3） | 高 | 最高 | 大规模文档库，企业知识库 |
#
# 何时使用 GraphRAG：
#   - 问题涉及多步推理（A→B→C）
#   - 实体关系对答案有影响
#   - 文档分散，单一文档信息不全
#   - 需要引用性强的答案（"根据...和...的关系"）
#
# 生产建议：
#   - 小规模（<1000 文档）：用 LLM 提取，简单直接
#   - 中规模（<10万）：用 Neo4j 或 NetworkX 存储图谱
#   - 大规模（>10万）：考虑 Microsoft GraphRAG 完整实现
#   - 可配合向量检索使用：向量搜索 Top-K → 图谱扩展 → 排序
