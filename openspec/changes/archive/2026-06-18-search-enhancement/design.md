# 搜索增强功能 - 设计文档

## 数据模型

```python
class UnifiedDocument(BaseModel):
    id: str
    title: str
    content: str          # 文档正文 / 问答内容
    doc_type: str         # wiki | pdf | word | text | faq | app_link
    source: str           # 来源路径/URL
    tags: list[str]       # 标签（关联推荐核心）
    url: str | None       # 跳转链接
```

## 架构

```
用户查询 → 混合检索（向量语义 + BM25 关键词） → RRF 融合排序
              ↓
        Top-K 搜索结果
              ↓
       提取标签 → 关联推荐（同标签文档/FAQ/链接）
              ↓
       输出：搜索结果 + 关联推荐
```

## 核心组件

| 组件 | 职责 | 技术 |
|------|------|------|
| `HybridRetriever` | BM25 + 向量检索 + RRF 融合 | ChromaDB + rank_bm25 |
| `TagRecommender` | 标签提取 → 关联条目召回 | 标签交集 + 向量相似度 |
| `SearchEngine` | 编排检索和推荐流程 | LangChain Runnable |

## 数据流

1. 数据加载 → ChromaDB 向量索引 + BM25 内存索引
2. 用户查询 → BM25 + 向量并行检索 → RRF 排序
3. 标签提取 → 同标签关联推荐
4. 返回结构化结果

## 错误处理

- BM25 缺失 → 降级纯向量搜索
- 向量库故障 → 降级纯 BM25
- 标签无匹配 → 跳过推荐

## 技术栈

- LangChain >=0.3, ChromaDB, rank_bm25
- Ollama + nomic-embed-text（嵌入）/ llama3.2:1b（LLM）
- Pydantic 数据模型
