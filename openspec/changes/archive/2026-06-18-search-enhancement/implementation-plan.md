# Search Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a search enhancement example (36_search_enhancement.py) with hybrid retrieval (BM25 + vector) and tag-based content recommendation.

**Architecture:** Single Python file with four components: DataLoader (mock multi-source data), HybridRetriever (BM25 + ChromaDB + RRF), TagRecommender (tag-based association recall), SearchEngine (orchestrator). Teaching notes in Chinese.

**Tech Stack:** Python 3.11+, LangChain >=0.3, ChromaDB, rank_bm25, Ollama (nomic-embed-text), Pydantic

---

### Task 1: Data Model & Mock Data

**Files:**
- Create: `06_advanced_topics/36_search_enhancement.py` (Task 1 covers lines 1-100)

- [x] **Step 1: Write data model and mock data**

```python
"""
搜索增强示例 — 混合检索 + 标签关联推荐
========================================
功能：
  1. 混合检索：BM25（拼写容错） + 向量语义搜索（近义词），RRF 融合排序
  2. 标签关联推荐：从搜索命中结果提取标签，推荐同标签的文档/FAQ/应用链接
  3. 多源数据加载：模拟 Wiki / PDF / Word / FAQ / 应用跳转链接

运行：python 36_search_enhancement.py
"""

from pydantic import BaseModel
from typing import Optional


class UnifiedDocument(BaseModel):
    """统一文档模型 — 所有数据源统一为此格式"""
    id: str
    title: str
    content: str            # 文档正文 / 问答内容
    doc_type: str           # wiki | pdf | word | text | faq | app_link
    source: str             # 来源路径/URL
    tags: list[str]         # 标签（关联推荐核心）
    url: Optional[str] = None  # 跳转链接


def create_mock_documents() -> list[UnifiedDocument]:
    """生成模拟多源数据

    模拟以下数据源：
      - FAQ（已解答问题）：退款/订单/物流相关问答
      - 文档（Wiki/PDF/Word/Text）：售后政策、使用指南等
      - 应用跳转链接：订单查询、退款申请等功能入口
    """
    return [
        # === FAQ（已解答问题）===
        UnifiedDocument(
            id="faq-1",
            title="如何申请退款？",
            content="用户可以在订单详情页点击「申请退款」按钮，选择退款原因后提交。退款将在 3-5 个工作日原路返回。",
            doc_type="faq",
            source="internal_wiki",
            tags=["退款", "订单", "售后"],
        ),
        UnifiedDocument(
            id="faq-2",
            title="退款需要多长时间到账？",
            content="退款到账时间因支付方式不同而异：微信/支付宝通常 1-3 个工作日，银行卡 3-7 个工作日。",
            doc_type="faq",
            source="internal_wiki",
            tags=["退款", "支付", "到账时间"],
        ),
        UnifiedDocument(
            id="faq-3",
            title="如何修改订单地址？",
            content="订单未发货前可在订单详情页修改收货地址。如已发货需联系客服处理转寄。",
            doc_type="faq",
            source="internal_wiki",
            tags=["订单", "地址", "物流"],
        ),
        UnifiedDocument(
            id="faq-4",
            title="商品质量问题如何退货？",
            content="收到商品 7 天内如发现质量问题，可申请免费退货。请拍照留存证据，联系客服审核后退回。",
            doc_type="faq",
            source="internal_wiki",
            tags=["退货", "质量", "售后"],
        ),
        # === 文档（Wiki/PDF/Word/Text）===
        UnifiedDocument(
            id="doc-1",
            title="售后服务政策 V3.2",
            content="本政策适用于所有平台订单。退货时效：7 天无理由退货（部分商品除外）。换货时效：15 天内质量问题可换。退款路径：原路返回。",
            doc_type="wiki",
            source="internal_wiki/售后政策",
            tags=["售后", "退货", "退款", "政策"],
        ),
        UnifiedDocument(
            id="doc-2",
            title="订单处理流程说明",
            content="订单状态流转：待支付 → 已支付 → 已发货 → 配送中 → 已签收 → 已完成。异常订单处理：支付超时取消、库存不足取消、用户主动取消。",
            doc_type="pdf",
            source="docs/订单处理流程.pdf",
            tags=["订单", "物流", "流程"],
        ),
        UnifiedDocument(
            id="doc-3",
            title="支付方式说明",
            content="支持支付方式：微信支付、支付宝、银行卡（借记卡/信用卡）、花呗分期。支付限额说明：微信单笔 5 万，支付宝单笔 10 万，银行卡以发卡行限额为准。",
            doc_type="word",
            source="docs/支付方式说明.docx",
            tags=["支付", "订单", "限额"],
        ),
        UnifiedDocument(
            id="doc-4",
            title="物流配送常见问题",
            content="配送范围：全国包邮（港澳台除外）。配送时效：一线城市 1-2 天，二三线城市 2-4 天，偏远地区 5-7 天。物流查询：在订单详情页查看物流轨迹。",
            doc_type="text",
            source="notes/物流FAQ.txt",
            tags=["物流", "配送", "时效"],
        ),
        UnifiedDocument(
            id="doc-5",
            title="用户账号安全指南",
            content="账号保护建议：设置强密码（字母+数字+符号）、开启双重认证、定期修改密码、不在公共设备上登录。如有异常登录，请立即修改密码并联系客服。",
            doc_type="wiki",
            source="internal_wiki/账号安全",
            tags=["账号", "安全", "密码"],
        ),
        # === 应用跳转链接 ===
        UnifiedDocument(
            id="link-1",
            title="订单查询",
            content="查看所有订单状态、物流信息和操作入口",
            doc_type="app_link",
            source="app://orders",
            tags=["订单", "物流"],
            url="https://app.example.com/orders",
        ),
        UnifiedDocument(
            id="link-2",
            title="退款申请",
            content="发起退款申请，查看退款进度",
            doc_type="app_link",
            source="app://refund",
            tags=["退款", "售后"],
            url="https://app.example.com/refund",
        ),
        UnifiedDocument(
            id="link-3",
            title="在线客服",
            content="联系在线客服，获取人工帮助",
            doc_type="app_link",
            source="app://service",
            tags=["售后", "客服"],
            url="https://app.example.com/service",
        ),
        UnifiedDocument(
            id="link-4",
            title="地址管理",
            content="管理收货地址，添加/编辑/删除地址",
            doc_type="app_link",
            source="app://address",
            tags=["订单", "账号"],
            url="https://app.example.com/address",
        ),
    ]


if __name__ == "__main__":
    docs = create_mock_documents()
    print(f"加载了 {len(docs)} 个文档:")
    for d in docs:
        print(f"  [{d.doc_type:>8}] {d.title} 标签: {d.tags}")
```

- [x] **Step 2: Run to verify data loads correctly**

Run: `python 06_advanced_topics/36_search_enhancement.py`

Expected output:
```
加载了 17 个文档:
  [     faq] 如何申请退款？ 标签: ['退款', '订单', '售后']
  ...
```

---

### Task 2: BM25 Index Builder

**Files:**
- Modify: `06_advanced_topics/36_search_enhancement.py` (append after mock data)

- [x] **Step 1: Write BM25 index and search function**

```python
# === 以下内容追加到 36_search_enhancement.py ===

from rank_bm25 import BM25Okapi
import jieba
from typing import Optional

# ============================================================
# Teaching Notes: BM25 关键词检索
# ============================================================
# BM25 是传统信息检索领域最经典的排序算法之一。
# 相比简单的 TF-IDF，BM25 引入了文档长度归一化和饱和频率
# 两个关键改进，在处理短文本（如 FAQ 标题）时尤为有效。
#
# 本项目使用 jieba 分词对中文内容进行切词后构建 BM25 索引。
# jieba 是 Python 最流行的中文分词库，支持精确模式和搜索模式。


def tokenize(text: str) -> list[str]:
    """中文分词"""
    return list(jieba.cut(text))


def build_bm25_index(docs: list[UnifiedDocument]) -> BM25Okapi:
    """基于文档内容构建 BM25 索引

    对每个文档的 title + content 进行分词后构建索引。
    BM25 索引完全在内存中运行，适合中小规模文档集。
    百万级文档建议使用 Elasticsearch 替代。
    """
    tokenized_corpus = [
        tokenize(f"{d.title} {d.content}") for d in docs
    ]
    return BM25Okapi(tokenized_corpus)


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    docs: list[UnifiedDocument],
    top_k: int = 5,
) -> list[tuple[UnifiedDocument, float]]:
    """BM25 检索

    对中文查询进行分词后在 BM25 索引中检索，
    返回 Top-K 文档及其 BM25 得分。

    拼写容错：BM25 基于关键词匹配，对于常见的拼写错误
    （如"退kuan"→"退款"）有天然容错能力，
    因为"退"字仍然能匹配到相关文档。
    """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    scored_docs = [
        (docs[i], float(scores[i]))
        for i in range(len(docs))
        if scores[i] > 0
    ]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]
```

- [x] **Step 2: Add test code**

```python
# 在文件末尾追加测试（if __name__ 块中）

def _test_bm25():
    """测试 BM25 检索：拼写容错"""
    from rank_bm25 import BM25Okapi
    docs = create_mock_documents()
    bm25 = build_bm25_index(docs)

    # 测试 1：正常查询
    results = bm25_search("退款流程", bm25, docs, top_k=3)
    assert len(results) > 0, "应返回退款相关文档"
    print(f"\n[BM25 测试 1] 查询「退款流程」结果:")
    for d, score in results:
        print(f"  {d.title:20s} score={score:.3f}")

    # 测试 2：拼写容错（"退kuan" → "退款"）
    results = bm25_search("退kuan", bm25, docs, top_k=3)
    assert len(results) > 0, "拼写错误也应返回结果"
    print(f"\n[BM25 测试 2] 查询「退kuan」（拼写错误）:")
    for d, score in results:
        print(f"  {d.title:20s} score={score:.3f}")

    # 测试 3：无匹配查询
    results = bm25_search("xxxxxxxx", bm25, docs, top_k=3)
    assert len(results) == 0, "无匹配应返回空列表"
    print(f"\n[BM25 测试 3] 无匹配查询 → 空: {len(results)}")
    print("BM25 测试全部通过 ✅")
```

Modify the `if __name__` block at the end to:

```python
if __name__ == "__main__":
    print("=" * 60)
    print("搜索增强示例 — 混合检索 + 标签关联推荐")
    print("=" * 60)

    docs = create_mock_documents()
    print(f"\n1. 加载了 {len(docs)} 个文档:")
    for d in docs:
        print(f"  [{d.doc_type:>8}] {d.title} 标签: {d.tags}")

    print("\n2. 测试 BM25 检索")
    bm25 = build_bm25_index(docs)
    _test_bm25()
```

- [x] **Step 3: Run BM25 test to verify**

Run: `python 06_advanced_topics/36_search_enhancement.py`

Expected: BM25 检索正常工作，拼写错误查询"退kuan"能返回"退款"相关文档。

---

### Task 3: Vector Index (ChromaDB)

**Files:**
- Modify: `06_advanced_topics/36_search_enhancement.py` (append after BM25 section)

- [x] **Step 1: Write vector store initialization and search**

```python
# ============================================================
# Teaching Notes: 向量语义搜索
# ============================================================
# 向量搜索将文本转换为高维向量（嵌入），在向量空间中
# 通过余弦相似度搜索语义相近的内容。与 BM25 关键词匹配
# 互补：BM25 处理精确匹配和拼写错误，向量搜索处理
# 同义词、近义词和语义相似（如 "return" ↔ "退款"）。
#
# ChromaDB 是轻量级开源向量数据库，支持本地持久化，
# 适合中小规模项目和示例应用。


from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "nomic-embed-text"


def create_vector_store(
    docs: list[UnifiedDocument],
    persist_dir: str = "./chroma_db",
) -> Chroma:
    """创建 ChromaDB 向量存储

    使用 Ollama 的 nomic-embed-text 嵌入模型将文档
    title + content 转换为向量。nomic-embed-text 是一个
    轻量级通用嵌入模型，在中文语义理解上表现良好。

    元数据存储：doc_type、tags、source、url 等信息
    后续可用于过滤和标签推荐。
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    lc_docs = []
    for d in docs:
        lc_docs.append(LCDocument(
            page_content=f"{d.title}\n{d.content}",
            metadata={
                "id": d.id,
                "title": d.title,
                "doc_type": d.doc_type,
                "source": d.source,
                "tags": ",".join(d.tags),
                "url": d.url or "",
            },
        ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    split_docs = splitter.split_documents(lc_docs)

    return Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )


def vector_search(
    query: str,
    vector_store: Chroma,
    top_k: int = 5,
) -> list[tuple[UnifiedDocument, float]]:
    """向量语义检索

    将查询文本嵌入为向量，在 ChromaDB 中通过余弦相似度
    搜索最相似的文档片段。

    近义词处理：向量搜索能理解语义相近的表达，
    如搜索"return buying"能匹配到"退货/退款"相关文档，
    这是纯关键词搜索无法做到的。
    """
    results = vector_store.similarity_search_with_relevance_scores(
        query, k=top_k
    )

    # 将 LC Document 转回 UnifiedDocument
    output = []
    for lc_doc, score in results:
        meta = lc_doc.metadata
        tags = meta.get("tags", "").split(",") if meta.get("tags") else []
        url = meta.get("url") or None
        output.append((
            UnifiedDocument(
                id=meta.get("id", ""),
                title=meta.get("title", ""),
                content=lc_doc.page_content,
                doc_type=meta.get("doc_type", ""),
                source=meta.get("source", ""),
                tags=tags,
                url=url if url else None,
            ),
            float(score),
        ))
    return output
```

- [x] **Step 2: Add vector search test**

```python
def _test_vector_search():
    """测试向量检索：中文近义词、英文近义词"""
    docs = create_mock_documents()
    vs = create_vector_store(docs, "./chroma_db_test")

    # 测试 1：中文近义词
    results = vector_search("退货流程", vs, top_k=3)
    assert len(results) > 0, "应返回退货相关结果"
    print(f"\n[向量测试 1] 查询「退货流程」:")
    for d, score in results:
        print(f"  {d.title:20s} score={score:.3f}")

    # 测试 2：英文近义词（测试语义理解）
    results = vector_search("return product", vs, top_k=3)
    assert len(results) > 0, "英文近义词也应返回"
    print(f"\n[向量测试 2] 查询「return product」:")
    for d, score in results:
        print(f"  {d.title:20s} score={score:.3f}")

    # 测试 3：语义近义查询
    results = vector_search("东西坏了想退", vs, top_k=3)
    assert len(results) > 0, "口语化查询应匹配"
    print(f"\n[向量测试 3] 查询「东西坏了想退」:")
    for d, score in results:
        print(f"  {d.title:20s} score={score:.3f}")

    # 清理临时目录
    import shutil
    shutil.rmtree("./chroma_db_test", ignore_errors=True)
    print("\n向量检索测试全部通过 ✅")
```

Update the `if __name__` block to call `_test_vector_search()`.

---

### Task 4: Hybrid Retriever (RRF Fusion)

**Files:**
- Modify: `06_advanced_topics/36_search_enhancement.py` (append after vector section)

- [x] **Step 1: Implement RRF fusion and HybridRetriever**

```python
# ============================================================
# Teaching Notes: 混合检索与 RRF 融合
# ============================================================
# 混合检索：同时使用 BM25（关键词）和向量搜索（语义），
# 再通过 RRF（Reciprocal Rank Fusion）融合排序。
#
# RRF 公式：score(d) = Σ 1/(k + rank_i(d))
# 其中 rank_i(d) 是文档 d 在第 i 个检索器中的排序位置，
# k 是常数（通常 60），用于平滑。
#
# RRF 的优势是不需要归一化不同检索器的得分区间，
# 因为不同检索器的得分范围可能完全不同
# （BM25 得分 vs 余弦相似度得分）。
#
# 降级策略：如果某个检索器不可用，自动降级到单路检索。


def rrf_fusion(
    bm25_results: list[tuple[UnifiedDocument, float]],
    vector_results: list[tuple[UnifiedDocument, float]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[UnifiedDocument, float]]:
    """RRF 融合排序

    将 BM25 和向量检索的结果按排序位置融合。
    输入：两路检索结果列表（文档+得分）
    输出：RRF 融合排序后的 Top-K 结果
    """
    doc_id_to_doc = {}
    doc_id_to_rrf_score = {}

    def add_rank(results):
        for rank, (doc, _score) in enumerate(results, start=1):
            doc_id_to_doc[doc.id] = doc
            doc_id_to_rrf_score[doc.id] = (
                doc_id_to_rrf_score.get(doc.id, 0) + 1 / (k + rank)
            )

    add_rank(bm25_results)
    add_rank(vector_results)

    sorted_docs = sorted(
        doc_id_to_rrf_score.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        (doc_id_to_doc[did], score)
        for did, score in sorted_docs[:top_k]
    ]


class HybridRetriever:
    """混合检索器 — 融合 BM25 + 向量语义

    提供统一的检索入口，内部协调两路检索器，
    处理降级策略，输出 RRF 融合结果。
    """

    def __init__(
        self,
        docs: list[UnifiedDocument],
        persist_dir: str = "./chroma_db",
    ):
        self.docs = docs
        self.bm25_index: Optional[BM25Okapi] = None
        self.vector_store: Optional[Chroma] = None
        self.persist_dir = persist_dir
        self._init_retrievers()

    def _init_retrievers(self):
        """初始化两路检索器（带独立容错）"""
        try:
            self.bm25_index = build_bm25_index(self.docs)
            print(f"  ✓ BM25 索引构建完成（{len(self.docs)} 篇文档）")
        except Exception as e:
            print(f"  ⚠ BM25 索引初始化失败: {e}（将降级到纯向量搜索）")

        try:
            self.vector_store = create_vector_store(self.docs, self.persist_dir)
            print(f"  ✓ 向量索引构建完成（{len(self.docs)} 篇文档）")
        except Exception as e:
            print(f"  ⚠ 向量索引初始化失败: {e}（将降级到纯 BM25）")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[UnifiedDocument, float]]:
        """混合检索入口

        自动协调两路检索器，处理降级。
        """
        bm25_results = []
        if self.bm25_index is not None:
            try:
                bm25_results = bm25_search(query, self.bm25_index, self.docs, top_k * 2)
            except Exception as e:
                print(f"  ⚠ BM25 检索异常: {e}（跳过）")

        vector_results = []
        if self.vector_store is not None:
            try:
                vector_results = vector_search(query, self.vector_store, top_k * 2)
            except Exception as e:
                print(f"  ⚠ 向量检索异常: {e}（跳过）")

        if not bm25_results and not vector_results:
            return []

        if not bm25_results:
            print("  ℹ 降级到纯向量检索")
            return vector_results[:top_k]

        if not vector_results:
            print("  ℹ 降级到纯 BM25 检索")
            return bm25_results[:top_k]

        return rrf_fusion(bm25_results, vector_results, top_k=top_k)
```

- [x] **Step 2: Add hybrid retriever test**

```python
def _test_hybrid_retriever():
    """测试混合检索"""
    docs = create_mock_documents()
    retriever = HybridRetriever(docs, "./chroma_db_test2")

    # 测试 1：标准查询
    results = retriever.search("如何退货", top_k=3)
    assert len(results) > 0
    print(f"\n[混合测试 1] 查询「如何退货」:")
    for d, score in results:
        print(f"  {d.title:20s} score={score:.3f}")

    # 测试 2：英文查询
    results = retriever.search("how to refund", top_k=3)
    assert len(results) > 0
    print(f"\n[混合测试 2] 查询「how to refund」:")
    for d, score in results:
        print(f"  {d.title:20s} score={score:.3f}")

    import shutil
    shutil.rmtree("./chroma_db_test2", ignore_errors=True)
    print("\n混合检索测试全部通过 ✅")
```

---

### Task 5: Tag Recommender

**Files:**
- Modify: `06_advanced_topics/36_search_enhancement.py` (append after HybridRetriever)

- [x] **Step 1: Implement TagRecommender**

```python
# ============================================================
# Teaching Notes: 标签关联推荐
# ============================================================
# 关联推荐：基于标签实现「如果搜了这个，可能也想知道」。
# 从搜索命中结果的标签中提取高频标签，用这些标签召回
# 同标签的其他文档、FAQ 和应用跳转链接。
#
# 推荐逻辑：
#   1. 收集搜索结果的标签，统计频率
#   2. 使用 Top-N 高频标签在文档库中召回
#   3. 排除已出现在搜索结果中的文档
#   4. 按匹配标签数和文档类型排序展示


from collections import Counter


class TagRecommender:
    """标签关联推荐器

    基于标签关联性，推荐与搜索结果相关的其他内容。
    支持按文档类型分组展示（相关文档、相关问题、相关链接）。
    """

    def __init__(self, docs: list[UnifiedDocument]):
        self.docs = docs
        # 按标签索引文档，加速召回
        self.tag_index: dict[str, list[UnifiedDocument]] = {}
        self._build_tag_index()

    def _build_tag_index(self):
        """构建标签倒排索引"""
        for doc in self.docs:
            for tag in doc.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                self.tag_index[tag].append(doc)

    def recommend(
        self,
        search_results: list[UnifiedDocument],
        top_k: int = 5,
    ) -> dict[str, list[UnifiedDocument]]:
        """基于搜索结果做关联推荐

        返回按类型分组的推荐结果：
        {
            "related_docs": [...],   # 同标签文档
            "related_questions": [...],  # 同标签 FAQ
            "related_links": [...],   # 同标签应用链接
        }
        """
        # 1. 提取搜索结果的标签
        result_ids = {d.id for d in search_results}
        tags = []
        for doc in search_results:
            tags.extend(doc.tags)
        tag_freq = Counter(tags)

        if not tag_freq:
            return {"related_docs": [], "related_questions": [], "related_links": []}

        # 2. 取 Top-3 高频标签召回
        top_tags = [tag for tag, _ in tag_freq.most_common(3)]

        # 3. 召回同标签文档并打分
        candidate_scores: dict[str, float] = {}
        candidate_docs: dict[str, UnifiedDocument] = {}

        for tag in top_tags:
            for doc in self.tag_index.get(tag, []):
                if doc.id in result_ids:
                    continue
                candidate_docs[doc.id] = doc
                candidate_scores[doc.id] = candidate_scores.get(doc.id, 0) + 1

        # 4. 按得分排序
        ranked = sorted(
            candidate_docs.items(),
            key=lambda x: candidate_scores[x[0]],
            reverse=True,
        )[:top_k]

        # 5. 按文档类型分组
        result: dict[str, list[UnifiedDocument]] = {
            "related_docs": [],
            "related_questions": [],
            "related_links": [],
        }
        for _doc_id, doc in ranked:
            if doc.doc_type == "faq":
                result["related_questions"].append(doc)
            elif doc.doc_type == "app_link":
                result["related_links"].append(doc)
            else:
                result["related_docs"].append(doc)

        return result
```

- [x] **Step 2: Add tag recommender test**

```python
def _test_tag_recommender():
    """测试标签关联推荐"""
    docs = create_mock_documents()
    recommender = TagRecommender(docs)

    # 搜索"退款"相关文档作为搜索命中
    search_hits = [
        d for d in docs if "退款" in d.tags and d.doc_type != "app_link"
    ]
    recs = recommender.recommend(search_hits, top_k=5)

    print(f"\n[推荐测试] 搜索命中标签: 退款 → 推荐结果:")
    for category, items in recs.items():
        if items:
            print(f"  {category}:")
            for item in items:
                print(f"    [{item.doc_type:>8}] {item.title}")

    # 验证推荐包含应用链接
    assert len(recs["related_links"]) > 0, "应推荐退款相关应用链接"
    print(f"\n标签关联推荐测试通过 ✅")
```

---

### Task 6: SearchEngine & Main Flow

**Files:**
- Modify: `06_advanced_topics/36_search_enhancement.py` (final assembly)

- [x] **Step 1: Implement SearchEngine orchestrator**

```python
# ============================================================
# Teaching Notes: 搜索编排引擎
# ============================================================
# SearchEngine 协调 HybridRetriever 和 TagRecommender，
# 提供一站式搜索 + 推荐入口。
#
# 输出格式为结构化数据，便于后续对接 API 或 Web 界面。


from dataclasses import dataclass


@dataclass
class SearchResult:
    """搜索结果"""
    query: str
    hits: list[UnifiedDocument]
    recommendations: dict[str, list[UnifiedDocument]]


class SearchEngine:
    """搜索增强引擎 — 搜索 + 关联推荐

    三步流程：
      1. 混合检索 → 获取搜索命中结果
      2. 标签推荐 → 从命中文档提取标签做关联推荐
      3. 组合输出 → 搜索结果 + 推荐结果
    """

    def __init__(
        self,
        docs: list[UnifiedDocument],
        persist_dir: str = "./chroma_db",
    ):
        self.docs = docs
        self.retriever = HybridRetriever(docs, persist_dir)
        self.recommender = TagRecommender(docs)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> SearchResult:
        """搜索 + 推荐一站式入口"""
        # 1. 混合检索
        hits = self.retriever.search(query, top_k=top_k)
        hit_docs = [doc for doc, _score in hits]

        # 2. 标签关联推荐
        recommendations = self.recommender.recommend(hit_docs)

        return SearchResult(
            query=query,
            hits=hit_docs,
            recommendations=recommendations,
        )
```

- [x] **Step 2: Write demo scenarios**

```python
def run_demo(engine: SearchEngine, query: str):
    """运行一个搜索演示并打印结果"""
    print(f"\n{'='*60}")
    print(f"查询: 「{query}」")
    print(f"{'='*60}")

    result = engine.search(query)

    # 搜索结果
    print(f"\n📌 搜索命中 ({len(result.hits)} 条):")
    for i, doc in enumerate(result.hits, 1):
        print(f"  {i}. [{doc.doc_type}] {doc.title}")
        if doc.url:
            print(f"     🔗 {doc.url}")
        print(f"     标签: {', '.join(doc.tags)}")

    # 关联推荐
    has_rec = False
    if result.recommendations.get("related_docs"):
        has_rec = True
        print(f"\n📄 相关文档:")
        for doc in result.recommendations["related_docs"]:
            print(f"  · [{doc.doc_type}] {doc.title}")

    if result.recommendations.get("related_questions"):
        has_rec = True
        print(f"\n❓ 相关问题:")
        for doc in result.recommendations["related_questions"]:
            print(f"  · {doc.title}")

    if result.recommendations.get("related_links"):
        has_rec = True
        print(f"\n🔗 相关链接:")
        for doc in result.recommendations["related_links"]:
            print(f"  · {doc.title} → {doc.url}")

    if not has_rec:
        print("  （无关联推荐）")
```

- [x] **Step 3: Update main entry point**

```python
if __name__ == "__main__":
    print("=" * 60)
    print("  搜索增强示例 — 混合检索 + 标签关联推荐")
    print("=" * 60)

    # === 1. 数据加载 ===
    print("\n📦 加载文档数据...")
    docs = create_mock_documents()
    print(f"  共加载 {len(docs)} 篇文档/链接")

    # === 2. 初始化引擎 ===
    print(f"\n🔧 初始化混合检索引擎...")
    engine = SearchEngine(docs)

    # === 3. 演示搜索 ===
    print(f"\n{'='*60}")
    print("  演示场景")
    print(f"{'='*60}")

    # 场景 1：退款相关搜索
    run_demo(engine, "退款流程")

    # 场景 2：拼写错误
    run_demo(engine, "tui款怎么办")

    # 场景 3：英文近义词
    run_demo(engine, "return product")

    # 场景 4：口语化查询
    run_demo(engine, "东西坏了想退钱")

    # 场景 5：订单相关
    run_demo(engine, "查订单")

    # === 4. 测试 ===
    print(f"\n{'='*60}")
    print("  运行单元测试")
    print(f"{'='*60}")
    _test_bm25()
    _test_hybrid_retriever()
    _test_tag_recommender()

    print(f"\n{'='*60}")
    print("  全部完成 ✅")
    print(f"{'='*60}")
```

---

### Task 7: Verify & Polish

- [x] **Step 1: Run the complete example**

Run: `python 06_advanced_topics/36_search_enhancement.py`

Expected: Demo scenarios print correctly with search hits + tag-based recommendations.

- [x] **Step 2: Fix any issues found during run**

- [x] **Step 3: Add teaching notes at the top of the file**

Include a docstring describing the six topics this example covers:
1. 多源数据建模
2. BM25 中文关键词搜索
3. ChromaDB 向量语义搜索
4. RRF 融合排序与降级策略
5. 标签关联推荐
6. 编排引擎
