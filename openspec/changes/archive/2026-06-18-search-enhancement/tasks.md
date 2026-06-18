# 搜索增强功能 - 任务清单

## Task 1: 数据模型和 Mock 数据
- 定义 UnifiedDocument Pydantic 模型
- 创建 Mock 数据（覆盖 wiki/pdf/word/faq/app_link 类型）
- 验证数据模型序列化

## Task 2: BM25 索引构建
- 基于 rank_bm25 构建关键词索引
- BM25 检索函数（输入查询 → 返回 Top-K 文档）
- 测试：拼写错误查询（"退kuan" → "退款"）

## Task 3: 向量索引构建（ChromaDB）
- 初始化 ChromaDB 客户端
- 文档分块（RecursiveCharacterTextSplitter）
- 嵌入（Ollama nomic-embed-text）
- 写入向量库

## Task 4: 混合检索器（HybridRetriever）
- BM25 检索 + 向量检索并行执行
- RRF 融合排序算法
- 错误处理与降级策略
- 测试：语义查询（"return" → "退款/退货"）

## Task 5: 标签关联推荐（TagRecommender）
- 从搜索结果提取标签
- 按标签召回关联文档/FAQ/链接
- 去重排序
- 测试：搜索"订单" → 推荐相关 FAQ 和链接

## Task 6: 搜索编排引擎（SearchEngine）
- 编排 HybridRetriever + TagRecommender
- 统一搜索入口
- 结构化结果输出

## Task 7: 主程序入口
- 整合所有模块
- 命令行交互示例
- 教学注释（中文）
