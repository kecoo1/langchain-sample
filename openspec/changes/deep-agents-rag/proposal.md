## Why

项目已有独立的 Deep Agents (22_deep_agents_example.py) 和 RAG (08_rag, 15_rag_advanced) 示例，但缺少两者的结合——用 deepagents 框架的 `create_deep_agent` 驱动智能 RAG 检索系统。学习者需要理解如何将 Agent 的规划、工具调用、子代理能力与 RAG 检索策略融合，构建自主检索系统。

## What Changes

- 新增 `06_advanced_topics/38_deep_agents_rag.py`：使用 deepagents 框架构建自主 RAG Agent
- 构建结构化技术知识库（LangChain 产品文档/FAQ）作为检索源
- 实现 Agent 驱动的多步检索管道：查询分解 → 多检索器路由 → 多源融合 → 结果评估迭代 → 溯源引用
- 采用"自主检索 Agent"模式：用户一句话提问，Agent 全自动编排检索流程
- 复用已有的 deepagents 框架能力（todo planning、子代理、虚拟文件系统、human-in-the-loop）

## Capabilities

### New Capabilities
- `deep-agents-rag`: deepagents 框架驱动的自主 RAG 检索 Agent，展示 Agent 与 RAG 的完整融合模式

### Modified Capabilities

无

## Impact

- 新增文件：`06_advanced_topics/38_deep_agents_rag.py`
- 依赖：deepagents (v0.6.10 已安装)、langchain-community、chromadb、rank_bm25
- 模型：ollama:llama3.2:1b（默认，tool calling 能力有限）
