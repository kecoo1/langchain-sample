# LangChain 样例工程

这个工程包含 LangChain 框架的各种使用示例，按功能分类组织。

## 环境准备

```bash
# 安装依赖
uv sync

# 设置 API Key
export OPENAI_API_KEY="sk-..."
# 可选
export ANTHROPIC_API_KEY="sk-ant-..."

# LangSmith 追踪（可选）
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"
```

## 目录结构

### 01_basics/ — 核心基础 (7 个)

| 文件 | 内容 | 难度 |
|------|------|------|
| `01_basic_usage.py` | 基础调用 (sync/async) | ⭐ |
| `02_prompt_templates.py` | 提示模板、Few-shot、对话历史 | ⭐ |
| `03_chains.py` | Runnable 链组合、并行、分支 | ⭐⭐ |
| `04_tools.py` | 工具定义、工具调用、结构化参数 | ⭐⭐ |
| `05_agent.py` | Agent (ReAct/Tool-calling)、记忆 | ⭐⭐⭐ |
| `06_streaming.py` | 流式输出、Async、事件 | ⭐⭐ |
| `07_structured_output.py` | Pydantic 结构化输出 | ⭐⭐ |

### 02_rag/ — 检索增强生成 (5 个)

| 文件 | 内容 | 难度 |
|------|------|------|
| `08_rag.py` | RAG 基础流程 | ⭐⭐⭐ |
| `15_rag_advanced.py` | 混合检索、重排序、引用溯源、HyDE | ⭐⭐⭐ |
| `19_document_loaders.py` | 文档加载：PDF/Web/JSON/CSV/Markdown | ⭐⭐ |
| `28_graph_rag.py` | 知识图谱 RAG：实体提取/关系扩展/社区检测 | ⭐⭐⭐ |
| `29_chunking_strategies.py` | 分块策略对比：固定/递归/Markdown/语义 | ⭐⭐ |

### 03_langgraph_agents/ — 图编排与 Agent (6 个)

| 文件 | 内容 | 难度 |
|------|------|------|
| `10_langgraph_basics.py` | LangGraph 状态图、Agent、聊天机器人 | ⭐⭐⭐ |
| `14_langgraph_advanced.py` | Supervisor/人类审批/嵌套子图/重试 | ⭐⭐⭐ |
| `22_deep_agents_example.py` | Deep Agents 规划/文件系统/子Agent/人机协同 | ⭐⭐⭐ |
| `27_plan_reflexion.py` | Plan-and-Execute / Reflexion / 完整循环 | ⭐⭐⭐ |
| `30_parallel_tool_calling.py` | 并行工具调用：基础/依赖/批量 | ⭐⭐⭐ |
| `31_code_generation_agent.py` | 代码生成：语法验证/修复循环/多文件/审查 | ⭐⭐⭐ |
| `32_multi_agent_dev_team.py` | 实战：多Agent开发团队（PM/架构/开发/测试） | ⭐⭐⭐ |
| `33_customer_service_agent.py` | 实战：智能客服（分类/专业Agent/转人工） | ⭐⭐⭐ |
| `34_data_analysis_agent.py` | 实战：数据分析报告（数据加载/分析/报告生成） | ⭐⭐⭐ |

### 04_monitoring_deployment/ — 监控与部署 (5 个)

| 文件 | 内容 | 难度 |
|------|------|------|
| `11_langsmith_demo.py` | LangSmith 追踪 | ⭐⭐⭐ |
| `12_langserve_basic.py` | LangServe API 部署 | ⭐⭐ |
| `13_langchain_hub.py` | Hub Prompt 管理、版本控制 | ⭐⭐ |
| `16_langsmith_evaluation.py` | 数据集评估、A/B 测试 | ⭐⭐⭐ |
| `20_callbacks_and_monitoring.py` | Token追踪/延迟/异步/审计日志 | ⭐⭐⭐ |

### 05_integrations_lcel/ — 生态集成 (4 个)

| 文件 | 内容 | 难度 |
|------|------|------|
| `09_multimodal.py` | 多模态（图片输入） | ⭐⭐ |
| `17_lcel_advanced_patterns.py` | LCEL 高级：.map/.reduce/.with_fallbacks | ⭐⭐⭐ |
| `18_vector_databases.py` | 向量数据库：FAISS/Pinecone/Qdrant/PGVector | ⭐⭐ |
| `21_multi_provider_llm.py` | 多提供商：Gemini/HuggingFace/Cohere | ⭐⭐ |

### 06_advanced_topics/ — 高级专题 (4 个)

| 文件 | 内容 | 难度 |
|------|------|------|
| `23_memory_systems.py` | 长期记忆：滑动窗口/摘要/向量/实体/组合 | ⭐⭐ |
| `24_text_to_sql.py` | Text-to-SQL：安全查询/数据分析Agent | ⭐⭐⭐ |
| `25_guardrails.py` | 安全防护：注入检测/PII脱敏/输出验证 | ⭐⭐⭐ |
| `26_agent_evaluation.py` | Agent评估：在环/RAGAS/自定义/A-B对比 | ⭐⭐⭐ |

## 运行示例

```bash
# 运行单个示例
python 01_basics/01_basic_usage.py

# 运行某个分类
for f in 02_rag/*.py; do echo "=== $f ==="; python "$f"; done

# 运行所有示例
for d in 0*; do for f in "$d"/*.py; do echo "=== $f ==="; python "$f"; done; done
```
