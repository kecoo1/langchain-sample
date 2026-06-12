# LangChain 样例工程

这个工程包含 LangChain 框架的各种使用示例，覆盖了从基础到高级的常见场景。

## 环境准备

```bash
# 安装依赖
uv sync
# 或者
pip install -r requirements.txt

# 设置 API Key
export OPENAI_API_KEY="sk-..."
# 可选
export ANTHROPIC_API_KEY="sk-ant-..."

# LangSmith 追踪（可选）
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"
```

## 示例列表

| 文件 | 内容 | 难度 |
|------|------|------|
| `01_basic_usage.py` | 基础调用 (sync/async) | ⭐ |
| `02_prompt_templates.py` | 提示模板、Few-shot、对话历史 | ⭐ |
| `03_chains.py` | Runnable 链组合、并行、分支 | ⭐⭐ |
| `04_tools.py` | 工具定义、工具调用、结构化参数 | ⭐⭐ |
| `05_agent.py` | Agent (ReAct/Tool-calling)、记忆 | ⭐⭐⭐ |
| `06_streaming.py` | 流式输出、Async、事件 | ⭐⭐ |
| `07_structured_output.py` | Pydantic 结构化输出 | ⭐⭐ |
| `08_rag.py` | RAG (检索增强生成) | ⭐⭐⭐ |
| `09_multimodal.py` | 多模态 (图片输入) | ⭐⭐ |
| `10_langgraph_basics.py` | LangGraph 状态图、Agent、聊天机器人 | ⭐⭐⭐ |
| `11_langsmith_demo.py` | LangSmith 追踪：链、Agent、LangGraph | ⭐⭐⭐ |
| `12_langserve_basic.py` | LangServe API 部署 | ⭐⭐ |
| `13_langchain_hub.py` | LangChain Hub Prompt 管理、版本控制 | ⭐⭐ |
| `14_langgraph_advanced.py` | LangGraph 进阶：Supervisor、人类审批、嵌套子图 | ⭐⭐⭐ |
| `15_rag_advanced.py` | RAG 进阶：混合检索、重排序、引用溯源、HyDE | ⭐⭐⭐ |
| `16_langsmith_evaluation.py` | LangSmith 进阶：数据集评估、A/B 测试 | ⭐⭐⭐ |
| `17_lcel_advanced_patterns.py` | LCEL 高级：.map/.reduce/.with_fallbacks/.with_retry | ⭐⭐⭐ |
| `18_vector_databases.py` | 向量数据库：FAISS/Pinecone/Qdrant/PGVector 对比 | ⭐⭐ |
| `19_document_loaders.py` | 文档加载：PDF/Web/JSON/CSV/Markdown/目录批量 | ⭐⭐ |
| `20_callbacks_and_monitoring.py` | 回调和监控：Token追踪/延迟追踪/异步回调/审计日志 | ⭐⭐⭐ |
| `21_multi_provider_llm.py` | 多提供商：Gemini/HuggingFace/Cohere 集成 | ⭐⭐ |

## 运行示例

```bash
# 运行单个示例
python 01_basic_usage.py

# 运行所有示例
for f in *.py; do echo "=== $f ==="; python "$f"; done

# 运行 LangSmith 追踪示例
python 11_langsmith_demo.py
```
