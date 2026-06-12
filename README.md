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

## 运行示例

```bash
# 运行单个示例
python 01_basic_usage.py

# 运行所有示例
for f in *.py; do echo "=== $f ==="; python "$f"; done

# 运行 LangSmith 追踪示例
python 11_langsmith_demo.py
```
