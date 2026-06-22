## Context

项目已有独立的 Deep Agents 示例（22_deep_agents_example.py）和 RAG 示例（08_rag.py, 15_rag_advanced.py），但缺少两者的融合。37_search_agent.py 使用 LangGraph StateGraph 实现了基础的搜索 Agent，但没有利用 deepagents 框架的内置能力（规划、子代理、虚拟文件系统、human-in-the-loop）。

本次新增 38_deep_agents_rag.py，用 deepagents 的 `create_deep_agent()` 构建一个自主 RAG Agent，展示 deepagents 框架如何驱动智能检索。

## Goals / Non-Goals

**Goals:**
- 使用 deepagents 框架构建自主 RAG Agent
- 实现 5 项核心 RAG 能力：查询分解、多检索器路由、多源融合、结果评估迭代、溯源引用
- 展示 deepagents 内置能力的 RAG 应用：todo planning、sub-agent、virtual filesystem、interrupt_on
- 使用 Ollama 本地模型（llama3.2:1b），标注能力限制和替代方案
- 遵循项目现有的教学风格：教学备注、渐进式示例、独立可运行

**Non-Goals:**
- 不引入新的向量数据库（复用 ChromaDB）
- 不构建生产级 RAG 系统（作为教学示例）
- 不覆盖 GraphRAG、HyDE 等（已有 15_rag_advanced + 28_graph_rag）
- 不对比 deepagents 与 LangGraph 的性能（侧重概念教学）

## Decisions

### Decision 1: 知识库使用结构化技术 FAQ 而非通用文档
**选择**：构建一个模拟的 LangChain 技术 FAQ 知识库，包含多个分类（基础概念、组件、最佳实践、故障排除、FAQ）。
**理由**：结构化知识库使 Agent 的"多源融合"和"检索器路由"能力更直观——不同类别问题需要不同检索策略。纯文本文档无法展示 Agent 的智能路由决策。
**替代方案**：使用现有 08_rag.py 的自由文本 — 过于简单，无法展示 Agent 的多步推理。

### Decision 2: RAG 工具作为独立函数，而非在 agent 逻辑中硬编码
**选择**：将 `vector_search`、`keyword_search`、`hybrid_search`、`evaluate_relevance` 定义为独立的 `@tool` 函数，通过 `tools` 参数注入 `create_deep_agent()`。
**理由**：deepagents 的 tool calling 模式是 Agent 获取外部能力的标准方式。独立工具让 Agent 可以自主选择何时用什么策略，而非硬编码流程。
**替代方案**：在 system_prompt 中描述 RAG 逻辑 — 不灵活，无法发挥 deepagents 的 tool calling 优势。

### Decision 3: 使用 write_todos 实现查询分解
**选择**：当用户提出多部分问题时，Agent 通过 `write_todos` 自动创建子任务，每个子任务对应一个子查询。
**理由**：这是 deepagents 的内置能力，不需要额外代码。Agent 自动将"比较 A 和 B"拆解为"检索 A"、"检索 B"、"对比分析"。
**替代方案**：手动实现 TaskList 逻辑 — 与 deepagents 重复，且失去了框架优势。

### Decision 4: 多检索器路由通过 tool description + Agent 自主决策
**选择**：每个检索工具的 description 清晰说明适用场景（如 `vector_search`：适合语义相似问题；`keyword_search`：适合精确术语匹配），Agent 根据问题自主选择。
**理由**：展示 deepagents 的 tool calling 智能，不硬编码路由逻辑。Agent 通过理解工具描述来选择正确的检索策略。
**替代方案**：用 LangGraph 条件边硬编码路由 — 那是 37_search_agent.py 的方式，违背了 deepagents 的"高自主性"设计理念。

### Decision 5: 结果评估通过 Agent 的自我反思
**选择**：将 `evaluate_relevance` 作为工具提供给 Agent，Agent 可以在检索后调用它评估结果，然后决定是否重新检索。
**理由**：Agent 自主决定"够了没、还要不要搜"——这是 Agent RAG 比固定 RAG pipeline 的核心优势。
**替代方案**：固定 RAG pipeline 没有评估环节 — Agent RAG 的核心差异点就在这。

### Decision 6: 溯源引用通过工具返回 metadata
**选择**：检索工具返回带 metadata 的结果（doc_id, title, score），Agent 在回答中引用 [Doc1][Doc2] 格式。
**理由**：Agent 已经能看到文档 metadata，prompt 引导即可实现溯源。无需额外格式化逻辑。
**替代方案**：后处理强制添加引用 — 不自然，且 Agent 可能忽略。

## Risks / Trade-offs

- **[风险] llama3.2:1b 工具调用不稳定** → 所有工具调用包裹 try-catch，提供 fallback 消息；教学备注标注推荐升级模型
- **[风险] Agent 可能偏离检索策略** → 通过精心设计的 tool description + system_prompt 引导行为；示例代码的 `__main__` 块演示典型查询路径
- **[风险] deepagents 版本兼容性** → 已安装 v0.6.10，导入用 try-catch 包裹，缺失时友好提示
- **[风险] 教学复杂度** → 兼顾 22_deep_agents.py 的 simple 风格和 RAG 的技术深度，每个示例有教学备注
