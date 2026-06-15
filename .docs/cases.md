# LangChain 实际案例学习

> 汇总 LangChain 生态中的真实项目、公司产品应用、RAG 场景和 LangGraph Agent 场景。

---

## 一、LangChain 开源项目实战

### 1.1 官方生态项目

- **langchain-ai/langchain** — 核心框架 (139k stars, Python)
  - 模型接口、Prompt 管理、Chains、Agents、Tools、RAG、流式输出
  - 定位：LLM 应用的积木库

- **langchain-ai/langgraph** — Agent 编排框架
  - 状态图、循环、分支、条件路由、持久化、human-in-the-loop
  - 定位：构建可控的 Agent 工作流

- **langchain-ai/deepagents** — 高层 Agent 抽象
  - 内置规划、子 Agent、文件系统、沙箱
  - 定位：快速构建长运行 Agent

- **langchain-ai/langserve** — 部署 REST API
  - 一行代码将 Chain/Agent 转为 REST API
  - 定位：Agent 部署中间件

### 1.2 热门第三方项目

- **RooVetGit/Roo-Code** — VS Code AI 编程助手
  - 基于 LangChain Agent，支持多模型
  - GitHub 50k+ stars

- **mem0ai/mem0** — 长期记忆系统
  - 跨会话记忆存储和检索
  - 可搭配任何 LLM/Agent 框架

- **run-llama/llama_index** — RAG 框架（与 LangChain 互补）
  - 专注数据连接和索引
  - 可与 LangChain 集成使用

### 1.3 官方模板（LangSmith Fleet Templates）

- 客服机器人模板 — 带知识库的对话 Agent
- RAG 知识库模板 — 文档加载 + 向量检索 + 回答
- 代码审查 Agent — 多步骤 Code Review 流程
- 数据分析 Agent — 自然语言转 SQL + 可视化

---

## 二、公司在产品中的 LangChain 应用

### 2.1 Box AI — 企业内容平台 AI-Native 转型

- **公司**：Box（企业云存储，NASDAQ: BOX）
- **场景**：企业内部文档智能搜索与总结
- **技术**：Deep Agents + LangSmith
- **亮点**：
  - 从传统搜索到 AI-Native 的完整转型
  - 企业级 RAG，支持多格式文档
  - 权限控制确保数据安全
  - 2026 年 6 月发布的最新案例

### 2.2 Harvey — 法律 AI

- **公司**：Harvey（法律科技独角兽，估值 $7B+）
- **场景**：法律文档审查、合同分析、判例检索
- **技术**：LangGraph Agent + 验证器
- **亮点**：
  - 极高精度要求（法律场景不能幻觉）
  - 内置验证器验证 Agent 输出
  - 合规性和可解释性优先
  - 2026 年 6 月发布《Designing Efficient Verifiers for Legal Agents》

### 2.3 Rippling — HR 平台

- **公司**：Rippling（HR 管理平台，估值 $35B+）
- **场景**：HR 智能助手、员工自助服务
- **技术**：Deep Agents + LangSmith
- **亮点**：
  - 6 个月内全产品线 AI 化
  - 跨产品统一 Agent 平台
  - 覆盖薪资、福利、设备管理等场景

### 2.4 Harmonic — 体育内容管理

- **公司**：Harmonic（体育内容分发平台）
- **场景**：体育内容智能分发
- **技术**：Deep Agents + Scout + LangSmith
- **亮点**：
  - 用户留存提升 4 倍
  - AI 驱动的内容匹配和推荐
  - 2026 年 6 月发布案例研究

### 2.5 Lyft — 出行平台客服

- **公司**：Lyft（美国第二大打车平台）
- **场景**：客服自助 Agent 平台
- **技术**：LangGraph + LangSmith
- **亮点**：
  - Self-serve 架构，多 Agent 协作
  - 处理订单查询、退款、投诉等场景
  - 人机协同，复杂情况转人工
  - 2026 年 5 月发布案例研究

### 2.6 Benchling — 生命科学平台

- **公司**：Benchling（生命科学协作平台）
- **场景**：科研数据智能分析
- **技术**：LangGraph Agent
- **亮点**：
  - 复杂科学场景的 Agent 编排
  - 智能体能力不足以覆盖所有场景时的混合方案
  - 2026 年 6 月 Max Agency Podcast 访谈

---

## 三、RAG 实际应用场景

### 3.1 企业知识库

- **场景**：公司内部文档/手册/FAQ 智能问答
- **痛点**：
  - 文档格式多样（PDF、Word、Markdown、Wiki）
  - 权限控制（不同部门看不同文档）
  - 文档更新后知识库同步
- **方案**：
  - 文档加载（PyPDFLoader、WebBaseLoader、DirectoryLoader）
  - 向量化（混合检索 + 重排序）
  - 权限过滤（metadata 过滤）
  - 定时增量更新

### 3.2 客服机器人

- **场景**：产品 FAQ、订单查询、技术支持
- **痛点**：
  - 答案准确性要求高
  - 用户会追问，需要多轮对话
  - 需要引用来源增加信任
- **方案**：
  - RAG + 来源引用（[Doc N] 标记）
  - 重排序提升召回精度
  - 多查询检索覆盖不同问法
  - 人工反馈闭环（用户点赞/踩）

### 3.3 法律文档分析

- **场景**：合同审查、判例检索、合规检查
- **痛点**：
  - 极高准确性要求（零容忍幻觉）
  - 需要引用具体条款
  - 数据隐私和合规
- **方案**：
  - RAG + 严格验证器（Harvey 模式）
  - 每个回答必须标注引用来源
  - 人工审批关键输出
  - 审计日志记录所有操作

### 3.4 医疗辅助

- **场景**：病历查询、药品交互检查、诊疗指南
- **痛点**：
  - 安全性第一（错误可能致命）
  - HIPAA 合规要求
  - 需要医生最终审核
- **方案**：
  - RAG + 强验证 + 医生审批
  - 加密存储和传输
  - 只允许医生查看自己患者的病历

### 3.5 代码知识库

- **场景**：代码搜索、架构理解、新人 Onboarding
- **痛点**：
  - 代码结构理解（跨文件关联）
  - 需要执行验证
  - 大仓库索引慢
- **方案**：
  - 代码分块（按函数/类分割）
  - 语义索引 + 关键字混合检索
  - 代码执行沙箱验证
  - 增量索引更新

### 3.6 金融研报分析

- **场景**：财报解读、市场趋势分析、投资组合建议
- **痛点**：
  - 多源数据融合（文本 + 表格 + 图表）
  - 时效性要求高
  - 合规审查
- **方案**：
  - 多文档 RAG（PDF + CSV + HTML）
  - 多模态（图片/图表理解）
  - 实时数据注入
  - 合规审核工作流

---

## 四、LangGraph Agent 实际应用场景

### 4.1 多 Agent 协作系统

- **场景 A：内容审核**
  - Supervisor 分发：安全检测、合规检查、质量评估并行
  - 每个专家 Agent 专注一个维度
  - Supervisor 综合决策：通过/修改/拒绝

- **场景 B：客户工单处理**
  - 技术问题 → 技术 Agent
  - 账单问题 → 账单 Agent
  - 退货问题 → 退货 Agent
  - Supervisor 根据意图分类路由

- **技术**：StateGraph + Supervisor 模式（对应样例 03_langgraph_agents/14_langgraph_advanced.py）

### 4.2 人机协同工作流

- **场景 A：金融交易审批**
  - Agent 分析交易 → 中断等待人工批准 → 批准后执行
  - 大额交易必须人工审批

- **场景 B：代码合并审批**
  - Agent 审查代码 → 生成报告 → 人工确认后合并

- **场景 C：医疗处方审核**
  - Agent 检查药物交互 → 标记风险 → 医生确认

- **技术**：interrupt() 暂停 + 人工恢复（对应样例 03_langgraph_agents/14_langgraph_advanced.py）

### 4.3 长期记忆 Agent

- **场景 A：个人助理**
  - 记住用户偏好、日程、联系人
  - 跨会话持续学习

- **场景 B：教育辅导**
  - 记录学生水平和薄弱点
  - 自适应调整教学策略

- **场景 C：健康管理**
  - 跟踪饮食、运动、病史
  - 提供个性化建议

- **技术**：VectorStoreRetrieverMemory + 持久化 Checkpointer

### 4.4 自主研究 Agent

- **场景 A：市场调研**
  - 规划研究框架 → 搜索信息 → 分析数据 → 生成报告
  - 多轮迭代，发现新问题自动深入

- **场景 B：竞品分析**
  - 识别竞品 → 收集产品/价格/功能数据 → 对比分析
  - 可视化输出

- **场景 C：学术文献综述**
  - 搜索论文 → 提取关键信息 → 归纳分类 → 撰写综述
  - 引用溯源

- **技术**：LangGraph 循环 + 反思 + 工具调用（search/web/code）

### 4.5 代码生成 Agent

- **场景 A：自动化开发**
  - 需求分析 → 代码生成 → 测试 → 修复 → 部署
  - 多轮迭代，直到测试通过

- **场景 B：Code Review**
  - 审查代码质量 → 检查安全性 → 检查性能 → 提出改进建议
  - 自动修复简单问题

- **场景 C：Bug 修复**
  - 分析错误日志 → 定位问题 → 生成修复 → 验证

- **技术**：多 Agent 协作 + 代码执行沙箱 + 测试工具

### 4.6 数据分析师 Agent

- **场景 A：BI 自助分析**
  - 自然语言查询 → SQL 生成 → 数据查询 → 可视化 → 报告
  - 无需写 SQL，业务人员自助分析

- **场景 B：经营数据查询**
  - "上季度华东区销售额趋势" → 自动生成图表
  - "对比各产品线利润率" → 自动计算

- **技术**：Text-to-SQL + 数据验证 + 图表生成 + 数据格式化工具

---

## 五、技术选型与架构模式

### 5.1 何时用 LangChain vs LlamaIndex vs 自建

| 维度 | LangChain | LlamaIndex | 自建 |
|------|-----------|------------|------|
| 最佳场景 | 通用 LLM 应用 | 数据连接/RAG | 极度定制化 |
| 优势 | 生态丰富、社区大 | RAG 深度优化 | 完全可控 |
| 劣势 | RAG 需组合多个组件 | Agent 生态较弱 | 开发成本高 |
| 推荐 | 大多数项目首选 | 重度 RAG 场景 | 有专门 ML 团队 |

### 5.2 RAG vs Fine-tuning vs Prompt Engineering

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 知识问答 | RAG | 数据频繁更新，Fine-tune 跟不上 |
| 风格迁移 | Prompt Engineering | 不需要新知识 |
| 专业领域 | RAG + Fine-tune | RAG 提供知识，Fine-tune 优化风格 |
| 工具调用 | Agent | 需要动态决策 |

### 5.3 Agent vs Chain vs 传统 API

| 方案 | 适用场景 | 示例 |
|------|----------|------|
| Chain | 确定性的多步流程 | 数据清洗 → 分析 → 报告 |
| Agent | 需要决策和工具调用 | 客服、研究、代码生成 |
| 传统 API + LLM | 单次生成任务 | 翻译、摘要、分类 |

### 5.4 向量数据库选型决策树

```
数据规模?
├── < 10 万条
│   ├── 原型/开发 → Chroma
│   └── 需要快速检索 → FAISS
├── 10 万 - 1000 万条
│   ├── 需要过滤 → Qdrant
│   ├── 已有 PostgreSQL → PGVector
│   └── 需要语义+结构化 → Weaviate
└── > 1000 万条
    ├── 需要分布式 → Milvus
    └── 不想管运维 → Pinecone
```

### 5.5 多模型回退架构设计

```
用户请求
  ↓
┌─────────────────────────────┐
│  主模型 (GPT-4o)           │ ← 高质量，成本较高
│  .with_fallbacks([         │
│    Gemini (便宜替代)       │
│    Claude (备选)           │
│    Llama3 (本地免费)       │
│  ])                        │
└─────────────────────────────┘
  ↓
[成功] 返回结果
[失败] 自动切换到下一个模型
```

---

## 六、学习路径建议

### 入门阶段（1-2 周）
1. 跑通 01-07 基础示例
2. 了解 LCEL 基本语法
3. 完成第一个 Chain + Agent 项目

### 进阶阶段（2-4 周）
4. 学习 RAG（08 + 15）+ 文档加载（19）
5. 学习 LangGraph 基础（10 + 14）
6. 实现一个带 RAG 的 Agent

### 高级阶段（4-8 周）
7. 学习 LangServe 部署（12）+ LangChain Hub（13）
8. 学习 LangSmith 评估（11 + 16）+ 监控（20）
9. 实现完整的 RAG + Agent + 部署 + 评估闭环

### 实战阶段
10. 选择一个实际场景（参考本章案例）
11. 从零搭建完整系统
12. 用 LangSmith 持续优化

---

## 七、参考资源

- **官方文档**：https://docs.langchain.com/
- **LangChain Blog**：https://blog.langchain.com/
- **LangSmith Stories**：https://smith.langchain.com/stories
- **Customer Stories**：https://www.langchain.com/customers
- **LangChain Academy**：https://academy.langchain.com/（免费课程）
- **官方 GitHub**：https://github.com/langchain-ai
- **社区论坛**：https://forum.langchain.com/
- **Max Agency Podcast**：https://www.youtube.com/playlist?list=PLfaIDFEXuae3UwB1QGEjsRAr8BzCQss7s
