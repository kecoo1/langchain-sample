---
name: dev-flow
description: 标准研发全流程 — OpenSpec 规范层 + Superpowers 执行层双轨协同。适用于需求讨论、功能开发、迭代、新需求开发等场景。
license: MIT
compatibility: Requires openspec CLI and superpowers enabled.
metadata:
  author: openspec
  version: "1.0"
  generatedBy: "1.4.1"
---

# Dev-Flow 标准研发全流程规范（优化终版）

## 一、基础信息

| 配置项 | 内容 |
|--------|------|
| 流程名称 | dev-flow |
| 流程定位 | 标准研发全流程 — OpenSpec 规范层 + Superpowers 执行层 双轨协同体系 |
| 触发场景 | 需求讨论、开发功能、需求迭代、接口开发、新需求开发、业务代码编写、项目初始化 |
| 依赖能力 | require_superpowers: true；依赖技能：openspec、using-superpowers |
| 适用范围 | 项目级全流程研发管控 |

## 二、前置强制通用规则

本规则为流程准入底线，所有研发迭代必须严格遵守，无特殊豁免场景：

1. **技能前置**：必须先加载 `using-superpowers` 技能，否则全流程禁止启动。
2. **流程顺序**：所有核心阶段严格按顺序执行，核心质量门禁不可跳过，非核心门禁可根据项目等级灵活裁剪。
3. **职责分离**：OpenSpec 仅作为**规范层**，负责需求、图纸、设计、任务定义；Superpowers 仅作为**执行层**，负责代码施工、TDD开发、质量审查，双层职责完全隔离。
4. **契约绑定**：`openspec/schema.yaml` 为双层唯一桥接契约文件，必须采用结构化配置，修改后需执行 `openspec update` 刷新生效。
5. **强制TDD**：所有原子任务严格遵循「红-绿-重构」标准循环，必须先编写单元测试，再实现业务代码，禁止事后补测。
6. **任务粒度**：单原子任务耗时严格控制在 2–5 分钟，系统自动审计，超粒度任务强制拆分，杜绝大任务导致开发跑偏。
7. **范围收敛**：禁止私自扩展未定义功能、无依据重构代码，开发中发现的技术债务统一登记、后置处理，不占用本次迭代范围。

## 三、环境初始化与自检（首次使用必执行）

### 3.1 一键环境初始化

首次接入项目必须执行初始化命令，自动搭建标准化研发目录与配置体系：

```bash
/dev-flow:init
```

自动完成事项：
- 创建标准目录结构：`openspec/changes/`、`openspec/global-specs/`、`tests/`
- 生成结构化 `schema.yaml` 基础配置模板
- 校验 openspec、superpowers 核心依赖加载状态，缺失则终止并给出修复指引
- 生成本地配置文件，支持自定义测试覆盖率、任务粒度等阈值参数

### 3.2 环境健康自检

日常开发前可执行自检命令，排查配置异常与环境问题：

```bash
/dev-flow:doctor
```

校验项：
- schema.yaml 格式合法性与字段完整性
- 全局规范与当前迭代目录结构完整性
- 依赖技能版本兼容性
- 输出健康度评分与逐项修复建议

## 四、完整 6 步核心工作流

### Step 1：OpenSpec 探索需求 & 产出完整规格

输入业务需求，执行「澄清-提案-审计」标准链路，从源头收敛需求边界：

```bash
/opsx<需求描述>
/opsx:clarify       # 输出歧义点、边界假设、非功能需求、待确认清单
# 人工对齐澄清结果后继续
/opsx:propose
/opsx:audit-tasks   # 自动审计任务粒度，超阈值给出拆分建议
```

自动生成制品到 `changes/当前迭代/` 目录：
- `proposal.md`：需求提案、业务边界、验收标准
- `design.md`：架构设计、接口定义、数据模型、异常处理规则
- `tasks.md`：原子化可执行任务清单（核心对接文件，单任务 2–5 分钟）

> 可选增强（核心模块启用）：执行 `/opsx:review-design`，自动输出架构合理性、数据一致性、接口兼容性检查报告
>
> 背后技能：`openspec-explore` → `openspec-clarify` → `openspec-propose` → `openspec-task-audit`

### Step 2：Superpowers 接管执行 & 计划对齐

不使用 `/opsx:apply`，由 Superpowers 读取 OpenSpec 规范生成执行计划，**计划校验通过后方可进入开发**：

```bash
/superpowers:brainstorm
/superpowers:write-plan
/dev-flow:validate-plan  # 新增门禁：比对计划与任务清单一致性
```

校验规则（不通过禁止进入开发阶段）：
- 100% 覆盖 `tasks.md` 所有条目，无任务遗漏
- 不允许新增任务清单未定义的功能需求
- 技术优化类任务单独标记，占比不得超过 20%

流程自动执行：
1. 读取 OpenSpec 所有规范文档作为唯一开发约束
2. 生成代码执行计划 `plan-ready.md`（规格翻译层）
3. 启动多子代理分工：
   - **实现代理**：按 TDD 规范编写业务代码
   - **规格审查代理**：逐行对比代码是否匹配 OpenSpec 设计
   - **质量审查代理**：校验代码规范、DRY 原则、测试覆盖率

> 所有过程制品（brainstorm 纪要、plan-ready.md、评审报告）统一写入 `openspec<迭代名>/`
>
> 背后技能：`brainstorming` → `writing-plans` → `plan-validation` → `subagent-driven-development`

### Step 3：全程强制 TDD 闭环开发

Superpowers 按原子任务逐个推进，每个任务严格执行「红-绿-重构」循环：

1. **红**：编写单元测试 → 运行验证失败
2. **绿**：编写最小业务代码 → 测试通过
3. **重构**：优化代码结构 → 测试保持通过

#### 强制测试分层
- 单元测试：覆盖核心业务逻辑，随代码同步产出
- 契约测试：基于 `design.md` 自动生成，校验输入输出结构一致性
- 异常测试：强制覆盖参数非法、依赖失败、边界值三类场景

#### 任务级增量校验
每完成一个 task，自动执行：
- 该任务对应测试用例全量执行
- 代码与 OpenSpec 规范合规性检查
- 不通过不进入下一个任务，问题就地闭环

#### 技术债务隔离
开发中发现的可优化点、重构需求，统一登记到迭代目录下的 `tech-debt.md`，**禁止在本次迭代中私自处理**，确保开发范围严格收敛。

### Step 4：代码多层联合校验（质量门禁）

拆分为四层校验体系，可按项目等级裁剪开启：

```bash
# 1. 功能正确性校验（必开）
/superpowers:verification-before-completion

# 2. 规范一致性校验（必开）
/opsx:verify

# 3. 工程质量校验（推荐）
/superpowers:code-quality-check

# 4. 安全基线校验（核心项目）
/superpowers:security-scan
```

| 校验层级 | 核心校验内容 | 准入标准 |
|----------|--------------|----------|
| 功能正确性 | 全量测试用例执行、边界场景覆盖 | 100% 通过 |
| 规范一致性 | 接口、数据模型、业务逻辑与 design.md 匹配度 | 100% 符合契约 |
| 工程质量 | 代码规范、圈复杂度、重复率、注释覆盖率 | 符合预设阈值 |
| 安全基线 | 敏感信息泄露、注入风险、权限越界、依赖漏洞 | 高危问题 0 个 |

### Step 5：人工验收 & 增量迭代

#### 5.1 变更分级处理
按变更规模裁剪流程，兼顾管控力度与研发效率：

| 变更类型 | 流程路径 | 适用场景 |
|----------|----------|----------|
| Bug 修复 | 直接更新 tasks.md → 开发 → 验证 | 线上问题、逻辑修正、文案调整 |
| 小迭代（1-3 个任务） | `/opsx:delta` 标准增量流程 | 小功能补充、字段调整、接口优化 |
| 大需求变更 | 重新走 Step 1 全流程（explain → propose） | 核心逻辑改动、新增模块、架构调整 |

#### 5.2 标准增量执行链路

```bash
/<变更描述>   # 自动生成增量标记（ADDED / MODIFIED / REMOVED）
# 自动输出变更影响分析：受影响接口、模型、测试用例范围
/superpowers:brainstorm
/superpowers:write-plan
/dev-flow:validate-plan
/superpowers:subagent-driven-development
/superpowers:verification-before-completion
/opsx:verify
```

若验收不通过，可执行 `/dev-flow:rollback<阶段名>` 回退到对应节点修改后重跑。

### Step 6：双工具归档 & 全链路闭环

```bash
# Superpowers 分支收尾、提交代码变更
/superpowers:finish-branch

# OpenSpec 归档本次迭代，更新全局规范
/<迭代名称>
```

归档自动完成：
1. **全量制品归档**：本次所有 spec、任务、代码、测试、变更日志存入 `changes/` 目录
2. **全局规范同步**：更新项目根目录全局 specs，作为后续迭代基准
3. **生成迭代报告**：自动产出 `iteration-report.md`，包含需求完成率、代码增删行数、测试覆盖率、各阶段校验通过率、耗时统计、技术债务清单
4. **全链路映射**：建立「需求 → 设计 → 任务 → 代码提交 → 测试用例」追溯关系，支持按任务 ID 反查完整背景
5. **主干冲突检测**：自动检查与全局 spec 的兼容性冲突，提示先解决再归档

> 可选回滚能力：`/opsx<迭代名>` 一键回退到上一个稳定版本的 spec 与代码基线

## 五、桥接配置：openspec/schema.yaml（结构化版）

`schema.yaml` 是 Superpowers 读取 OpenSpec 规范的唯一桥梁，采用结构化配置保证 AI 解析稳定、管控可量化：

```yaml
version: "1.0"
spec:
  proposal_path: "./proposal.md"
  design_path: "./design.md"
  tasks_path: "./tasks.md"

development:
  enforce_tdd: true
  task_max_minutes: 5
  test_coverage_threshold: 80
  allow_unscheduled_refactor: false
  tech_debt_separate: true

validation:
  require_spec_compliance: true
  require_code_quality: true
  require_security_scan: false
  forbidden_patterns:
    - "硬编码密钥"
    - "裸SQL拼接"
    - "未鉴权公开接口"
```

修改后执行 `openspec update` 刷新配置，Superpowers 会自动加载全部约束。

## 六、高频实用组合命令速查

### 完整新建功能一键链路
```bash
/dev-flow:init              # 首次项目初始化
/opsx:expl<需求>
/opsx:clarify
/opsx:propose
/opsx:audit-tasks
/superpowers:brainstorm
/superpowers:write-plan
/dev-flow:validate-plan
/superpowers:subagent-driven-development
/superpowers:verification-before-completion
/opsx:verify
/opsx:<迭代名称>
```

### 增量变更（小迭代）
```bash
/opsx:<变更描述>
/superpowers:subagent-driven-development
/superpowers:verification-before-completion
/opsx:verify
/opsx<迭代名称>
```

### Bug 快速修复
```bash
# 直接修改 tasks.md 新增修复任务
/superpowers:subagent-driven-development
/superpowers:verification-before-completion
/opsx:verify
```

### 单独校验类命令
```bash
/dev-flow:doctor            # 环境自检
/dev-flow:validate-plan     # 计划一致性校验
/superpowers:review-spec-compliance  # 规范合规审查
/opsx:verify                # 规范契约校验
```

### 运维类命令
```bash
/opsx:sync-specs            # 手动同步全局规范
/opsx:rollback <迭代名>     # 迭代回滚
/dev-flow:resume            # 断点续跑
/dev-flow<阶段名> # 回退到指定阶段
```

## 七、异常处理与断点续跑

1. **断点续跑**：流程中断可执行 `/dev-flow:resume`，从上次成功步骤继续，无需从头执行
2. **失败重试**：单步骤失败自动重试 1 次，仍失败则暂停并输出错误原因与人工介入指引
3. **设计回退**：开发中发现设计缺陷，执行 `/dev-flow:rollback-to design` 回到 Step 1 修改后重跑
4. **范围越界拦截**：Superpowers 出现超范围开发时，自动终止并回退到上一个合规提交点

## 八、常见踩坑与解决方案

| 现象 | 修复方案 |
|------|----------|
| Superpowers 忽略 OpenSpec 设计文档 | 执行 `openspec update` 刷新 schema；确认 tasks.md 任务粒度 ≤ 5 分钟 |
| 任务过大，AI 实现跑偏 | 回到 OpenSpec 执行 `/opsx:audit-tasks`，按建议拆分至单任务 5 分钟内 |
| 归档后全局 spec 未同步 | archive 后手动执行 `/opsx:sync-specs` 合并到根规范 |
| 制品路径分散不统一 | 所有过程文件统一写入 `openspec/changes<迭代名称>/`，覆盖 skill 默认路径 |
| 计划与任务清单不一致 | 执行 `/dev-flow:validate-plan` 查看差异，调整 plan 或补充 tasks 后再继续 |
| TDD 执行不彻底，变成补测试 | 在 schema.yaml 中开启 `enforce_tdd: true`，强制红-绿-重构分步留痕 |
| 多迭代并发出现规范冲突 | 归档前自动冲突检测，冲突项标注后人工合并，禁止强制覆盖全局 spec |

## 九、分级落地建议

- **必选落地（所有项目）**：初始化命令、计划一致性校验、任务粒度审计、结构化 schema、断点续跑、两层必开校验
- **推荐落地（中型以上项目）**：需求澄清节点、四层全量校验、变更分级处理、迭代报告、技术债务隔离
- **可选增强（大型团队/核心项目）**：设计评审门禁、安全扫描、全链路追溯、回滚机制、多迭代冲突检测