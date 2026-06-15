"""Plan-and-Execute 与 Reflexion —— 让 Agent 学会规划和反思。

基础 Agent 的局限："想到哪做到哪"，容易迷失在复杂任务中。
Plan-and-Execute + Reflexion 让 Agent 具备"先规划后执行，执行完再反思"的能力。

本文件覆盖 3 个递进模式：
1. Plan-and-Execute：先规划再执行
2. Reflexion：执行后反思修正
3. 完整规划-执行-反思-修正循环
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ==============================================================================
# 示例 1：Plan-and-Execute —— 先规划再执行
# ==============================================================================
print("=== 示例 1: Plan-and-Execute ===")

# Plan-and-Execute 的核心思想：
#   1. Planner（规划器）：LLM 分析任务，拆分为有序子步骤
#   2. Executor（执行器）：按顺序执行每个步骤
#   3. 这种 "先想好再做" 的模式大幅减少中途迷失的概率


class PlanAndExecute:
    """简单的 Plan-and-Execute 实现。"""

    def __init__(self):
        self.planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a planner. Break down the user's task into 3-5 sequential steps.
For each step, specify:
- step_number: the order
- action: what to do
- expected_output: what this step should produce

Return as a numbered list."""),
            ("human", "{task}"),
        ])
        self.executor_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a task executor. Execute the given step and return the result."),
            ("human", "Step {step_num}: {step_description}"),
        ])
        self.planner = self.planner_prompt | model | StrOutputParser()
        self.executor = self.executor_prompt | model | StrOutputParser()

    def run(self, task: str) -> dict:
        # Phase 1: Plan
        print(f"   [规划] 任务: {task}")
        plan = self.planner.invoke({"task": task})
        print(f"   [规划] 结果:\n{plan}\n")

        # Phase 2: Execute
        results = []
        steps = [s.strip() for s in plan.split("\n") if s.strip() and s[0].isdigit()]
        for i, step in enumerate(steps, 1):
            print(f"   [执行] 步骤 {i}...")
            result = self.executor.invoke({
                "step_num": i,
                "step_description": step,
            })
            results.append({"step": step, "result": result})
            print(f"   [执行] 完成: {result[:100]}...\n")

        return {"plan": plan, "results": results}


planner = PlanAndExecute()
result = planner.run("Explain the difference between REST and GraphQL, including pros and cons for each.")
print(f"   规划步骤数: {len(result['results'])}")
print()

print("   Plan-and-Execute 适用场景:")
print("   - 多步骤研究任务（市场分析、竞品调研）")
print("   - 内容创作（写文章、生成报告）")
print("   - 数据分析（数据清洗→分析→可视化）")
print()


# ==============================================================================
# 示例 2：Reflexion —— 执行后反思修正
# ==============================================================================
print("=== 示例 2: Reflexion 模式 ===")

# Reflexion 的核心思想：
#   1. Generate（生成）：模型生成输出
#   2. Reflect（反思）：模型评估自己的输出
#   3. Revise（修正）：基于反思改进输出
# 循环直到评估通过或达到最大迭代次数


class Reflexion:
    """简单的 Reflexion 实现。"""

    def __init__(self, max_iterations=3):
        self.generator_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a content creator. Produce high-quality output."),
            ("human", "{task}"),
        ])
        self.critic_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critical reviewer. Evaluate the output and identify:
1. What's good
2. What's missing or could be improved
3. Specific, actionable feedback

Be honest and constructive."""),
            ("human", "Task: {task}\n\nOutput:\n{output}"),
        ])
        self.reviser_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a content reviser. Improve the output based on the feedback."),
            ("human", "Task: {task}\n\nPrevious output:\n{output}\n\nFeedback:\n{feedback}\n\nPlease provide an improved version."),
        ])
        self.generator = self.generator_prompt | model | StrOutputParser()
        self.critic = self.critic_prompt | model | StrOutputParser()
        self.reviser = self.reviser_prompt | model | StrOutputParser()
        self.max_iterations = max_iterations

    def run(self, task: str) -> dict:
        iterations = []

        # Step 1: Generate
        print(f"   [生成] 任务: {task}")
        output = self.generator.invoke({"task": task})
        iterations.append({"phase": "generate", "output": output})
        print(f"   [生成] 完成: {len(output)} 字符\n")

        for i in range(self.max_iterations):
            # Step 2: Reflect
            feedback = self.critic.invoke({
                "task": task,
                "output": output,
            })
            print(f"   [反思 #{i+1}] 反馈: {feedback[:100]}...\n")

            # Check if output is good enough
            if "no issues" in feedback.lower() or "looks good" in feedback.lower() or "excellent" in feedback.lower():
                print(f"   [反思] 输出已达标，停止迭代\n")
                iterations.append({"phase": "reflect", "feedback": feedback})
                break

            iterations.append({"phase": "reflect", "feedback": feedback})

            # Step 3: Revise
            output = self.reviser.invoke({
                "task": task,
                "output": output,
                "feedback": feedback,
            })
            iterations.append({"phase": "revise", "output": output})
            print(f"   [修正 #{i+1}] 完成: {len(output)} 字符\n")

        return {"iterations": iterations, "final_output": output}


reflexion = Reflexion(max_iterations=2)
result = reflexion.run("Write a clear one-paragraph explanation of what a vector database is, suitable for beginners.")
print(f"   总迭代数: {len(result['iterations'])} 步")
print(f"   最终输出: {result['final_output'][:200]}...\n")

print("   Reflexion 适用场景:")
print("   - 代码生成（生成→测试→修改）")
print("   - 写作任务（初稿→审阅→修改）")
print("   - 翻译质量优化")
print("   - 需要高质量输出的场景")
print()


# ==============================================================================
# 示例 3：完整规划-执行-反思 循环（Plan → Execute → Reflect → Revise）
# ==============================================================================
print("=== 示例 3: 完整循环（Plan → Execute → Reflect → Revise）===")

# 合并 Plan-and-Execute + Reflexion 的完整循环:
#   规划 → 执行步骤1 → 检查结果 → 修正 → 执行步骤2 → 检查 → 修正 → ...

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
import operator


class PlanReflectState(TypedDict):
    task: str
    plan: str
    current_step: int
    total_steps: int
    results: Annotated[list, operator.add]
    feedback: str
    iteration: int


# 规划节点
def plan_task(state: PlanReflectState) -> PlanReflectState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Break this task into 3 numbered steps."),
        ("human", "{task}"),
    ])
    chain = prompt | model | StrOutputParser()
    plan = chain.invoke({"task": state["task"]})
    steps = [s.strip() for s in plan.split("\n") if s.strip() and s[0].isdigit()]
    return {
        "plan": plan,
        "total_steps": len(steps),
        "current_step": 0,
    }


# 执行节点
def execute_step(state: PlanReflectState) -> PlanReflectState:
    steps = [s.strip() for s in state["plan"].split("\n") if s.strip() and s[0].isdigit()]
    step_idx = state["current_step"]

    if step_idx >= len(steps):
        return {"current_step": state["current_step"] + 1}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Execute this step thoroughly."),
        ("human", "Task: {task}\nStep: {step}"),
    ])
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({
        "task": state["task"],
        "step": steps[step_idx],
    })

    return {
        "current_step": state["current_step"] + 1,
        "results": [{"step": steps[step_idx], "output": result}],
    }


# 反思节点
def reflect_on_step(state: PlanReflectState) -> PlanReflectState:
    last_result = state["results"][-1] if state["results"] else {}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate the quality of this step's output. Identify any issues or areas for improvement."),
        ("human", "Task: {task}\nStep output: {output}"),
    ])
    chain = prompt | model | StrOutputParser()
    feedback = chain.invoke({
        "task": state["task"],
        "output": last_result.get("output", ""),
    })
    return {"feedback": feedback}


# 路由：需要修正还是继续下一步？
def should_continue(state: PlanReflectState) -> str:
    if state["current_step"] >= state["total_steps"]:
        return "finish"

    # 检查反馈质量 （简单启发式）
    feedback = state.get("feedback", "")
    if "error" in feedback.lower() or "incorrect" in feedback.lower() or "missing" in feedback.lower():
        return "revise"
    return "next"


# 构建图
builder = StateGraph(PlanReflectState)
builder.add_node("planner", plan_task)
builder.add_node("executor", execute_step)
builder.add_node("reflector", reflect_on_step)

builder.set_entry_point("planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "reflector")
builder.add_conditional_edges(
    "reflector",
    should_continue,
    {"next": "executor", "revise": "executor", "finish": END},
)

graph = builder.compile()

# 运行
result = graph.invoke({
    "task": "Explain the concept of 'dimensionality reduction' in machine learning, including PCA and t-SNE.",
    "plan": "",
    "current_step": 0,
    "total_steps": 0,
    "results": [],
    "feedback": "",
    "iteration": 0,
})

print(f"   总步骤数: {len(result['results'])}")
for i, r in enumerate(result["results"]):
    print(f"   步骤 {i+1}: {r['output'][:100]}...")
print()

print("   完整循环 vs 单独 Plan-and-Execute:")
print("   - 有 Reflexion：每步执行后检查质量，发现问题即时修正")
print("   - 无 Reflexion：执行完所有步骤才发现问题，返工成本高")
print()


# ==============================================================================
# 教学备注：Plan-and-Execute + Reflexion —— 从"试错"到"设计"
# ==============================================================================
# 核心问题：基础 Agent 是"反应式"的，接到任务就直接开干
#   遇到复杂任务容易：做到一半忘了上下文、陷进死胡同、输出质量不稳定
#
# Plan-and-Execute 解决"方向问题"：
#   先把任务想清楚再动手，像写代码前画架构图
#   规划结果可以给人审查，确保方向正确
#
# Reflexion 解决"质量问题"：
#   每次输出后自我复盘，找出问题并改进
#   像 Code Review：自己写完代码自己审查一遍
#
# 三种模式的演进关系：
#
#   基础 Agent（反应式）
#     → 输入 → 直接输出
#     → 问题：复杂任务迷失方向
#     ↓
#   Plan-and-Execute（规划式）
#     → 规划 → 执行步骤1 → 执行步骤2 → ...
#     → 问题：如果某步执行错误，后面全错
#     ↓
#   Reflexion（反思式）
#     → 生成 → 反思 → 修正 → 反思 → ...
#     → 问题：没有全局规划，可能反复修正局部
#     ↓
#   完整循环（规划 + 反思）
#     → 规划 → 执行 → 反思 → 修正 → 下一步 → ...
#     → 最佳方案：全局有规划，局部有反思
#
# 跟 LangGraph 的关系：
#   本文件用简单代码实现 Plan-and-Execute 和 Reflexion 的核心思想
#   LangGraph 可以更优雅地实现这些模式（状态管理、循环、条件路由）
#   参考 03_langgraph_agents/10_langgraph_basics.py 和 03_langgraph_agents/14_langgraph_advanced.py
#
# 何时需要用这些模式：
#   - 任务需要 3 步以上 → 用 Plan-and-Execute
#   - 输出质量要求高 → 用 Reflexion
#   - 又长又复杂 → 组合使用
#   - 简单任务（一步问答）→ 基础 Agent 就够了，不要过度设计
