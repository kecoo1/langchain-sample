"""LangGraph 进阶示例 —— 多 Agent 协作、Supervisor 模式、人类审批工作流。

本文件展示 LangGraph 在生产级应用中的高级用法：
1. Supervisor 模式：一个 Agent 协调多个专家 Agent
2. 人类审批工作流：在关键步骤暂停等待人工确认
3. 嵌套子图：复杂任务拆分为子流程
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 示例 1：Supervisor 模式 —— 多专家 Agent 协作
# ==============================================================================
print("=== 示例 1: Supervisor 模式 ===")


@tool
def search_code(query: str) -> str:
    """Search for code patterns and best practices."""
    return f"Found: Using dependency injection in {query}. See pattern at line 42."


@tool
def search_security(vulnerability: str) -> str:
    """Check for security vulnerabilities."""
    return f"Found 2 potential SQL injection risks in {vulnerability}."


@tool
def search_performance(issue: str) -> str:
    """Analyze performance issues."""
    return f"N+1 query pattern detected in {issue}. Suggested fix: eager loading."


@tool
def search_deployment(platform: str) -> str:
    """Check deployment configuration."""
    return f"Dockerfile multi-stage build recommended for {platform}."


# 每个专家 Agent 的 State
class ExpertState(TypedDict):
    messages: Annotated[list, operator.add]
    task: str
    expert: str


def code_expert(state: ExpertState) -> ExpertState:
    response = model.invoke([
        HumanMessage(content=f"You are a senior code reviewer. Review this task: {state['task']}")
    ])
    return {"messages": [AIMessage(content=f"[Code Expert]: {response.content}")]}


def security_expert(state: ExpertState) -> ExpertState:
    response = model.invoke([
        HumanMessage(content=f"You are a security engineer. Review this task: {state['task']}")
    ])
    return {"messages": [AIMessage(content=f"[Security Expert]: {response.content}")]}


def performance_expert(state: ExpertState) -> ExpertState:
    response = model.invoke([
        HumanMessage(content=f"You are a performance engineer. Review this task: {state['task']}")
    ])
    return {"messages": [AIMessage(content=f"[Performance Expert]: {response.content}")]}


# Supervisor 的 State
class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    tasks: list[str]
    next_expert: str


# Supervisor 决定调用哪个专家
def supervisor_router(state: SupervisorState) -> str:
    """Supervisor 根据任务内容决定派发给哪个专家。"""
    last_message = state["messages"][-1].content if state["messages"] else ""
    task = state["tasks"][0] if state["tasks"] else ""

    # 简单规则路由（生产环境中可以用 LLM 做路由决策）
    task_lower = (task + " " + last_message).lower()
    if "security" in task_lower or "vuln" in task_lower or "injection" in task_lower:
        return "security"
    elif "performance" in task_lower or "slow" in task_lower or "optim" in task_lower:
        return "performance"
    else:
        return "code"


# 构建专家图
expert_builder = StateGraph(ExpertState)
expert_builder.add_node("code", code_expert)
expert_builder.add_node("security", security_expert)
expert_builder.add_node("performance", performance_expert)
expert_builder.add_edge("code", END)
expert_builder.add_edge("security", END)
expert_builder.add_edge("performance", END)
expert_graph = expert_builder.compile()


# Supervisor 图
def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor 分析任务并决定调用哪个专家。"""
    task = state["tasks"][0] if state["tasks"] else "Review code quality"

    # 获取专家结果
    expert_state = ExpertState(task=task, expert="code", messages=[])
    result = expert_graph.invoke(expert_state)
    expert_result = result["messages"][-1].content if result["messages"] else ""

    return {
        "messages": [AIMessage(content=f"Supervisor: {expert_result}")],
        "next_expert": "code",
    }


supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor_node)
supervisor_builder.add_edge("supervisor", END)
supervisor_builder.set_entry_point("supervisor")
supervisor_graph = supervisor_builder.compile()

result = supervisor_graph.invoke({
    "messages": [],
    "tasks": ["Check for security vulnerabilities in authentication module"],
    "next_expert": "code",
})
print(f"1. Supervisor 结果: {result['messages'][-1].content}\n")


# ==============================================================================
# 示例 2：人类审批工作流 —— interrupt() 暂停等待确认
# ==============================================================================
print("=== 示例 2: 人类审批工作流 ===")


@tool
def deploy_to_production(app_name: str, environment: str = "production") -> str:
    """Deploy application to production environment."""
    return f"Deployment of {app_name} to {environment} initiated."


class DeployState(TypedDict):
    messages: Annotated[list, operator.add]
    deploy_requested: bool
    approved: bool | None
    app_name: str


def analyze_deploy(state: DeployState) -> DeployState:
    """分析部署请求并生成报告。"""
    report = model.invoke([
        HumanMessage(content=f"""
        Analyze the deployment request for: {state['app_name']}
        Provide:
        1. Risk level (low/medium/high)
        2. Recommended rollback strategy
        3. Pre-deployment checklist items
        """)
    ])
    return {
        "messages": [AIMessage(content=f"Deployment Analysis:\n{report.content}")],
        "deploy_requested": True,
    }


def request_human_approval(state: DeployState) -> DeployState:
    """暂停等待人类审批。"""
    from langgraph.types import interrupt

    approval = interrupt({
        "type": "deployment_approval",
        "question": "Should deployment proceed?",
        "risk_analysis": state["messages"][-1].content if state["messages"] else "",
        "app_name": state["app_name"],
    })

    return {
        "approved": approval.get("approved", False),
        "messages": [AIMessage(content=f"Approval decision: {'APPROVED' if approval.get('approved') else 'REJECTED'}")],
    }


def execute_deploy(state: DeployState) -> DeployState:
    """执行部署。"""
    result = deploy_to_invoke.invoke({"app_name": state["app_name"]})
    return {
        "messages": [AIMessage(content=f"Deployment result: {result.content}")],
    }


def should_approve(state: DeployState) -> Literal["approve", "reject"]:
    if state["approved"] is None:
        return "approve"
    elif state["approved"]:
        return "approve"
    else:
        return "reject"


deploy_builder = StateGraph(DeployState)
deploy_builder.add_node("analyze", analyze_deploy)
deploy_builder.add_node("approve", request_human_approval)
deploy_builder.add_node("execute", execute_deploy)

deploy_builder.add_edge(START, "analyze")
deploy_builder.add_edge("analyze", "approve")
deploy_builder.add_conditional_edges(
    "approve",
    should_approve,
    {"approve": "execute", "reject": END},
)
deploy_builder.add_edge("execute", END)

# 注意：interrupt() 需要 checkpointer
deploy_memory = MemorySaver()
deploy_graph = deploy_builder.compile(checkpointer=deploy_memory)

# 演示：第一阶段——分析和请求审批
# config = {"configurable": {"thread_id": "deploy_demo"}}
# result = deploy_graph.invoke(
#     {"messages": [], "deploy_requested": False, "approved": None, "app_name": "my-app"},
#     config
# )
# # 此时会暂停，等待 interrupt 返回值
# # 然后 resume:
# # result = deploy_graph.invoke(None, config, resume={"approved": True})

# print("2. 人类审批工作流（注释，需要实际审批输入）\n")


# ==============================================================================
# 示例 3：嵌套子图 —— 复杂分析流程
# ==============================================================================
print("=== 示例 3: 嵌套子图 ===")


# 子图 1：代码分析
class CodeAnalysisState(TypedDict):
    code: str
    issues: list[str]
    rating: int


def analyze_syntax(state: CodeAnalysisState) -> CodeAnalysisState:
    result = model.invoke([
        HumanMessage(content=f"Analyze syntax and style: {state['code']}")
    ])
    return {
        "issues": [f"Syntax: {result.content[:200]}"],
        "rating": 8,
    }


def check_best_practices(state: CodeAnalysisState) -> CodeAnalysisState:
    result = model.invoke([
        HumanMessage(content=f"Check best practices: {state['code']}")
    ])
    return {
        "issues": state["issues"] + [f"Best practices: {result.content[:200]}"],
    }


code_analysis_builder = StateGraph(CodeAnalysisState)
code_analysis_builder.add_node("syntax", analyze_syntax)
code_analysis_builder.add_node("practices", check_best_practices)
code_analysis_builder.add_edge(START, "syntax")
code_analysis_builder.add_edge("syntax", "practices")
code_analysis_builder.add_edge("practices", END)
code_analysis_graph = code_analysis_builder.compile()


# 子图 2：安全审计
class SecurityAuditState(TypedDict):
    code: str
    vulns: list[str]
    risk_level: str


def scan_vulnerabilities(state: SecurityAuditState) -> SecurityAuditState:
    result = model.invoke([
        HumanMessage(content=f"Scan for vulnerabilities: {state['code']}")
    ])
    return {"vulns": [result.content[:300]], "risk_level": "medium"}


def check_dependencies(state: SecurityAuditState) -> SecurityAuditState:
    result = model.invoke([
        HumanMessage(content=f"Check dependency security for: {state['code']}")
    ])
    return {"vulns": state["vulns"] + [result.content[:300]]}


security_builder = StateGraph(SecurityAuditState)
security_builder.add_node("scan", scan_vulnerabilities)
security_builder.add_node("deps", check_dependencies)
security_builder.add_edge(START, "scan")
security_builder.add_edge("scan", "deps")
security_builder.add_edge("deps", END)
security_graph = security_builder.compile()


# 主图：编排子图
class MainReviewState(TypedDict):
    code: str
    code_analysis: dict
    security_audit: dict
    final_report: str


def run_code_analysis(state: MainReviewState) -> MainReviewState:
    result = code_analysis_graph.invoke({"code": state["code"], "issues": [], "rating": 0})
    return {"code_analysis": result}


def run_security_audit(state: MainReviewState) -> MainReviewState:
    result = security_graph.invoke({"code": state["code"], "vulns": [], "risk_level": ""})
    return {"security_audit": result}


def generate_report(state: MainReviewState) -> MainReviewState:
    analysis = state["code_analysis"]
    audit = state["security_audit"]
    report = model.invoke([
        HumanMessage(content=f"""
        Generate a comprehensive code review report:
        Code Analysis: {analysis}
        Security Audit: {audit}
        """)
    ])
    return {"final_report": report.content}


main_builder = StateGraph(MainReviewState)
main_builder.add_node("analyze_code", run_code_analysis)
main_builder.add_node("security", run_security_audit)
main_builder.add_node("report", generate_report)

main_builder.add_edge(START, "analyze_code")
main_builder.add_edge(START, "security")  # 并行执行
main_builder.add_edge("analyze_code", "report")
main_builder.add_edge("security", "report")
main_builder.add_edge("report", END)

main_graph = main_builder.compile()

# 演示
sample_code = """
def process_payment(user_id, amount, card_number):
    charge_stripe(card_number, amount)
    db.save_transaction(user_id, amount)
    return True
"""

result = main_graph.invoke({"code": sample_code, "code_analysis": {}, "security_audit": {}, "final_report": ""})
print(f"3. 综合审查报告: {result['final_report'][:300]}...\n")


# ==============================================================================
# 示例 4：条件路由 + 重试机制
# ==============================================================================
print("=== 示例 4: 条件路由 + 重试 ===")


class RetryState(TypedDict):
    messages: Annotated[list, operator.add]
    attempt: int
    max_attempts: int
    success: bool


def attempt_response(state: RetryState) -> RetryState:
    """尝试生成响应，随机成功/失败。"""
    response = model.invoke(state["messages"])
    # 模拟：有时生成不完整
    content = response.content
    success = len(content) > 10  # 简单判定
    return {
        "messages": [AIMessage(content=content)],
        "success": success,
        "attempt": state["attempt"] + 1,
    }


def should_retry(state: RetryState) -> Literal["retry", "done", "fail"]:
    if state["success"]:
        return "done"
    elif state["attempt"] < state["max_attempts"]:
        return "retry"
    else:
        return "fail"


retry_builder = StateGraph(RetryState)
retry_builder.add_node("attempt", attempt_response)
retry_builder.add_conditional_edges(
    "attempt",
    should_retry,
    {"retry": "attempt", "done": END, "fail": END},
)
retry_graph = retry_builder.compile()

final_state = retry_graph.invoke({
    "messages": [HumanMessage(content="Write a short poem about programming")],
    "attempt": 0,
    "max_attempts": 3,
    "success": False,
})
print(f"4. 重试机制结果: {final_state['messages'][-1].content[:200]}...\n")


# ==============================================================================
# 教学备注：LangGraph 高级模式 —— 从"简单循环"到"生产级编排"
# ==============================================================================
# 核心概念 1：Supervisor 模式
#   是什么：一个"指挥官"Agent 协调多个"专家"Agent 分工协作
#   为什么需要：复杂任务拆分为子任务，每个子任务由最擅长的专家处理
#   类比：软件开发中，代码质量由 Code Reviewer 审、安全由 Security Engineer 审、性能由 Performance Engineer 审
#   实现方式：
#     - Supervisor 接收任务 -> LLM 路由决策 -> 调用对应专家子图
#     - 所有专家的结果汇总给 Supervisor 做最终整合
#   生产场景：客户支持（技术/账单/退货分流）、内容审核（安全/合规/质量分流）
#
# 核心概念 2：interrupt() —— 人类在环（Human-in-the-loop）
#   是什么：在图执行过程中暂停，等待外部输入后才继续
#   为什么需要：某些决策不能全交给 AI，需要人类审批（资金操作、删除数据、公开发布）
#   实现机制：
#     interrupt({...}) 返回 None -> 图暂停 -> 状态保存到 checkpointer
#     外部系统调用 .invoke(None, config, resume={...}) 恢复执行
#   类比：Git merge 冲突时需要你手动解决；CI/CD 流水线需要人工审批才能部署
#   关键：interrupt 让图变成"事件驱动"而非"一次性运行"
#
# 核心概念 3：嵌套子图（Subgraph）
#   是什么：一个图可以包含另一个图作为节点
#   为什么需要：复杂流程拆分为独立子流程，每个子流程可独立测试和复用
#   类比：函数调用——main 函数调用 analyze_code() 和 security_audit()
#   关键优势：
#     - 子图可以独立编译、测试、版本管理
#     - 子图可以并行执行（如示例 3 中 code_analysis 和 security_audit 并行）
#     - 子图结果结构化传递，主图只做编排
#
# 核心概念 4：条件路由 + 重试
#   是什么：根据运行时状态决定下一步，支持自动重试和失败降级
#   为什么需要：LLM 输出不完全稳定，某些调用可能失败
#   模式：
#     attempt -> 判断成功? -> 成功则 done / 失败则 retry / 超次数则 fail
#   生产价值：提高系统鲁棒性，对最终用户透明
#
# 从基础到进阶的认知跃迁:
#   03_langgraph_agents/10_langgraph_basics.py: 状态机 + 简单循环
#   本文件: 子图编排 + 人类审批 + 条件路由
#   进阶方向:
#     - 持久化存储（Redis/Postgres checkpointer 替代 MemorySaver）
#     - 分布式执行（LangGraph Platform）
#     - 实时监控（.stream() 逐节点观察）
#     - 调试可视化（.get_graph().draw_mermaid()）
#
# 教学建议顺序:
#   1. 先跑通示例 1（Supervisor），理解"多 Agent 协作"模式
#   2. 理解示例 2（人类审批）的概念——interrupt() 是 LangGraph 最强大的功能之一
#   3. 再跑通示例 3（嵌套子图），理解"子图编排"
#   4. 最后理解示例 4（重试机制），掌握鲁棒性设计
