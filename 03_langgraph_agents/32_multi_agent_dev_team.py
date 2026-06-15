"""实战项目：多 Agent 软件开发团队协作。

模拟一个完整的软件开发团队，多个 Agent 角色协作完成项目：
1. 产品经理 (PM)：分析需求，编写用户故事
2. 架构师 (Architect)：设计技术方案和接口
3. 开发者 (Developer)：编写代码实现
4. 测试 (Tester)：编写测试用例验证
5. 主管 (Supervisor)：协调分工，汇总结果

对比 14_langgraph_advanced.py 的 Supervisor 模式，本示例更贴近真实开发流程。
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
import operator

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


class DevTeamState(TypedDict):
    requirements: str
    user_stories: str
    architecture: str
    code: str
    test_cases: str
    review_result: str
    current_phase: str
    messages: Annotated[list, operator.add]
    iterations: int


PROMPTS = {
    "pm": ChatPromptTemplate.from_messages([
        ("system", """You are a Product Manager. Given requirements, write user stories.
For each story include: title, description, acceptance criteria, priority.
Output as a numbered list."""),
        ("human", "Requirements: {requirements}"),
    ]),
    "architect": ChatPromptTemplate.from_messages([
        ("system", """You are a Software Architect. Design the technical solution.
Include: tech stack recommendation, component diagram, data flow, API design.
Be specific and actionable."""),
        ("human", "User Stories: {user_stories}"),
    ]),
    "developer": ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Developer. Write clean, well-documented Python code."),
        ("human", "Architecture: {architecture}\n\nWrite the implementation."),
    ]),
    "tester": ChatPromptTemplate.from_messages([
        ("system", "You are a QA Engineer. Write pytest test cases. Include edge cases."),
        ("human", "Code:\n{code}\n\nWrite comprehensive tests."),
    ]),
    "reviewer": ChatPromptTemplate.from_messages([
        ("system", "You are a Tech Lead. Review the code and tests. Identify issues."),
        ("human", "Code:\n{code}\n\nTests:\n{test_cases}\n\nProvide review feedback."),
    ]),
}

chains = {role: prompt | model | StrOutputParser() for role, prompt in PROMPTS.items()}


def pm_agent(state: DevTeamState) -> DevTeamState:
    result = chains["pm"].invoke({"requirements": state["requirements"]})
    return {"user_stories": result, "current_phase": "architect", "messages": [AIMessage(content=f"[PM] {result[:100]}...")]}


def architect_agent(state: DevTeamState) -> DevTeamState:
    result = chains["architect"].invoke({"user_stories": state["user_stories"]})
    return {"architecture": result, "current_phase": "developer", "messages": [AIMessage(content=f"[Architect] {result[:100]}...")]}


def developer_agent(state: DevTeamState) -> DevTeamState:
    result = chains["developer"].invoke({"architecture": state["architecture"]})
    return {"code": result, "current_phase": "tester", "messages": [AIMessage(content=f"[Developer] {result[:100]}...")]}


def tester_agent(state: DevTeamState) -> DevTeamState:
    result = chains["tester"].invoke({"code": state["code"]})
    return {"test_cases": result, "current_phase": "review", "messages": [AIMessage(content=f"[Tester] {result[:100]}...")]}


def reviewer_agent(state: DevTeamState) -> DevTeamState:
    result = chains["reviewer"].invoke({"code": state["code"], "test_cases": state["test_cases"]})
    return {"review_result": result, "current_phase": "complete", "messages": [AIMessage(content=f"[Review] {result[:100]}...")]}


def router(state: DevTeamState) -> str:
    phase = state.get("current_phase", "pm")
    return {"pm": "pm_node", "architect": "architect_node", "developer": "developer_node",
            "tester": "tester_node", "review": "reviewer_node", "complete": "end"}.get(phase, "end")


builder = StateGraph(DevTeamState)
builder.add_node("pm_node", pm_agent)
builder.add_node("architect_node", architect_agent)
builder.add_node("developer_node", developer_agent)
builder.add_node("tester_node", tester_agent)
builder.add_node("reviewer_node", reviewer_agent)
builder.set_entry_point("pm_node")
builder.add_conditional_edges("pm_node", router, {"architect_node": "architect_node"})
builder.add_conditional_edges("architect_node", router, {"developer_node": "developer_node"})
builder.add_conditional_edges("developer_node", router, {"tester_node": "tester_node"})
builder.add_conditional_edges("tester_node", router, {"reviewer_node": "reviewer_node"})
builder.add_conditional_edges("reviewer_node", router, {"end": END})

graph = builder.compile()

print("=" * 60)
print("实战项目：多 Agent 软件开发团队")
print("=" * 60)

result = graph.invoke({
    "requirements": "Build a Python CLI tool that tracks personal expenses. "
                    "Users can add expenses with category and amount, view monthly summaries, "
                    "and export data to CSV. Data should persist in a JSON file.",
    "user_stories": "",
    "architecture": "",
    "code": "",
    "test_cases": "",
    "review_result": "",
    "current_phase": "pm",
    "messages": [],
    "iterations": 0,
})

print(f"\n需求: {result['requirements'][:80]}...")
print(f"\n--- 用户故事 (PM) ---\n{result['user_stories'][:200]}...")
print(f"\n--- 架构设计 (Architect) ---\n{result['architecture'][:200]}...")
print(f"\n--- 代码 (Developer) ---\n{result['code'][:200]}...")
print(f"\n--- 测试用例 (Tester) ---\n{result['test_cases'][:200]}...")
print(f"\n--- 审查结果 (Tech Lead) ---\n{result['review_result'][:200]}...")

print("\n" + "=" * 60)
print("多 Agent 协作的关键设计模式")
print("=" * 60)
print("""
1. 角色分工：每个 Agent 有明确的职责边界
   - PM 只做需求分析，不写代码
   - Developer 只写代码，不做测试
   - Tester 只写测试，不修改代码

2. 信息传递：上游产出是下游的输入
   PM -> Architect -> Developer -> Tester -> Review
   每个阶段都有明确的交付物

3. 可追溯性：每条消息都标注了角色 [PM]/[Architect]/[Developer]
   方便定位哪个环节出了问题

4. 扩展性：可以轻松添加新角色
   比如加 Security Reviewer、DevOps Engineer

5. 对比 14_langgraph_advanced.py：
   - 14 展示的是 Supervisor 派发给多个专家并行执行
   - 本示例展示的是流水线式的多步骤串行协作
   两种模式适用不同场景，可以组合使用
""")
