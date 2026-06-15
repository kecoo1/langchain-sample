"""实战项目：智能客服系统（分类 + 转人工）。

一个生产级的客服系统需要：意图分类、多 Specialist Agent、人工转接。
本示例用 LangGraph 实现完整的客服流程：
1. 接待 Agent：初步回应 + 分类
2. 技术 Agent：处理技术问题
3. 账单 Agent：处理账单和退款
4. 售后 Agent：处理退货和投诉
5. 人工转接：复杂情况交给人处理（interrupt 模拟）
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
import operator

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPTS = {
    "router": """You are a customer service router. Classify the user's issue into one category:
- technical: login issues, bugs, error messages, app crashes
- billing: charges, invoices, refunds, payment methods
- after_sale: returns, exchanges, complaints, shipping
- simple: general questions, how-to guides, feature requests (answer directly)
- human: complex issues, angry customers, account security, anything sensitive

Respond with ONLY the category name.""",

    "technical": """You are a Technical Support Agent.
Be patient and step-by-step. Ask clarifying questions before suggesting fixes.
If the issue requires account access or seems too complex, advise contacting human support.""",

    "billing": """You are a Billing Agent.
Handle charges, refunds, invoices professionally.
Be transparent about policies. Never promise refunds outside policy.""",

    "after_sale": """You are an After-Sales Agent.
Handle returns, exchanges, complaints with empathy.
Follow company policy strictly. Escalate valid complaints to human team.""",
}


def create_agent(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    return prompt | model | StrOutputParser()


router_agent = create_agent(SYSTEM_PROMPTS["router"])
tech_agent = create_agent(SYSTEM_PROMPTS["technical"])
billing_agent = create_agent(SYSTEM_PROMPTS["billing"])
after_sale_agent = create_agent(SYSTEM_PROMPTS["after_sale"])


class CustomerServiceState(TypedDict):
    messages: Annotated[list, operator.add]
    category: str
    user_input: str
    response: str
    needs_human: bool


def classify(state: CustomerServiceState) -> CustomerServiceState:
    category = router_agent.invoke({"input": state["user_input"]}).strip().lower()
    return {"category": category, "messages": [AIMessage(content=f"[Router] Category: {category}")]}


def handle_technical(state: CustomerServiceState) -> CustomerServiceState:
    response = tech_agent.invoke({"input": state["user_input"]})
    return {"response": response, "messages": [AIMessage(content=f"[Tech] {response[:100]}...")]}


def handle_billing(state: CustomerServiceState) -> CustomerServiceState:
    response = billing_agent.invoke({"input": state["user_input"]})
    return {"response": response, "messages": [AIMessage(content=f"[Billing] {response[:100]}...")]}


def handle_after_sale(state: CustomerServiceState) -> CustomerServiceState:
    response = after_sale_agent.invoke({"input": state["user_input"]})
    return {"response": response, "messages": [AIMessage(content=f"[AfterSale] {response[:100]}...")]}


def route_after_classify(state: CustomerServiceState) -> str:
    """路由函数：根据分类结果决定下一步去哪个节点。

    这是一个"条件路由"——每一步该往哪走，取决于上一步的输出。
    LangGraph 的 StateGraph 本身是"图"结构，边走边决定下一步。

    参数 state: 当前状态，包含上一步（classify 节点）产出的 category。
    返回值 str: 路由表的 key，LangGraph 用这个 key 查表找到下一个节点。
    """
    cat = state["category"]
    if cat == "human":
        return "human"
    # .get(cat, "simple") 的含义：
    #   - 如果 cat 在字典里 → 返回对应的 value（technical/ billing/ after_sale）
    #   - 如果 cat 不在字典里（比如意料之外的分类）→ 返回默认值 "simple"
    #   这是一种安全的兜底策略：分类器可能出错，但不会导致系统崩溃
    return {"technical": "technical", "billing": "billing", "after_sale": "after_sale",
            "simple": "simple"}.get(cat, "simple")


# ─── 构建状态图 ──────────────────────────────────────────────
builder = StateGraph(CustomerServiceState)
builder.add_node("classifier", classify)
builder.add_node("technical", handle_technical)
builder.add_node("billing", handle_billing)
builder.add_node("after_sale", handle_after_sale)

# ─── 条件路由：分类器之后的路由表 ──────────────────────────────
# 格式：add_conditional_edges(起点, 路由函数, {返回值→下一个节点})
#
# 这个路由表的含义：
#   分类器输出 → route_after_classify 返回 → 路由表决定去向
#
#   "technical"  → technical 节点  （技术Agent处理）
#   "billing"    → billing 节点    （账单Agent处理）
#   "after_sale" → after_sale 节点 （售后Agent处理）
#   "simple"     → END（结束）      （简单问题，已在分类节点直接回答）
#   "human"      → END（结束）      （需要转人工，本示例简单展示）
#
# 类比快递分拣：
#   classifier = 扫码员（扫包裹上的地址）
#   route_after_classify = 分拣逻辑（"这个去北京"）
#   路由表 = 传送带出口（北京口→北京车，上海口→上海车）
#   END = 特殊包裹（查无此地→退回，敏感件→专人处理）
builder.set_entry_point("classifier")
builder.add_conditional_edges("classifier", route_after_classify, {
    "technical": "technical", "billing": "billing", "after_sale": "after_sale",
    "simple": END, "human": END,
})
builder.add_edge("technical", END)
builder.add_edge("billing", END)
builder.add_edge("after_sale", END)

graph = builder.compile()

print("=" * 60)
print("实战项目：智能客服系统")
print("=" * 60)

test_cases = [
    ("技术问题", "I can't log in to my account. It says 'invalid credentials' even though I'm sure my password is correct."),
    ("账单问题", "I was charged twice for my subscription this month. Can you help me get a refund for the duplicate charge?"),
    ("售后问题", "The product I received is damaged. The screen has a crack. I want to return it and get a replacement."),
    ("简单问题", "How do I reset my password?"),
    ("需转人工", "This is unacceptable! I've been waiting for 3 weeks and nobody helped me. I want to speak to a manager right now!"),
]

for scenario, user_input in test_cases:
    print(f"\n--- [{scenario}] ---")
    print(f"用户: {user_input[:60]}...")
    result = graph.invoke({
        "messages": [],
        "category": "",
        "user_input": user_input,
        "response": "",
        "needs_human": False,
    })
    cat = result.get("category", "unknown")
    if cat == "human":
        print(f"   → 分类: human")
        print(f"   → ⚠️  转人工处理（复杂/敏感请求）")
    else:
        print(f"   → 分类: {cat}")
        print(f"   → 回复: {result.get('response', '')[:120]}...")

print("\n" + "=" * 60)
print("智能客服架构要点")
print("=" * 60)
print("""
1. 三级处理架构：
   第一级：Router 分类（快速，低成本模型）
   第二级：Specialist Agent 处理（中等成本）
   第三级：人工客服（高成本，仅复杂情况）

2. 意图分类的关键：
   - 分错类 = 后面全错
   - 可以用专用分类模型（更便宜、更准）
   - "human" 类必须保守：不确定就转人工

3. 转人工设计模式：
   本示例用 route 条件边实现
   生产系统用 interrupt() + 工单系统：
   - LangGraph interrupt → 暂停等待人工
   - 人工处理后 resume 继续
   参考 14_langgraph_advanced.py 的 human-in-loop

4. 对比传统客服机器人：
   传统：关键词匹配 + 固定流程
   Agent：理解意图 + 灵活应对 + 可转人工
   两者可以混合使用，Rule-based 兜底
""")

# ==============================================================================
# 扩展：带人工审批的数据查询（演示中断模式）
# ==============================================================================
print("\n" + "=" * 60)
print("扩展：人工审批流程（interrupt 模式）")
print("=" * 60)


@tool
def query_user_account(user_id: str) -> str:
    """Query user account details.
    
    Args:
        user_id: The user's ID
    """
    accounts = {"U001": "Alice, Premium, $49.99/mo", "U002": "Bob, Standard, $19.99/mo"}
    return accounts.get(user_id, "Account not found")


@tool
def process_refund(user_id: str, amount: float, reason: str) -> str:
    """Process a refund for a user.
    
    Args:
        user_id: User ID
        amount: Refund amount
        reason: Reason for refund
    """
    return f"Refund of ${amount} for {user_id} initiated. Reason: {reason}"


print("""
在真实生产系统中，敏感操作（退款、删除数据、修改权限）需要人工审批。
使用 LangGraph 的 interrupt() 实现：

1. Agent 发起退款请求 → 自动触发 interrupt
2. 系统暂停等待人工审批
3. 人工审核后在 Dashboard 点击"批准"或"拒绝"
4. Agent 收到审批结果后继续执行

关键代码模式：
   from langgraph.types import interrupt
  
   def refund_node(state):
       decision = interrupt({
           "user_id": state["user_id"],
           "amount": state["amount"],
           "reason": state["reason"],
           "question": "Approve this refund?",
       })
       if decision["approved"]:
           return {"status": "refunded"}
       return {"status": "rejected"}

优势：
- 关键操作不会自动执行
- 每笔操作都有审批记录（审计日志）
- 审批延迟用户可接受（总比自动退款犯错好）
- 可以参考 03_langgraph_agents/14_langgraph_advanced.py
""")
