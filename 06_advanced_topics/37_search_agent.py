"""搜索 Agent — LLM 驱动的端到端搜索与问答 Agent

在 36_search_enhancement.py 的混合搜索之上增加 LLM 决策层：
  1. LLM 理解用户问题 → 生成搜索关键词
  2. 混合搜索（BM25 + 向量）
  3. LLM 评估结果是否足够
  4. 不足则换关键词重搜（最多 3 轮）
  5. 足够后 LLM 生成自然语言回答

使用 LangGraph StateGraph 编排，流程清晰可追踪。

模型要求：使用默认的 llama3.2:1b 时，LLM 在「查询改写」
和「结果评估」环节的指令遵循能力有限，容易输出元解释
而非结构化决策。示例已内置多级 fallback 确保不崩溃。
建议切换至 gpt-4o-mini 或 qwen2.5:7b 以获得更好的决策效果。

运行：python 06_advanced_topics/37_search_agent.py
"""

import importlib
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# 模块名以数字开头，无法用 from/import 语法导入
_search_mod = importlib.import_module("36_search_enhancement")
SearchEngine = _search_mod.SearchEngine
create_mock_documents = _search_mod.create_mock_documents
UnifiedDocument = _search_mod.UnifiedDocument

LLM_MODEL = "llama3.2:1b"
MAX_ATTEMPTS = 3


class SearchAgentState(TypedDict):
    query: str
    keywords: list[str]
    all_results: list[UnifiedDocument]
    attempt: int
    max_attempts: int
    decision: str
    answer: str


# ==============================================================================
# Teaching Notes: 搜索引擎与 LLM 的全局初始化
# ==============================================================================
# SearchEngine 内部包含 ChromaDB 向量索引和 BM25 索引，初始化较慢，
# 因此在模块加载时一次性初始化，后续节点函数直接复用。
#
# LLM 使用 Ollama 本地部署的 llama3.2:1b，注意小模型的工具调用
# 和指令遵循能力有限，生产环境建议 GPT-4o-mini 或 Claude。

engine = SearchEngine(create_mock_documents())
llm = ChatOllama(model=LLM_MODEL, temperature=0)


# ==============================================================================
# Teaching Notes: 节点 1 — 查询理解
# ==============================================================================
# 用户原始问题往往不适合直接作为搜索关键词。
# 例如「东西坏了怎么退钱」应该拆解为关键词。
# LLM 在此扮演查询理解引擎的角色。

def decide_query(state: SearchAgentState) -> dict:
    """LLM 根据用户问题生成 1-3 个搜索关键词

    llama3.2:1b 对开放式指令容易生成元解释而非关键词，
    使用 JSON 格式约束 + 多级 fallback。
    """
    prompt = f"""Generate 2 search keywords for this question. Return ONLY a JSON array: ["kw1", "kw2"]
Question: {state['query']}"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    content = resp.content.strip()

    # 尝试解析 JSON 数组
    import json as _json
    kws = None
    try:
        # 从文本中提取 [...] 内容
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            parsed = _json.loads(content[start:end+1])
            kws = [str(k).strip() for k in parsed if isinstance(k, (str,))]
    except Exception:
        pass

    # fallback: 按行解析，过滤噪音
    if not kws:
        kws = []
        for line in content.split("\n"):
            s = line.strip().lstrip("-*").strip()
            if not s or s.startswith(("{", "[", "```")):
                continue
            if s.startswith(("JSON", "Output", "Here", "Sure", "Question")):
                continue
            if len(s) > 40 or len(s) < 2:
                continue
            kws.append(s)

    # 质量检查：LLM 可能输出英文拒绝/无关内容，fallback 到原始查询
    import re as _re
    _zh = _re.findall(r"[\u4e00-\u9fff]", state["query"])
    if _zh:
        _zh_set = set(_zh)
        valid = [kw for kw in kws if set(_re.findall(r"[\u4e00-\u9fff]", kw)) & _zh_set]
        kws = valid[:3] if valid else [state["query"]]
    print(f"  LLM 生成关键词: {kws}")
    return {"keywords": kws}


# ==============================================================================
# Teaching Notes: 节点 2 — 混合搜索
# ==============================================================================
# 对每个关键词逐一调用 SearchEngine，返回的结果去重合并。
# SearchEngine 内部已经是 BM25 + 向量双路 RRF 融合了。

def search_hybrid(state: SearchAgentState) -> dict:
    """遍历关键词执行混合搜索，去重合并"""
    seen = {d.id for d in state["all_results"]}
    new_results = []

    for kw in state["keywords"]:
        try:
            hits = engine.search(kw, top_k=3)
            new_docs = [d for d in hits.hits if d.id not in seen]
            for d in new_docs:
                seen.add(d.id)
            new_results.extend(new_docs)
            if new_docs:
                print(f"  「{kw}」→ {len(new_docs)} 条新结果")
        except Exception as e:
            print(f"  「{kw}」搜索异常: {e}")

    return {
        "all_results": state["all_results"] + new_results,
        "attempt": state["attempt"] + 1,
    }


# ==============================================================================
# Teaching Notes: 节点 3 — 结果评估
# ==============================================================================
# LLM 判断已有搜索结果是否足够回答用户问题。
# 如果不够，还给出一组新的搜索关键词供下一轮使用。
# 这是 Agent 的核心智能点：自主决定「够不够、还缺什么」。

def evaluate_results(state: SearchAgentState) -> dict:
    """LLM 判断搜索结果是否足够，不足则生成新关键词"""
    if not state["all_results"]:
        return {"decision": "insufficient"}

    context = "\n".join(
        f"- [{d.doc_type}] {d.title}: {d.content[:200]}"
        for d in state["all_results"][:5]
    )

    prompt = f"""Question: {state['query']}

Search results:
{context}

Are these results sufficient to answer? Reply SUFFICIENT or INSUFFICIENT. Nothing else."""

    resp = llm.invoke([HumanMessage(content=prompt)])
    decision = resp.content.strip().upper()

    if "SUFFICIENT" in decision and "INSUFFICIENT" not in decision:
        print("  评估结果: 足够 ✓")
        return {"decision": "sufficient"}

    # 结果不足 → 第二轮 LLM 调用生成新关键词
    kw_prompt = f"""A search for "{state['query']}" returned results but more info is needed.
Suggest 1-2 new search keywords. Return ONLY a JSON array: ["kw1", "kw2"]"""

    resp2 = llm.invoke([HumanMessage(content=kw_prompt)])

    import json as _json
    keywords = []
    content = resp2.content.strip()
    try:
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            parsed = _json.loads(content[start:end+1])
            keywords = [str(k).strip() for k in parsed if isinstance(k, (str,))]
    except Exception:
        for line in content.split("\n"):
            s = line.strip().lstrip("-*").strip()
            if len(s) > 30 or len(s) < 2 or s.startswith(("{", "[", "```")):
                continue
            keywords.append(s)
    keywords = keywords[:2]

    print(f"  评估结果: 不足，新关键词 {keywords}")
    return {"decision": "insufficient", "keywords": keywords}


# ==============================================================================
# Teaching Notes: 路由函数 — 条件边
# ==============================================================================
# LangGraph 的条件边根据当前状态返回下一个节点的名称。
# 三种情况触发终止：
#   1. LLM 判断结果已足够
#   2. 达到最大搜索轮次
#   3. 没有结果且没有新关键词可用

def route_from_evaluate(state: SearchAgentState) -> str:
    if state["decision"] == "sufficient":
        return "synthesize"
    if state["attempt"] >= state["max_attempts"]:
        print(f"  已达最大轮次（{MAX_ATTEMPTS}），强制生成回答")
        return "synthesize"
    if not state.get("keywords"):
        print("  无新关键词，结束搜索")
        return "synthesize"
    return "search_hybrid"


# ==============================================================================
# Teaching Notes: 节点 4 — 答案生成
# ==============================================================================
# 经典 RAG 环节：将搜索结果作为上下文，LLM 生成自然语言回答。
# 无搜索结果时 LLM 也会给出相应说明。

def synthesize_answer(state: SearchAgentState) -> dict:
    """LLM 基于搜索结果生成自然语言回答"""
    if not state["all_results"]:
        answer = "未找到相关信息，请尝试其他查询。"
        print(f"  回答: {answer}")
        return {"answer": answer}

    context = "\n".join(
        f"- [{d.doc_type}] {d.title}: {d.content[:300]}"
        for d in state["all_results"][:5]
    )

    prompt = f"""基于以下搜索结果回答用户问题。

用户问题：{state['query']}

搜索结果：
{context}

请用中文给出简洁准确的回答。如果搜索结果不足以完整回答，
请说明知道什么、不知道什么。"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    answer = resp.content.strip()
    print(f"\n  回答: {answer}")
    return {"answer": answer}


# ==============================================================================
# Teaching Notes: 构建图
# ==============================================================================
# 四节点线性流程 + 条件循环：
#
#   decide_query → search_hybrid → evaluate_results
#       ↑                              │
#       └────── 不足 + 有轮次 ─────────┘
#                                      │ 足够 / 超轮次
#                                      ▼
#                               synthesize_answer → END

builder = StateGraph(SearchAgentState)
builder.add_node("decide_query", decide_query)
builder.add_node("search_hybrid", search_hybrid)
builder.add_node("evaluate_results", evaluate_results)
builder.add_node("synthesize_answer", synthesize_answer)

builder.set_entry_point("decide_query")
builder.add_edge("decide_query", "search_hybrid")
builder.add_edge("search_hybrid", "evaluate_results")
builder.add_conditional_edges(
    "evaluate_results",
    route_from_evaluate,
    {
        "search_hybrid": "search_hybrid",
        "synthesize": "synthesize_answer",
    },
)
builder.add_edge("synthesize_answer", END)

graph = builder.compile()


def run_agent(query: str) -> dict:
    """运行搜索 Agent，返回完整结果"""
    print(f"\n{'='*60}")
    print(f"  问: {query}")
    print(f"{'='*60}")

    result = graph.invoke({
        "query": query,
        "keywords": [],
        "all_results": [],
        "attempt": 0,
        "max_attempts": MAX_ATTEMPTS,
        "decision": "",
        "answer": "",
    })

    print(f"\n  ── 最终回答 ──")
    print(f"  {result['answer']}")
    print(f"  （共搜索 {result['attempt']} 轮，参考 {len(result['all_results'])} 篇文档）")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  搜索 Agent — LLM 增强的端到端搜索")
    print(f"  模型: {LLM_MODEL}  |  最大轮次: {MAX_ATTEMPTS}")
    print("=" * 60)
    print("  注意: llama3.2:1b 工具调用和指令遵循能力有限，")
    print("        关键词生成和结果评估可能不如 GPT/Claude 准确。")
    print("=" * 60)

    queries = [
        "东西坏了怎么退钱",
        "订单地址怎么改",
        "return product",
    ]

    for q in queries:
        run_agent(q)

    print(f"\n{'='*60}")
    print("  全部完成")
    print(f"{'='*60}")
