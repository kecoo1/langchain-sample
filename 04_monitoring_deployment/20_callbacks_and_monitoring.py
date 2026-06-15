"""回调和监控示例 —— 自定义 BaseCallbackHandler、AsyncCallbackHandler、Token 追踪、成本计算。

回调机制让你可以注入自定义逻辑到 LangChain Runnable 的整个生命周期中。
适用于：Token 计费、自定义日志、指标收集、缓存检查、审计追踪等。

运行前设置: export OPENAI_API_KEY="sk-..."
"""

import time
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 示例 1：自定义同步回调 —— Token 使用和成本追踪
# ==============================================================================
print("=== 示例 1: Token 使用和成本追踪 ===")


class TokenUsageHandler(BaseCallbackHandler):
    """追踪每次调用的 token 数量和估算成本。"""

    # OpenAI 模型价格（每 1000 tokens）
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }

    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.calls = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        print(f"   [LLM Start] Model: {serialized.get('name', 'unknown')}")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 流式输出时每个 token 都会触发
        pass

    def on_llm_end(self, response, **kwargs) -> None:
        # 从响应中获取 token 用量
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            model_name = response.llm_output.get("model_name", "unknown")

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total = input_tokens + output_tokens

            # 计算成本
            pricing = self.PRICING.get(model_name, {"input": 0, "output": 0})
            cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000

            call_record = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total,
                "cost_usd": round(cost, 6),
            }
            self.calls.append(call_record)

            self.total_tokens += total
            self.total_cost += cost

            print(f"   [Token Usage] Input: {input_tokens}, Output: {output_tokens}, "
                  f"Cost: ${cost:.6f}")


# 使用回调
token_handler = TokenUsageHandler()

chain_with_tracking = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ])
    | model
    | StrOutputParser()
)

questions = [
    "What is LangChain?",
    "Explain transformers in one sentence.",
]

for q in questions:
    result = chain_with_tracking.invoke(
        {"question": q},
        config={"callbacks": [token_handler]}
    )
    print(f"   Response: {result[:50]}...\n")

print(f"   总计: {token_handler.total_tokens} tokens, 花费: ${token_handler.total_cost:.6f}")
print(f"   调用次数: {len(token_handler.calls)}\n")


# ==============================================================================
# 示例 2：延迟追踪
# ==============================================================================
print("=== 示例 2: 延迟追踪 ===")


class LatencyHandler(BaseCallbackHandler):
    """追踪每个 Runnable 的执行延迟。"""

    def __init__(self):
        self.latencies = defaultdict(list)
        self.start_times = {}

    def on_chain_start(self, serialized, inputs, **kwargs):
        run_id = kwargs.get("run_id", "unknown")
        self.start_times[run_id] = time.time()

    def on_chain_end(self, outputs, **kwargs) -> None:
        run_id = kwargs.get("run_id", "unknown")
        if run_id in self.start_times:
            elapsed = time.time() - self.start_times[run_id]
            self.latencies["chain"].append(elapsed)
            print(f"   [Chain] 耗时: {elapsed:.3f}s")

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        run_id = kwargs.get("run_id", "unknown")
        self.start_times[run_id] = time.time()

    def on_llm_end(self, response, **kwargs) -> None:
        run_id = kwargs.get("run_id", "unknown")
        if run_id in self.start_times:
            elapsed = time.time() - self.start_times[run_id]
            self.latencies["llm"].append(elapsed)
            print(f"   [LLM] 耗时: {elapsed:.3f}s")

    def summary(self):
        for run_type, times in self.latencies.items():
            avg = sum(times) / len(times)
            print(f"   {run_type}: 平均 {avg:.3f}s, 共 {len(times)} 次调用")


latency_handler = LatencyHandler()

for q in ["What is AI?", "What is ML?", "What is DL?"]:
    result = chain_with_tracking.invoke(
        {"question": q},
        config={"callbacks": [latency_handler]}
    )

print("\n   延迟摘要:")
latency_handler.summary()
print()


# ==============================================================================
# 示例 3：异步回调
# ==============================================================================
print("=== 示例 3: 异步回调 ===")


class AsyncLoggingHandler(AsyncCallbackHandler):
    """异步回调处理器，适合高并发场景。"""

    def __init__(self):
        self.events = []

    async def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.events.append({
            "type": "model_start",
            "tokens": len(messages[0].content) if hasattr(messages[0], 'content') else 0,
        })
        print("   [Async] Model started")

    async def on_chat_model_end(self, output, **kwargs) -> None:
        self.events.append({"type": "model_end"})
        print("   [Async] Model finished")

    async def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        self.events.append({"type": "chain_start"})

    async def on_chain_end(self, outputs, **kwargs) -> None:
        self.events.append({"type": "chain_end"})
        print("   [Async] Chain completed")


async def async_demo():
    import asyncio

    async_handler = AsyncLoggingHandler()

    async_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Brief answer."),
            ("human", "{question}"),
        ])
        | model
        | StrOutputParser()
    )

    # 并发调用
    questions = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
    results = await async_chain.abatch(
        [{"question": q} for q in questions],
        config={"callbacks": [async_handler]}
    )

    print(f"   异步结果: {results}")
    print(f"   捕获事件: {len(async_handler.events)}\n")


import asyncio
asyncio.run(async_demo())


# ==============================================================================
# 示例 4：.with_handlers() —— LCEL 原生事件处理
# ==============================================================================
print("=== 示例 4: .with_handlers() 原生事件处理 ===")


def on_start(run_id, run):
    print(f"   [Handler] Run started: {run.name}")


def on_end(run_id, run):
    if run.outputs:
        output = str(run.outputs.get("output", ""))[:50]
        print(f"   [Handler] Run ended: {output}...")


handler_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a translator."),
        ("human", "Translate to {language}: {text}"),
    ])
    | model
    | StrOutputParser()
    .with_handlers({
        "on_chat_model_start": on_start,
        "on_chat_model_end": on_end,
    })
)

result = handler_chain.invoke({"text": "Hello world", "language": "French"})
print(f"   结果: {result}\n")


# ==============================================================================
# 示例 5：缓存检查回调
# ==============================================================================
print("=== 示例 5: 缓存检查回调 ===")


class CacheCheckingHandler(BaseCallbackHandler):
    """模拟缓存检查，避免重复调用 LLM。"""

    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _hash_input(self, prompts):
        """简单哈希输入用于缓存键。"""
        return str(sorted(prompts))

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        cache_key = self._hash_input(prompts)

        if cache_key in self.cache:
            self.hits += 1
            print(f"   [Cache HIT] '{prompts[0][:30]}...'")
            # 返回缓存结果（在实际使用中需要特殊处理）
            # 这里仅演示概念
        else:
            self.misses += 1
            print(f"   [Cache MISS] '{prompts[0][:30]}...'")

    def summary(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        print(f"   缓存命中率: {hit_rate:.1f}% ({self.hits}/{total})")


cache_handler = CacheCheckingHandler()

# 注意：实际缓存需要使用 langchain.globals.set_llm_cache()
# 这里仅演示回调模式的缓存检查概念
for q in ["What is LangChain?", "What is AI?", "What is LangChain?"]:
    result = chain_with_tracking.invoke(
        {"question": q},
        config={"callbacks": [cache_handler]}
    )

cache_handler.summary()
print()


# ==============================================================================
# 示例 6：错误处理和审计日志
# ==============================================================================
print("=== 示例 6: 错误处理和审计日志 ===")


class AuditLogHandler(BaseCallbackHandler):
    """审计日志：记录所有调用的详细信息。"""

    def __init__(self):
        self.audit_log = []

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        self.audit_log.append({
            "event": "chain_start",
            "name": serialized.get("name", "unknown"),
            "inputs": inputs,
            "timestamp": time.time(),
        })

    def on_chain_end(self, outputs, **kwargs) -> None:
        self.audit_log.append({
            "event": "chain_end",
            "outputs": outputs,
            "timestamp": time.time(),
        })

    def on_chain_error(self, error, **kwargs) -> None:
        self.audit_log.append({
            "event": "chain_error",
            "error": str(error),
            "timestamp": time.time(),
        })
        print(f"   [Audit] Error caught: {error}")

    def export_log(self, filepath: str = "/tmp/audit_log.json") -> None:
        with open(filepath, "w") as f:
            json.dump(self.audit_log, f, indent=2, default=str)
        print(f"   审计日志已导出到: {filepath}")


audit_handler = AuditLogHandler()

try:
    result = chain_with_tracking.invoke(
        {"question": "Hello"},
        config={"callbacks": [audit_handler]}
    )
    print(f"   正常执行: {result[:30]}...")
except Exception as e:
    print(f"   捕获异常: {e}")

audit_handler.export_log()
print()


# ==============================================================================
# 教学备注：回调和监控 —— LLM 应用的可观测性
# ==============================================================================
# 核心问题：你怎么知道你的 LLM 应用在"做什么"和"花多少钱"？
#   1. Token 计费：每次调用都花钱，需要追踪用量
#   2. 延迟监控：找出性能瓶颈
#   3. 错误追踪：捕获和处理异常
#   4. 审计日志：合规要求和故障排查
#
# 回调的生命周期事件:
#   on_chain_start/end/error    - 链的开始/结束/错误
#   on_llm_start/end/error      - LLM 调用的开始/结束/错误
#   on_chat_model_start/end     - 聊天模型的开始/结束
#   on_tool_start/end/error     - 工具调用的开始/结束/错误
#   on_text                     - 文本输出
#   on_llm_new_token            - 流式输出的每个 token
#
# BaseCallbackHandler vs AsyncCallbackHandler:
#   同步: 用于同步 invoke/stream 调用
#   异步: 用于异步 ainvoke/astream 调用
#   注意: 异步回调不能阻塞事件循环，必须使用 async def
#
# .with_handlers() vs 回调:
#   .with_handlers() 是 LCEL 原生的事件处理方式
#   更简洁，适合简单的事件监听
#   回调更适合复杂的逻辑（缓存、计费、审计）
#
# 生产级监控建议:
#   1. Token 追踪: 每次调用记录 token 数 -> 成本核算
#   2. 延迟追踪: 记录 P50/P90/P99 延迟 -> 性能优化
#   3. 错误率: 记录失败次数 -> 可靠性监控
#   4. 与 LangSmith 结合: 回调 + LangSmith = 完整可观测性
#   5. 与 Prometheus/Grafana 结合: 指标导出 -> 可视化监控
#
# 教学建议顺序:
#   1. 先理解回调的生命周期事件
#   2. 跑通 TokenUsageHandler（最实用的示例）
#   3. 再理解延迟追踪和异步回调
#   4. 最后理解 .with_handlers() 和审计日志
