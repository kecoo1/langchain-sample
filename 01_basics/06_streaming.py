"""流式与异步示例。"""

import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)

# 1. 同步流式
print("=== 同步流式 ===")
for chunk in model.stream("Tell me a short story about a robot learning to paint."):
    print(chunk.content, end="", flush=True)
print("\n")


# 2. 与链结合的流式
prompt = ChatPromptTemplate.from_template(
    "Write a recipe for {dish}. Include ingredients and steps."
)
chain = prompt | model | StrOutputParser()

print("=== 链流式 ===")
for chunk in chain.stream({"dish": "pad thai"}):
    print(chunk, end="", flush=True)
print("\n")


# 3. 异步流式
async def async_stream():
    print("=== Async streaming ===")
    async for chunk in model.astream(
        "Explain quantum computing in simple terms."
    ):
        print(chunk.content, end="", flush=True)
    print()


asyncio.run(async_stream())


# 4. Async with event handling
async def stream_with_events():
    print("=== Async with events ===")
    async for event in model.astream_events(
        "List 3 programming languages and their uses.",
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        elif kind == "on_chat_model_start":
            print("[Model started] ", flush=True)
        elif kind == "on_chat_model_end":
            print("\n[Model finished]")


asyncio.run(stream_with_events())

# =============================================================================
# 教学备注：流式（Streaming）与异步（Async）——解决"用户等待"的两把钥匙
# =============================================================================
# 核心问题：之前的示例都是 model.invoke() 等完整输出——为什么要流式？
#   用户体验问题：invoke 等 5-10 秒出完整文本 vs stream 500ms 出第一个 token
#     "用户看着光标闪烁等待" vs "用户看着文字逐渐出现"
#   技术本质：LLM 生成是逐个 token 产生的（自回归），不是一次性算出来的
#     流式就是"边生成边返回"，不是"生成完了再返回"
#
# 模式 1：model.stream() - 同步流式（最简单的流式）
#   是什么：阻塞式迭代器，逐 chunk 返回模型输出
#   为什么叫"同步"：代码会阻塞在 for 循环上，但每次只等一个 chunk（毫秒级）
#   chunk.content 是什么：一个 token 或一组 token 的文本片段
#   关键理解：.stream() 和 .invoke() 的底层走的是同一个生成过程，只是返回方式不同
#   适用场景：CLI 工具、Jupyter Notebook、同步脚本、教学演示
#
# 模式 2：chain.stream() - 链式流式（LCEL 自动传递流）
#   是什么：LCEL 管道自动沿路传递流——prompt 瞬间完成，model 流式产出，parser 增量转换
#   为什么不用额外配置：因为 LCEL 的每个组件都实现了 stream 接口
#     如果一个组件不支持流式（如某些 parser），框架自动 fallback 到 invoke
#   关键：不是所有 parser 都支持流式！
#     支持：StrOutputParser（直接透传文本）、JsonOutputParser（增量解析 JSON）
#     不支持：PydanticOutputParser（必须等完整 JSON 才能反序列化）
#   链流式的意义：真正生产级的"打字机效果"——用户看到的不是模型输出，是解析后的最终结果
#
# 模式 3：model.astream() - 异步流式（高并发的关键）
#   是什么：async for 非阻塞迭代
#   为什么需要异步：同步 stream 会阻塞整个线程，一个请求没处理完不能处理下一个
#     在 FastAPI/WebSocket 场景中，异步意味着"一个服务同时服务 N 个用户"
#   异步的前提：调用栈必须全链路 async——async def handler -> await/async for
#   FastAPI 示例：async def chat(): async for chunk in model.astream(): yield chunk
#   适用场景：高并发 Web 服务、WebSocket 推送、实时通信
#
# 模式 4：model.astream_events(version="v2") - 事件驱动流式
#   是什么：不仅仅返回 token，还返回整个执行过程的各类事件
#   为什么需要：stream() 只能拿到最终文本，但有时候需要知道"什么时候开始"、"用了什么工具"
#     事件流能拿到：
#       on_chat_model_start -> 模型开始推理
#       on_chat_model_stream -> 每个 token
#       on_chat_model_end -> 模型完成
#       on_tool_start/end -> 工具调用前后
#       on_chain_start/end -> 链中每个节点的开始和结束
#   代价：事件量大（一个 token 一个事件），需要过滤处理
#   适用场景：监控面板、进度条显示、调试追踪、需要"模型思考过程"的 UI
#
# 模式 5：回调处理器（企业级可观测性）
#   是什么：通过 BaseCallbackHandler / AsyncCallbackHandler 注入到链的生命周期
#   与 astream_events 的区别：
#     - astream_events：从"调用方"角度获取事件，适合前端/API
#     - CallbackHandler：在"框架内部"注册，适合后台监控/日志
#   典型用途：记录每次调用的 token 数、计费、延迟统计、缓存检查
#   配置方式：config={"callbacks": [handler1, handler2]} 或 with_config
#
# 流式的不兼容陷阱（必读）：
#   1. PydanticOutputParser + stream()：直接报错或返回空——必须用 with_structured_output
#   2. 非流式模型 + stream()：自动 fallback 到 invoke，拿到的是完整输出
#   3. 自定义 Runnable + stream()：必须实现 _stream 方法，否则 fallback
#   4. 部分回调 Handler + stream()：某些回调只触发一次（on_chain_start），不是流式的
#
# 什么时候用同步 vs 异步？
#   这不是"哪个更好"的问题，是"你的框架要求什么"的问题：
#   - CLI/Jupyter/同步脚本 -> stream() 同步就够
#   - FastAPI/Starlette/Sanic -> 必须 astream()，否则阻塞事件循环
#   - WebSocket -> astream() 配合 websocket.send()
#   - SSE (Server-Sent Events) -> astream() 配合 StreamingResponse
#
# 教学建议顺序：
#   1. 先用 model.stream() 感受"token 一个一个出来"的效果
#   2. 再用 chain.stream() 理解"链也支持流式"
#   3. 再用 astream() 了解异步版本（不一定要跑，但要理解概念）
#   4. 最后 astream_events() 和 CallbackHandler 属于高级调试/监控话题
