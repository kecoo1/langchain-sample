"""链与可运行组合示例。"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama import ChatOllama
# 模型实例
model = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")

# 1. 简单链：提示 -> 模型 -> 输出解析器
prompt = ChatPromptTemplate.from_template(
    "Write a short poem about {topic} in the style of {style}."
)
chain = prompt | model | StrOutputParser()
result = chain.invoke({"topic": "coding", "style": "haiku"})
print(f"1. Simple chain:\n{result}\n")

# 2. 并行链（同时运行多个链）
prompt1 = ChatPromptTemplate.from_template("What is the capital of {country}?")
prompt2 = ChatPromptTemplate.from_template("What is the population of {country}?")

chain1 = prompt1 | model | StrOutputParser()
chain2 = prompt2 | model | StrOutputParser()

parallel_chain = RunnableParallel(capital=chain1, population=chain2)
result = parallel_chain.invoke({"country": "Japan"})
print(f"2. Parallel chain:\n{result}\n")

# 3. 带 passthrough 的链（在传递输入的同时添加派生值）
prompt = ChatPromptTemplate.from_template(
    "Given the word '{word}', provide: a synonym, an antonym, and a sentence using it."
)
chain = (
    RunnablePassthrough.assign(
        word_length=lambda x: len(x["word"]),
    )
    | prompt
    | model
    | StrOutputParser()
)
result = chain.invoke({"word": "serendipity"})
print(f"3. Passthrough chain:\n{result}\n")

# 4. 使用 RunnableBranch 的条件分支
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: len(x["text"]) < 50, lambda x: f"Short text: {x['text']}"),
    (lambda x: len(x["text"]) < 200, lambda x: f"Medium text: {x['text']}"),
    lambda x: f"Long text: {x['text']}",
)

branch_chain = branch | model | StrOutputParser()
result = branch_chain.invoke({"text": "Hello world"})
print(f"4. Branch chain:\n{result}\n")

## 5. 流式链
print("5. 流式链:")
for chunk in chain.stream({"word": "innovation"}):
    print(chunk, end="", flush=True)
print("\n")

# =============================================================================
# 教学备注：链组合的五种模式——从"单步调用"到"复杂工作流编排"
# =============================================================================
# 核心问题：为什么要设计链（Chain）？直接调用模型不好吗？
#   答案：真实应用从来不是"输入->模型->输出"这么简单；你需要——
#   1. 预处理：格式化输入、检索上下文、注入历史
#   2. 后处理：解析结构、校验格式、重试纠错
#   3. 编排：多个模型协作、条件路由、并行计算、流式输出
#   链的本质：将"数据处理"和"模型调用"统一抽象为 Runnable 接口，用管道符组合
#
# 模式 1：prompt | model | parser - 管道串联（最核心）
#   是什么：LCEL (LangChain Expression Language) 的核心语法，用 | 连接任意 Runnable
#   为什么这样设计：| 是 Unix 管道的思想——上一个的输出是下一个的输入；
#     这样任何组件都可以自由组合，无需知道上下游的具体实现
#   核心机制：每个 Runnable 必须实现 invoke/stream/batch/ainvoke 等统一接口
#   关键特性：自动追踪（LangSmith 自动记录每一步的输入/输出/延迟）、自动批处理
#   教学重点：这是 LangChain 最重要的设计模式，建议先用它理解"链"的概念
#   适用场景：所有线性流程——模板->模型->解析、检索->模板->模型->解析
#
# 模式 2：RunnableParallel - 并行分叉（解决性能问题）
#   是什么：同时运行多个独立链，结果用 dict 聚合，键名对应各链标识
#   为什么需要：假设要同时查首都和人口，串行要 2 次模型调用，并行只需要 1 次总时间
#   关键理解：并行不是"多个模型同时跑"，而是"多个输入共享一次调用"或"多调用同时触发"
#   限制：所有分支共享同一个输入 dict；一个分支失败会连带影响所有
#   适用场景：多维度分析一次输入（查首都+查人口+查货币）、多模型投票集成、同时查多个知识库
#
# 模式 3：RunnablePassthrough.assign() - 数据增强（解决输入不足问题）
#   是什么：不改变原始输入，增加派生字段后传递给下游
#   为什么需要：模型需要的信息往往不是用户给的"原始输入"
#     比如用户说"帮我翻译这个"——模型需要 context_length、目标语言、词汇表等额外信息
#   为什么不用手动 dict.update()：因为 assign 是声明式的，可序列化、可追踪、可组合
#   教学重点：数据转换也是"链"的一部分，prompt 不是唯一需要预处理的步骤
#   适用场景：计算输入长度/语言检测、注入用户元数据/权限信息、从外部 API 获取实时数据
#
# 模式 4：RunnableBranch - 条件路由（解决策略分化问题）
#   是什么：根据条件选择不同的下游处理路径，本质是 if-else 的声明式版本
#   为什么需要：业务逻辑常说"如果短文本就用小模型，长文本就用大模型"或"中文走这个，英文走那个"
#     if-else 在链式结构中破坏可组合性，Branch 将"选择"也变成链的一部分
#   核心设计：[(条件函数, 处理函数)] + 默认兜底函数；条件函数接收 same input dict
#   注意：条件函数不调用 LLM，只是做规则判断；复杂判断可先封装为 Runnable
#   适用场景：文本长度路由、语种路由、兜底/降级策略、A/B 测试分配
#
# 模式 5：.stream() / .astream() - 流式输出（解决响应延迟问题）
#   是什么：不等待完整响应，按 token 或 chunk 逐步输出
#   为什么需要：用户不耐烦——让用户"看到打字"比让用户"等待然后看到完整文本"体验好得多
#     而且第一个 token 通常在 500ms 内到达，而完整回复可能要 5-10 秒
#   关键机制：Runnable 的 stream 会自动沿管道传递；意味著 parser 也要支持流式
#   支持流式的解析器：StrOutputParser（字符串）、JsonOutputParser（增量 JSON）
#   不支持流式的解析器：PydanticOutputParser（需完整 JSON 后反序列化）
#   适用场景：聊天 UI、代码补全、实时翻译、长文档生成
#
# 进阶理解：链的隐式转换
#   - 同步 invoke 自动适配异步：chain.ainvoke() 内部如果组件只有 invoke 没有 ainvoke，会在线程池执行
#   - 流式兼容非流式：chain.stream() 对不流式的组件自动 fallback 到 invoke
#   - batch 自动并行：chain.batch(inputs) 并发执行，内部线程池管理
#   - 自动重试/退避：通过 .with_retry() 包装即可
#
# 选型决策树：
#   ┌─ 线性流程（模板->模型->解析）？ → | 管道串联
#   ├─ 同一输入需要多角度分析？ → RunnableParallel
#   ├─ 需要注入额外信息/派生值？ → RunnablePassthrough.assign()
#   ├─ 根据条件走不同路径？ → RunnableBranch
#   └─ 用户需要实时看到结果？ → .stream() 流式
#
# 教学建议顺序：
#   1. 先理解 | 管道 —— 这是最基础的"链"的概念
#   2. 再理解 RunnableParallel —— 用并行解决实际性能问题
#   3. 再理解 RunnablePassthrough.assign() —— 数据预处理也是链
#   4. 再理解 RunnableBranch —— 条件逻辑也能声明式
#   5. 最后理解流式 —— 在前四个基础上开启实时输出
