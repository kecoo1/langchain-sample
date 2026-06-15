"""工具定义与工具调用示例。"""

import json
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434", temperature=0)

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 1. 使用 @tool 装饰器定义工具
@tool
def get_weather(location: str, unit: Optional[str] = None) -> str:
    """Get current weather for a location.

    Args:
        location: City name, e.g. 'Tokyo', 'Paris'
        unit: Temperature unit, 'celsius' or 'fahrenheit'
    """
    return f"25°{'C' if unit != 'fahrenheit' else 'F'} and sunny in {location}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression, e.g. '2 + 2 * 3'
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# 2. 将工具绑定到模型并调用
model_with_tools = model.bind_tools([get_weather, calculator])
result = model_with_tools.invoke("What's the weather in Tokyo and what is 25 * 4?")
print(f"Tool calls: {result.tool_calls}\n")

# 3. 手动执行工具调用
for tc in result.tool_calls:
    tool_map = {"get_weather": get_weather, "calculator": calculator}
    selected_tool = tool_map[tc["name"]]
    tool_result = selected_tool.invoke(tc["args"])
    print(f"  {tc['name']}({tc['args']}) = {tool_result}")
    print()

# 4. 带结构化输出的工具示例
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, description="Maximum results to return")


@tool(args_schema=SearchInput)
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
    """
    return [
        {"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": f"Snippet for {query}"}
        for i in range(max_results)
    ]


model_with_search = model.bind_tools([web_search])
result = model_with_search.invoke("Search for AI news")
print(f"Structured tool calls: {result.tool_calls}")

# 5. 在一次调用中进行多个工具调用
result = model_with_tools.invoke(
    "What's the weather in Paris? Also calculate 2^10."
)
print(f"\nMultiple tool calls: {json.dumps(result.tool_calls, indent=2)}")

# =============================================================================
# 教学备注：工具（Tool）系统的完整认知——从定义到调用的三条路线
# =============================================================================
# 核心问题：为什么要引入"工具"的概念？模型不能直接做这些事吗？
#   答案：模型天生有三个局限性——
#   1. 无法获取实时数据（天气、股价、数据库记录）
#   2. 无法执行精确计算（数学、规则校验、加密）
#   3. 无法与外部系统交互（发邮件、查数据库、调用 API）
#   工具系统的本质：为模型提供"数字世界的双手和眼睛"，让模型不仅能"说"，还能"做"
#   一次完整的工具调用流程：模型决策 -> 解析参数 -> 执行函数 -> 返回结果 -> 模型整合答案
#
# 路线 1：@tool 装饰器 - 从 Python 函数到 LLM 工具（最常用）
#   是什么：一个装饰器，把普通 Python 函数自动转换为 LLM 可理解的"工具描述"
#   为什么这样设计：避免了手写 JSON Schema——函数名 -> 工具名、文档字符串 -> 工具描述、
#     参数类型+默认值+注释 -> 参数 schema；这是"约定优于配置"的最佳体现
#   自动推导规则：
#     - 函数名 `get_weather` -> 工具名 {name: "get_weather"}
#     - 文档字符串第一行 -> description；Args 部分 -> 参数的 description
#     - 类型注解 str/Optional[str]/int/float -> type string/number
#     - 默认值 None -> optional；默认值 5 -> 注入 default
#   易错点：文档字符串格式必须规范（Google/NumPy 风格），否则描述为空或乱码
#   适用场景：95% 的工具定义场景，初学者和进阶都推荐
#
# 路线 2：@tool(args_schema=PydanticModel) - 结构化参数验证
#   是什么：在 @tool 基础上，用 Pydantic 模型代替函数签名定义参数 schema
#   为什么需要：函数签名能表达的信息有限——
#     - 不能给参数写详细描述（docs 字符串的 Args 不标准且易解析错误）
#     - 不能做复杂验证（邮箱格式、枚举值范围、字段间依赖关系）
#     - 不能表达嵌套结构（列表、字典、自定义类型）
#   与路线 1 的区别：
#     - @tool 自动：函数注解 + 文档字符串 -> JSON Schema
#     - args_schema：Pydantic Field(description=...) -> 更准确的 JSON Schema
#     - args_schema 还可调用 model_validate 做运行时数据验证
#   适用场景：生产环境工具、参数超过 5 个、需要详细描述/验证、被其他团队调用的共享工具
#
# 路线 3：手动构建 — StructuredTool / BaseTool（最底层）
#   是什么：直接实例化 StructuredTool 或继承 BaseTool 定义工具
#   为什么需要：装饰器只能修饰函数；当工具逻辑来自其他系统时（数据库记录、配置中心、
#     动态注册插件、用户自定义 Lisp 代码），需要运行时创建工具对象
#   典型场景：插件系统的工具注册、从配置文件读取工具定义、同一函数不同参数注册多次
#   注意：90% 的情况下不需要用到这一层，知道存在即可
#
# 模型调用工具的方式：model.bind_tools([tool1, tool2, ...])
#   是什么：把工具列表"绑定"到模型，模型输出时会自动判断是否要调用工具
#   为什么不是自动的：不是每次对话都需要工具——有时只是闲聊；显式绑定让模型"知道有工具可用"
#   底层机制：bind_tools 把工具描述嵌入到 system prompt 中（OpenAI 的 tools 参数）
#   输出的产物：result.tool_calls = [{"name": "get_weather", "args": {"location": "Tokyo"}}]
#   关键认知：bind_tools 只负责"模型生成调用参数"，不负责"执行工具"——执行需要你自己写循环
#
# 手动执行工具 vs 自动执行（Agent 章节展开）
#   手动：解析 result.tool_calls -> 在代码中 if/elif/dispatch -> 调用 -> 整合结果
#   自动：交给 AgentExecutor/Agent 自动循环：模型决策->执行->反馈->模型决策->结束
#   什么时候手动：只需一轮工具调用（查个天气就走）、工具调用结果是最终输出
#   什么时候自动：需要多轮推理（查天气 -> 根据天气推荐活动 -> 再查活动地址...）
#
# 一个完整的工具调用生命周期（理解这个比记住 API 更重要）：
#   1. 用户说："东京天气怎么样？25*4 等于多少？"
#   2. 模型分析：需要一次调用两个工具（并行工具调用）
#   3. 模型返回：tool_calls = [{name: "get_weather", args: {location: "Tokyo"}},
#                              {name: "calculator", args: {expression: "25 * 4"}}]
#   4. 开发者（或 Agent）执行这两个函数
#   5. 得到结果：["25°C and sunny in Tokyo", "100"]
#   6. 开发者把结果构造为 ToolMessage 返回给模型
#   7. 模型整合：给出最终自然语言回答
#
# 教学建议顺序：
#   1. 先学会 @tool 定义工具 —— 3 行代码让函数变工具
#   2. 再学会 bind_tools + 手动调用 —— 理解"模型生成参数，代码执行"的分离
#   3. 再学会 args_schema —— 当参数复杂时升级
#   4. 避免过早跳入手动构建，那是高级主题
