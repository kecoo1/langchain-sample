"""使用 Pydantic 模型的结构化输出示例。"""

from typing import Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 1. 简单的结构化输出
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    rating: Optional[int] = Field(default=None, description="Funny rating 1-10")


structured_model = model.with_structured_output(Joke)
result = structured_model.invoke("Tell me a programming joke")
print(f"1. Joke: {result}\n")

# 2. 从文本中提取结构化数据
class Person(BaseModel):
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(default=None, description="Age in years")
    occupation: str = Field(description="Job or profession")
    skills: list[str] = Field(description="List of skills")
    experience_years: int = Field(description="Years of experience")


class PeopleExtraction(BaseModel):
    people: list[Person] = Field(description="List of people mentioned")


extractor = model.with_structured_output(PeopleExtraction)
text = """
Alice Johnson is a 30-year-old software engineer with 8 years of experience.
She's skilled in Python, Rust, and Kubernetes.
Bob Smith is a data scientist who knows machine learning and statistics.
He has 5 years of experience.
"""

result = extractor.invoke(f"Extract people from: {text}")
print(f"2. Extracted people:")
for p in result.people:
    print(f"  - {p.name}: {p.occupation}, {p.experience_years}yrs, skills: {p.skills}")
print()


# 3. TypedDict 类型的输出
from typing import TypedDict


class MovieReview(TypedDict):
    title: str
    year: int
    rating: float  # 1-10
    summary: str
    genres: list[str]


dict_model = model.with_structured_output(MovieReview)
result = dict_model.invoke("Review the movie Inception")
print(f"3. Movie review: {result}\n")


# 4. 使用自定义输出解析器进行解析
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


class Recipe(BaseModel):
    name: str = Field(description="Recipe name")
    ingredients: list[str] = Field(description="List of ingredients")
    steps: list[str] = Field(description="Cooking steps")
    prep_time_minutes: int = Field(description="Preparation time in minutes")


parser = PydanticOutputParser(pydantic_object=Recipe)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You generate recipes. Follow the format instructions."),
    ("human", "Create a recipe for {dish}.\n{format_instructions}"),
])
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
result = chain.invoke({"dish": "pancakes"})
print(f"4. Recipe: {result.name}")
print(f"   Ingredients: {result.ingredients}")
print(f"   Prep time: {result.prep_time_minutes} min")
print(f"   Steps: {result.steps}")

# =============================================================================
# 教学备注：结构化输出——让 LLM 从"聊天机器"变成"数据处理引擎"
# =============================================================================
# 核心问题：之前的例子都是 result.content（字符串）——为什么需要"结构化"输出？
#   想象一个场景：你需要从简历中提取[姓名、年龄、技能列表、工作经验]
#     字符串输出：你只能靠正则/提示词去猜结构，解析脆弱、字段缺失时无从验证
#     结构化输出：模型直接返回 {"name": "Alice", "age": 30, "skills": ["Python"]}
#     你拿到的是 Python 对象，可以直接 .name，直接 .skills
#   核心道理：文本是给"人"看的，结构才是给"程序"用的
#   当 LLM 的输出要接入下游系统（数据库/API/前端表单）时，结构就是刚需
#
# 方式 1：model.with_structured_output(PydanticModel) - 首选方案
#   是什么：一行代码把"文本模型"变成"结构化输出模型"
#   为什么叫"首选"：它不会生成字符串再解析——底层走模型的 function calling 或 JSON mode
#     模型直接返回符合 schema 的 JSON，省去了"解析-校验"这一步
#   底层机制：随模型不同而不同——
#     OpenAI: function calling (tools 参数)
#     Anthropic: tool_use
#     Gemini: response_mime_type="application/json"
#     Ollama: response_format={"type": "json_object"}
#   开发者无需关心底层差异，with_structured_output 帮你适配
#   关键优势：类型安全——result.joke_setup -> IDE 自动补全、类型检查
#   Field(description=...) 的重要性：description 就是给模型的"提示"
#     哪个字段是 optional 的、取值范围是多少、格式是什么——都写在 description 里
#   适用场景：所有生产级结构化输出需求，首选方案，没有之一
#
# 方式 2：model.with_structured_output(TypedDict) - 轻量方案
#   是什么：同上，但用 TypedDict 替代 Pydantic
#   为什么存在：Pydantic 是外部依赖，有些项目不想引入；或者只需要"定义一个简单的 dict 结构"
#   与 Pydantic 的关键差异：
#     TypedDict 没有运行时验证——模型返回 {"rating": "good"} 而不是 {"rating": 8.5}，你也拿它没办法
#     TypedDict 没有默认值——所有字段"必填"
#     TypedDict 没有 Field(description=...)——你没法给字段写说明，模型更容易理解错
#   适用场景：快速原型、团队不使用 Pydantic、只需要类型提示不需要运行时校验
#   教学建议：能用 Pydantic 就用 Pydantic，差别只在于多 import 一行
#
# 方式 3：PydanticOutputParser + 手工提示 - 兼容方案
#   是什么：先让模型生成 JSON 字符串，再用 Parser 解析成 Pydantic 对象
#   为什么还要学它：方式 1 依赖模型的 function calling 能力——
#     但是开源模型、本地微调模型可能不支持或支持不好
#     这时就要"教"模型输出 JSON：在 prompt 中嵌入 format_instructions
#   format_instructions 长什么样？
#     "You must output JSON matching this schema: {name: string, ingredients: string[]}"
#     模型看到这个指令，更大概率输出合法 JSON
#   缺点：模型可能输出格式错误的 JSON -> parser 抛异常 -> 你需要重试逻辑
#     token 消耗更大（JSON 格式说明占用了大量上下文）
#   适用场景：不支持 function calling 的模型、本地模型、自行微调的模型
#
# 方式 4：JsonOutputParser / StrOutputParser - 自由格式
#   是什么：最轻量的解析方式——只做"模型输出 -> dict/string"的转换，不做 schema 验证
#   JsonOutputParser 的独特价值：支持流式增量解析（partial JSON）
#     模型输出 '{ "name": "Inception", "ye' -> 解析器可能还无法解析
#     模型输出 '{ "name": "Inception", "year": 2010' -> 能解析出部分结果 {name: "Inception"}
#     最终输出完整 JSON -> 完整解析
#   适用场景：流式场景下实时展示"部分结果"、简单键值对提取
#
# 流式结构化输出：with_structured_output(..., streaming=True)
#   是什么：在 with_structured_output 基础上开启流式
#   为什么不是默认的：大多数时候你不需要"逐步看到结构化输出的填充过程"
#     你可能只需要最终完整的 Pydantic 对象
#   开启后返回 PartialModel：逐步填充字段，状态从 invalid -> partial -> complete
#   代价：不是所有模型支持；底层实现复杂（需增量 JSON 解析）
#   适用场景：长报告生成（markdown 结构逐步涌现）、实时表单填充预览
#
# 教学建议顺序：
#   1. 先用 with_structured_output(Pydantic) —— 最省心，一行代码解决
#   2. 再理解 with_structured_output 的底层原理（本质是 function calling）
#   3. 再了解 PydanticOutputParser，应对不支持 function calling 的模型
#   4. JsonOutputParser 和 流式结构化输出属于进阶话题，按需学习
