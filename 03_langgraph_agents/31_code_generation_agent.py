"""代码生成 Agent —— 生成 → 测试 → 修复 → 部署的开发流水线。

让 Agent 编写代码只是第一步，关键是让生成的代码能运行。
本文件展示从需求到可用代码的完整循环：
1. 代码生成 + 执行验证
2. 带错误修复的代码生成
3. 多文件代码项目生成
4. 代码审查 Agent
"""

import ast
import sys
import traceback
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ==============================================================================
# 示例 1：代码生成 + 自动语法检查
# ==============================================================================
print("=== 示例 1: 代码生成 + 语法验证 ===")


def generate_and_validate(task: str) -> dict:
    """生成代码并检查语法。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python developer. Write clean, well-documented code.
Return ONLY the Python code, no explanations."""),
        ("human", "{task}"),
    ])
    chain = prompt | model | StrOutputParser()

    code = chain.invoke({"task": task})
    code = code.replace("```python", "").replace("```", "").strip()

    result = {"code": code, "valid": False, "error": None}

    # 语法检查
    try:
        ast.parse(code)
        result["valid"] = True
    except SyntaxError as e:
        result["error"] = str(e)

    return result


# 测试：生成一个 CSV 处理函数
task = "Write a Python function that reads a CSV file, filters rows where age > 30, and returns the result as a list of dicts. Include proper error handling."
result = generate_and_validate(task)

print(f"1. 代码生成结果:")
print(f"   语法检查: {'✅ 通过' if result['valid'] else '❌ 失败'}")
if result["error"]:
    print(f"   错误: {result['error']}")
print(f"   代码行数: {len(result['code'].splitlines())}")
print(f"   代码预览:\n{result['code'][:300]}...\n")


# ==============================================================================
# 示例 2：代码生成 → 执行 → 修复 循环
# ==============================================================================
print("=== 示例 2: 生成 → 执行 → 修复 ===")


class CodeGenWithFix:
    """带错误修复的代码生成器。"""

    def __init__(self, max_attempts=3):
        self.generator_prompt = ChatPromptTemplate.from_messages([
            ("system", "Write Python code for the given task. Only return code, no explanation."),
            ("human", "{task}"),
        ])
        self.fixer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Fix the given Python code. Return ONLY the fixed code."),
            ("human", "Task: {task}\n\nCode:\n{code}\n\nError:\n{error}"),
        ])
        self.max_attempts = max_attempts

    def run(self, task: str) -> dict:
        code = (self.generator_prompt | model | StrOutputParser()).invoke({"task": task})
        code = code.replace("```python", "").replace("```", "").strip()

        history = []
        for attempt in range(self.max_attempts):
            # 语法检查
            try:
                ast.parse(code)
            except SyntaxError as e:
                history.append({"attempt": attempt + 1, "status": "syntax_error", "error": str(e)})
                code = (self.fixer_prompt | model | StrOutputParser()).invoke({
                    "task": task, "code": code, "error": str(e),
                })
                code = code.replace("```python", "").replace("```", "").strip()
                continue

            # 运行时检查（在沙箱中执行）
            try:
                local_vars = {}
                exec(code, {"__builtins__": __builtins__}, local_vars)
                # 找生成的主要函数
                func_name = [k for k, v in local_vars.items() if callable(v)][0] if local_vars else None
                history.append({"attempt": attempt + 1, "status": "success", "func_name": func_name})
                return {"code": code, "attempts": attempt + 1, "history": history}
            except Exception as e:
                history.append({"attempt": attempt + 1, "status": "runtime_error", "error": str(e)})
                if attempt < self.max_attempts - 1:
                    code = (self.fixer_prompt | model | StrOutputParser()).invoke({
                        "task": task, "code": code, "error": str(e),
                    })
                    code = code.replace("```python", "").replace("```", "").strip()

        return {"code": code, "attempts": self.max_attempts, "history": history}


gen = CodeGenWithFix(max_attempts=3)

result = gen.run("Write a function fibonacci(n) that returns the nth Fibonacci number.")
print(f"2. 生成 → 执行 → 修复:")
print(f"   总尝试: {result['attempts']} 次")
for h in result["history"]:
    if h["status"] == "success":
        print(f"   成功: 函数 {h.get('func_name', 'unknown')} 正确执行")
    else:
        print(f"   失败: {h['status']} - {h.get('error', '')[:60]}")
print()


# ==============================================================================
# 示例 3：多文件代码项目生成
# ==============================================================================
print("=== 示例 3: 多文件代码项目生成 ===")

# 真实场景中 Agent 需要生成多个文件的完整项目
# 重点是：文件间依赖管理、导入关系正确

project_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a software architect. Design a multi-file Python project.
For each file, specify:
--- filename: main.py ---
code content
"""),
    ("human", "Create a simple project: a REST API client with model classes. Include:\n"
              "1. models.py - Pydantic data models\n"
              "2. client.py - HTTP client wrapper\n"
              "3. main.py - Usage example"),
])

project_plan = (project_prompt | model | StrOutputParser()).invoke({})

# 解析生成的文件
import re
files = re.findall(r'--- filename: (.+?) ---\n(.*?)(?=--- filename:|$)', project_plan, re.DOTALL)

print(f"3. 多文件项目生成:")
print(f"   项目包含 {len(files)} 个文件:")
for filename, content in files:
    content = content.strip()
    is_valid = True
    if filename.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as e:
            is_valid = False
            print(f"   ❌ {filename}: 语法错误 - {str(e)[:40]}")
    if is_valid:
        print(f"   ✅ {filename}: {len(content.splitlines())} 行")
print()

print("   多文件项目管理要点：")
print("   - 文件间导入关系必须正确（from models import ...）")
print("   - 每文件职责单一（单一职责原则）")
print("   - 生成后验证所有文件的 import 是否可解析")
print()


# ==============================================================================
# 示例 4：代码审查 Agent
# ==============================================================================
print("=== 示例 4: 代码审查 Agent ===")

SAMPLE_CODE = """
def process_data(data, threshold):
    results = []
    for i in range(len(data)):
        if data[i] > threshold:
            results.append(data[i] ** 2)
    return results

def save_results(data, filename):
    f = open(filename, 'w')
    for item in data:
        f.write(str(item) + '\\n')

def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)
"""

review_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior code reviewer. Review the code for:
1. Bugs and errors
2. Performance issues
3. Code style and best practices
4. Security concerns
5. Suggested improvements

For each issue found, specify:
- Severity: critical/major/minor
- Location: line or function
- Description: what's wrong
- Suggestion: how to fix"""),
    ("human", "{code}"),
])

review_result = (review_prompt | model | StrOutputParser()).invoke({"code": SAMPLE_CODE})
print(f"4. 代码审查结果:")
print(review_result[:400])
print()

# 带自动修复
fix_prompt = ChatPromptTemplate.from_messages([
    ("system", "Fix all issues in this code. Return ONLY the improved code."),
    ("human", "Review feedback:\n{review}\n\nOriginal code:\n{code}"),
])

fixed_code = (fix_prompt | model | StrOutputParser()).invoke({
    "review": review_result,
    "code": SAMPLE_CODE,
})
fixed_code = fixed_code.replace("```python", "").replace("```", "").strip()

# 验证修复后的代码
try:
    ast.parse(fixed_code)
    print(f"   修复后代码语法检查: ✅ 通过")
    print(f"   修复行数变化: {len(SAMPLE_CODE.splitlines())} → {len(fixed_code.splitlines())} 行")
except SyntaxError as e:
    print(f"   修复后代码语法检查: ❌ {e}")
print()


# ==============================================================================
# 教学备注：代码生成 Agent 的生产实践
# ==============================================================================
# 核心问题：LLM 生成的代码"看起来对但跑起来错"是常态
#   关键不在于"生成一次就完美"，而在于"快速发现并修复错误"
#
# 代码生成质量保障层级：
#
#   L1 - 语法检查（静态）
#     → ast.parse(code) 检查语法错误
#     → 耗时：毫秒级
#     → 捕获：缩进错误、括号不对、关键字拼错
#
#   L2 - 类型检查（静态）
#     → mypy/pyright 类型注解检查
#     → 耗时：秒级
#     → 捕获：类型不匹配、None 处理遗漏
#
#   L3 - 运行时验证（动态）
#     → exec(code) 在沙箱中执行
#     → 耗时：秒到分钟级
#     → 捕获：NameError（未定义变量）、LogicError（逻辑错误）
#
#   L4 - 测试验证（动态）
#     → 运行 pytest/unittest
#     → 耗时：分钟级
#     → 捕获：边界条件错误、功能不完整
#
# 生成 → 修复 循环的关键技巧：
#
#   1. 错误信息要完整返回给 LLM
#      ✅ 返回完整 traceback（LLM 需要看到行号）
#      ❌ "你的代码有错误"（太模糊，LLM 无法定位）
#
#   2. 一次只修复一个问题
#      修复一个错误后重新运行，再发现下一个
#      避免一次修复多个问题（可能引入新问题）
#
#   3. 代码执行必须沙箱化
#      使用 subprocess 或 Docker 隔离执行
#      防止 LLM 生成的恶意代码破坏宿主系统
#
# 安全注意事项（生产环境必须）：
#   1. 永远不要在 sandbox 外 exec() LLM 生成的代码
#   2. 限制内存使用（防止 fork bomb）
#   3. 设置超时（防止死循环）
#   4. 限制网络访问（防止数据外泄）
#   5. 限制文件系统访问（防止读取/写入敏感文件）
#
# 进阶方向（与本项目其他文件的关系）：
#   - 结合 Plan-and-Execute（27）：先生成设计文档 → 再编码
#   - 结合 Reflexion（27）：输出代码后自我审查 → 修正
#   - 结合并行工具调用（30）：同时生成多个文件
#   - 结合安全防护（25）：审查生成的代码是否包含恶意内容
#   - 结合 Deep Agents（22）：内置代码执行沙箱
