"""聊天模型的基础用法。"""

from langchain.chat_models import init_chat_model

# 使用带提供者前缀的 `init_chat_model`
# model = init_chat_model("openai:gpt-4o-mini")
# result = model.invoke("Hello, world!")
# print(f"Response: {result.content}")
# print(f"Token usage: {result.usage_metadata}")

# 直接使用 partner 包
# from langchain_openai import ChatOpenAI

# model2 = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# result2 = model2.invoke("What is the capital of France?")
# print(f"Response: {result2.content}")


# from langchain_openai import ChatOpenAI
# model3 = ChatOpenAI(
#     model="gpt-4o-mini",
#     base_url="https://your
# -custom-endpoint.com/v1",
#     api_key="your-api-key"
# )
from langchain_ollama import ChatOllama
model3 = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
result3 = model3.invoke("你是什么模型，现在在什么地方运行?")
print(f"Response: {result3.content}")
print(f"Token usage: {result3.usage_metadata}")
# 异步调用示例
import asyncio


async def async_example():
    result = await model3.ainvoke("Tell me a fun fact.")
    print(f"Async response: {result.content}")


asyncio.run(async_example())

# # 第三方模型配置示例（含自定义 base_url）
# # ---
# # 1. OpenAI 兼容接口（如 vLLM, LocalAI, 代理等）
# # from langchain_openai import ChatOpenAI
# # model3 = ChatOpenAI(
# #     model="gpt-4o-mini",
# #     base_url="https://your-custom-endpoint.com/v1",
# #     api_key="your-api-key"
# # )

# # 2. Anthropic
# # from langchain_anthropic import ChatAnthropic
# # model4 = ChatAnthropic(
# #     model="claude-3-5-haiku-latest",
# #     base_url="https://your-custom-endpoint.com",  # 或通过 ANTHROPIC_BASE_URL 环境变量
# # )

# # 3. Ollama（本地模型）
# # from langchain_ollama import ChatOllama
# # model5 = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

# =============================================================================
# 教学备注：三种模型实例化方式的核心区别与选型指南
# =============================================================================
# 为什么要有三种方式？因为实际开发中面临三类典型需求：
# 1. "我就想快速跑通流程，不用管底层细节" -> init_chat_model
# 2. "我要用 OpenAI 的特有参数、调控 temperature、配代理" -> provider-specific 类
# 3. "数据不能出公司/要离线跑/成本控制" -> 本地/自托管
#
# 方式 1：init_chat_model("provider:model") - 统一工厂模式
#   是什么：LangChain 提供的统一入口，通过字符串前缀自动路由到对应 provider
#   为什么需要：解决"切换模型要改导入、改类名、改参数名"的痛点
#   核心优势：代码与 provider 解耦，配置外置（环境变量/配置文件），便于 CI/CD 多环境切换
#   局限性：只暴露通用参数；provider 独有参数（如 OpenAI 的 reasoning_effort、Anthropic 的 thinking）无法直接传递
#   适用场景：快速原型、教学演示、需要在配置文件中动态指定模型的生产系统
#
# 方式 2：provider-specific 类 - 直接使用 ChatOpenAI/ChatAnthropic 等
#   是什么：每个 provider 维护的原生 Python 类，完整暴露该 provider 的所有能力
#   为什么需要：生产环境往往需要精细控制——超时重试、流式回调、特有参数、自定义 HTTP 客户端
#   核心优势：类型提示完整、IDE 补全友好、文档齐全、可访问 provider 100% 功能
#   代价：代码耦合到具体 provider，切换模型需修改导入和实例化代码
#   适用场景：生产系统、需要 provider 独有功能、团队已锁定单一 provider
#
# 方式 3：本地/自托管 - ChatOllama / 自定义 base_url
#   是什么：将模型推理服务部署在自己可控的基础设施上
#   为什么需要：数据隐私合规（金融/医疗/政务）、离线环境、极致成本控制、避免厂商锁定
#   核心优势：数据不出境、无 API 限流、可微调专用模型、长期成本可预测
#   代价：需运维 GPU 集群、模型版本管理、推理加速优化（vLLM/TensorRT-LLM）、冷启动延迟
#   适用场景：私有化部署、敏感数据处理、边缘计算、模型蒸馏/微调后的专用推理
#
# 选型决策树：
#   ┌─ 需要快速验证想法？ → init_chat_model
#   ├─ 生产环境 + 单一云厂商 + 需精细控制？ → provider-specific 类
#   ├─ 数据不能出公司/离线/成本敏感？ → 本地部署 (Ollama/vLLM/LocalAI)
#   └─ 多云策略/避免厂商锁定？ → init_chat_model + 环境变量切换 provider
#
# 最佳实践建议：
#   - 封装一个 ModelFactory 类，内部根据配置决定用哪种方式实例化
#   - 统一用 BaseChatModel 类型注解，业务代码只依赖接口不依赖实现
#   - 生产环境务必配置：timeout、max_retries、streaming=True、callbacks 监控
