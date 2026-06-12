"""多模态输入（图片）与聊天模型示例。"""

import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o", temperature=0)


# 1. 从 URL 加载图片
def image_from_url():
    message = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image? Describe it."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
                },
            },
        ]
    )
    result = model.invoke([message])
    print(f"1. Image from URL:\n{result.content}\n")


# 2. 从本地文件加载图片（base64）
def image_from_local(image_path: str):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract any text you see in this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                },
            },
        ]
    )
    result = model.invoke([message])
    print(f"2. Image from local file:\n{result.content}\n")


# 3. 多图像比较
def compare_images():
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Compare these two images."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                },
            },
        ]
    )
    result = model.invoke([message])
    print(f"3. Image comparison:\n{result.content}\n")


if __name__ == "__main__":
    image_from_url()
    # image_from_local("path/to/image.png")  # Uncomment with your image path
    compare_images()

# =============================================================================
# 教学备注：多模态输入——让 LLM "看见"图片
# =============================================================================
# 核心问题：之前的例子模型只能读文字——但如果用户说"这张图里写了什么？"
#   传统方案：OCR 提取文字 -> 发给模型（两步，额外成本，丢失图片语义信息）
#   多模态方案：直接把图片发给模型，模型自己"看"并理解
#   本质区别：模型不仅能读图片中的文字，还能理解布局、颜色、物体关系、图表趋势
#
# 消息结构——HumanMessage.content 不仅仅是个字符串
#   之前的理解：HumanMessage(content="你好") —— content 是字符串
#   多模态的理解：content 是一个列表，元素可以是不同类型
#     {"type": "text", "text": "描述这张图"}  # 文字
#     {"type": "image_url", "image_url": {"url": "..."}}  # 图片
#   为什么这样设计：让"文字+多张图片+其他媒体"在一条消息里按顺序传递
#     模型看到的是：用户先说了一段话 -> 看了第一张图 -> 又看了第二张图
#     这种"上下文顺序"在单字符串中是无法表达的
#
# 图片输入方式 1：URL（网络图片）
#   是什么：传入图片的公开 URL，模型服务端自行下载
#   为什么需要：不必先把图片传给我们服务器，再传给模型——省带宽省时间
#   底层：OpenAI 收到 URL 后自己去下载图片
#     但注意：图片体积仍然计入 token（按图片尺寸折算）
#   限制：
#     - URL 必须公开可达（内网/本地文件 URL 不行）
#     - URL 必须稳定（不会在模型处理时过期）
#     - 不支持 HTTPS 证书异常的 URL
#   适用场景：CDN 托管的图片、公开网络图片、不需要上传场景
#
# 图片输入方式 2：Base64（本地/私有图片）
#   是什么：把图片文件编码为 base64 字符串，嵌入到 data URI 中
#   为什么需要：用户上传的图片、内网资源、动态生成的图表——没有公开 URL
#   流程：读文件 -> base64.b64encode() -> data:image/png;base64,... -> 传给模型
#   代价：base64 编码让体积膨胀约 33%；大图（10MB）编码后 ~13MB，可能超 token 限制
#   关键理解：token 消费——
#     GPT-4o 对图片的 token 计算 = f(width, height, detail_level)
#     一张 1024x1024 的图在 "high" detail 下约消耗 255 token
#     不是 base64 字符串的字节数/4，而是图片尺寸的折算值
#   适用场景：用户上传图片、私有数据、实时截图
#
# 方式 3：多图同时输入（对比/融合理解）
#   是什么：在一条消息中传入多张图片，让模型比较或综合分析
#   为什么需要：很多任务需要"对比"——
#     "这两张截图有什么不同？"、"帮我从这几张图中找出猫"
#     "这张图是原始设计，这张是修改建议，请评估变化"
#   注意事项：token 是每张图分别计算的，多图 = 乘法增长
#     如果每张图 255 token，5 张图就是 1275 token——还在合理范围
#     但 20 张高细节图可能超过上下文窗口
#   适用场景：A/B 对比、多视角识别、图片间关系分析
#
# 混合消息体的设计思路（理解这个比记住 API 更重要）：
#   一条 HumanMessage 的 content 就是"你说的内容"的有序列表：
#     [文字讲解, 图A, 图B, "现在比较这两张", 文字讲解, 图C]
#   模型按顺序"看到"所有这些内容，就像你在对话中依次展示图片
#   你可以反复插文字说明来引导模型的注意力——
#     "先看图A" -> 图A -> "再看图B" -> 图B -> "图A和图B有什么区别？"
#
# 视频/音频怎么处理？
#   当前主流多模态模型原生支持：图片 + 文字 + 音频（GPT-4o 支持音频输入）
#   视频：模型不支持视频输入，需自行抽帧 -> 多张图片输入
#     方案：ffmpeg 抽关键帧 -> 每隔 2 秒一张 -> 多图片输入 -> 模型理解视频内容
#   音频：GPT-4o 原生支持音频（直接传入音频字节）；其他模型如 Claude 不支持
#     方案：Whisper 转文字 -> 文字输入（通用兼容方案）
#
# 多模态模型选型指南：
#   GPT-4o：最均衡的选择——速度快、支持音频、图片理解能力强、token 价格适中
#   GPT-4o-mini：更便宜、更快，适合简单图片理解（截图、文档、图表）
#     注意：mini 的图片理解能力低于 full 版本，复杂图表分析可能不够
#   Claude 3.5 Sonnet：图表/文档理解非常强，适合"密集文字+复杂布局"的图片
#   Gemini 1.5 Pro/Flash：上下文超长（百万 token），适合大量图片帧分析
#   开源：LLaVA-NeXT、Qwen2-VL、InternVL2——可本地部署，但推理速度/准确率不如商业
#
# 图片预处理的关键性（强烈建议实践）：
#   压缩：长边 1024px 以内，JPEG 质量 85%
#     -> 文件体积从 5MB 降到 200KB，token 不变（按尺寸算）
#     -> 传输更快、延迟更低
#   裁剪：裁掉无关边框、空白、签名区域
#     -> 减少 token（图片尺寸缩小），提高模型注意力
#   格式：PNG 用于文字密集图（截图/文档），JPEG 用于照片
#
# 教学建议顺序：
#   1. 先用 URL 方式，这是最简单的体验方式
#   2. 理解 HumanMessage content 从"字符串"到"列表"的转变
#   3. 再学习 Base64 方式处理本地图片
#   4. 多图对比是高级用法，理解逻辑即可
#   5. 图片预处理和 token 意识是生产环境的关键
