import os
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from dotenv import load_dotenv

class ModelServiceError(RuntimeError):
    """模型服务调用失败。"""

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen3.8-max")

REQUEST_TIMEOUT = float(
    os.getenv("DASHSCOPE_TIMEOUT", "180")
)

MAX_TOKENS = int(
    os.getenv("DASHSCOPE_MAX_TOKENS", "5000")
)

ENABLE_THINKING = (
    os.getenv("DASHSCOPE_ENABLE_THINKING", "false").lower()
    == "true"
)

if not API_KEY:
    raise ModelServiceError("缺少环境变量 DASHSCOPE_API_KEY。")

if not BASE_URL:
    raise ModelServiceError(
        "缺少环境变量 DASHSCOPE_BASE_URL。"
        "示例: https://ws-exo8h1rsukcadjni.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=REQUEST_TIMEOUT,
)


def ask_deepseek(prompt):
    """
    调用阿里云百炼兼容接口。

    保留原函数名，避免影响 app.py 中的既有调用。
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """
                    你是一名非线性动力系统领域科研助手。
                    你的回答需要保持学术严谨。
                    """,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=MAX_TOKENS,
            extra_body={
                "enable_thinking": ENABLE_THINKING,
            },
        )
    except AuthenticationError as exc:
        raise ModelServiceError(
            "模型服务认证失败，请检查 API Key 是否正确，以及 Key 是否与业务空间匹配。"
        ) from exc
    except APITimeoutError as exc:
        raise ModelServiceError(
            f"模型请求超过 {REQUEST_TIMEOUT:g} 秒仍未完成，请稍后重试或减少论文内容。"
        ) from exc
    except RateLimitError as exc:
        raise ModelServiceError(
            "模型服务请求过于频繁或额度受限，请稍后重试并检查阿里云额度。"
        ) from exc
    except APIConnectionError as exc:
        raise ModelServiceError(
            "无法连接模型服务，请检查网络、代理和 Base URL。"
        ) from exc
    except BadRequestError as exc:
        raise ModelServiceError(
            "模型拒绝了当前请求，请检查模型名称、输入长度和请求参数。"
        ) from exc
    except APIStatusError as exc:
        raise ModelServiceError(
            f"模型服务返回异常状态码：{exc.status_code}。"
        ) from exc

    if not response.choices:
        raise ModelServiceError("模型没有返回任何候选结果。")

    choice = response.choices[0]
    answer = choice.message.content

    if not answer:
        raise ModelServiceError("模型返回了空内容。")

    if choice.finish_reason == "length":
        answer += (
            "\n\n---\n"
            "⚠️ 本次回答达到输出长度上限，结果可能不完整。"
        )

    return answer