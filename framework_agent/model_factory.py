import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from models.deepseek_api import (
    DEFAULT_ENABLE_THINKING,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_REQUEST_TIMEOUT,
)

load_dotenv()


class LangChainModelConfigurationError(
    RuntimeError
):
    """LangChain模型配置不完整。"""


def create_aliyun_langchain_model(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> ChatOpenAI:
    """创建连接阿里云兼容接口的LangChain模型。"""

    cleaned_model_name = model_name.strip()

    if not cleaned_model_name:
        raise ValueError(
            "模型名称不能为空。"
        )

    if request_timeout <= 0:
        raise ValueError(
            "请求超时时间必须大于0。"
        )

    if max_tokens <= 0:
        raise ValueError(
            "最大输出Token数必须大于0。"
        )

    if temperature < 0:
        raise ValueError(
            "temperature不能小于0。"
        )

    if max_retries < 0:
        raise ValueError(
            "最大重试次数不能小于0。"
        )

    resolved_api_key = (
        api_key
        or os.getenv(
            "DASHSCOPE_API_KEY"
        )
    )

    resolved_base_url = (
        base_url
        or os.getenv(
            "DASHSCOPE_BASE_URL"
        )
    )

    if not resolved_api_key:
        raise (
            LangChainModelConfigurationError(
                "缺少环境变量"
                "DASHSCOPE_API_KEY。"
            )
        )

    if not resolved_base_url:
        raise (
            LangChainModelConfigurationError(
                "缺少环境变量"
                "DASHSCOPE_BASE_URL。"
            )
        )

    return ChatOpenAI(
        model=cleaned_model_name,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        timeout=request_timeout,
        temperature=temperature,
        max_retries=max_retries,
        extra_body={
            "enable_thinking": (
                enable_thinking
            ),
            "max_tokens": max_tokens,
        },
    )

