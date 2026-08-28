import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen3.8-max")

if not API_KEY:
    raise ValueError("缺少环境变量 DASHSCOPE_API_KEY。")

if not BASE_URL:
    raise ValueError(
        "缺少环境变量 DASHSCOPE_BASE_URL。"
        "示例: https://ws-exo8h1rsukcadjni.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


def ask_deepseek(prompt):
    """
    保留原函数名，避免改动 app.py 里的既有调用。
    实际底层已切换为阿里云百炼兼容接口。
    """
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
    )

    return response.choices[0].message.content