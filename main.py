import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")
model_name = os.getenv("DASHSCOPE_MODEL", "qwen3.8-max")

if not api_key:
    raise ValueError("缺少环境变量 DASHSCOPE_API_KEY。")

if not base_url:
    raise ValueError(
        "缺少环境变量 DASHSCOPE_BASE_URL。"
        "示例: https://ws-exo8h1rsukcadjni.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，你是谁？"},
    ],
)

print(response.choices[0].message.content)