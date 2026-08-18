import os
from openai import OpenAI
from dotenv import load_dotenv


# 加载.env
load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)



def ask_deepseek(prompt):

    response = client.chat.completions.create(

        model="deepseek-v4-pro",

        messages=[

            {
                "role": "system",
                "content":
                """
                你是一名非线性动力系统领域科研助手。
                你的回答需要保持学术严谨。
                """
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )


    return response.choices[0].message.content