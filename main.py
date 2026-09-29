from models.deepseek_api import ask_deepseek


def main():
    prompt = "你好，请简单介绍你自己。"
    answer = ask_deepseek(prompt)
    print(answer)


if __name__ == "__main__":
    main()