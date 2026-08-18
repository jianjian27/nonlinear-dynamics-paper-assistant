# 非线性动力系统论文阅读助手

一个面向非线性动力系统论文的本地 AI 辅助阅读工具。项目使用 Gradio 提供 Web 界面，通过兼容 OpenAI SDK 的 DeepSeek API 分析论文 PDF 或用户粘贴的摘要。

## 主要功能

- 上传 PDF 并提取逐页文本，或直接粘贴论文摘要
- 提供四种分析模式：
  - 快速阅读：梳理研究背景、问题、方法和主要贡献
  - 数学模型分析：提取方程、变量和参数，并分析稳定性与动力学性质
  - 混沌动力学分析：关注 Lyapunov 指数、分岔图、Poincare 截面、功率谱和吸引子等内容
  - 创新点评价：从已有研究、创新类型、创新程度和不足等角度进行评价
- 清理模型输出中的部分 Markdown 与 LaTeX 格式问题

## 工作流程

1. `app.py` 接收 PDF、摘要和分析模式。
2. `utils/pdf_reader.py` 使用 PyMuPDF 提取 PDF 每一页的文本。
3. `utils/text_splitter.py` 按约 4000 个字符切分文本，相邻文本块保留 500 个字符的重叠。
4. `utils/chunk_selector.py` 根据分析模式选择部分文本块。
5. `prompts/` 中对应的提示词模板与论文文本组合后发送给 DeepSeek。
6. `utils/markdown_cleaner.py` 清理返回内容，并在 Gradio 页面中展示结果。

## 项目结构

```text
.
├── app.py                         # Gradio Web 应用入口
├── main.py                        # DeepSeek API 连通性测试脚本
├── agent.py                       # 预留文件，当前为空
├── tools.py                       # 预留文件，当前为空
├── models/
│   └── deepseek_api.py            # DeepSeek 客户端及请求封装
├── prompts/
│   ├── paper_summary_prompt.py    # 快速阅读提示词
│   ├── model_analysis_prompt.py   # 数学模型分析提示词
│   ├── chaos_analysis_prompt.py   # 混沌动力学分析提示词
│   └── innovation_prompt.py       # 创新点评价提示词
├── utils/
│   ├── pdf_reader.py              # PDF 文本提取
│   ├── text_splitter.py           # 文本分块
│   ├── chunk_selector.py          # 按分析模式选择文本块
│   └── markdown_cleaner.py        # Markdown/LaTeX 输出清理
└── notebooks/
    └── 测试存储位置.ipynb          # 实验 Notebook
```

## 环境要求

- Python 3.10 或更高版本
- DeepSeek API Key

安装当前源码使用的依赖：

```powershell
pip install gradio openai python-dotenv pymupdf
```

## 配置

在项目根目录创建 `.env` 文件：

```dotenv
DEEPSEEK_API_KEY=你的_API_Key
```

`.env` 已被 `.gitignore` 排除，请勿将真实密钥提交到 Git 或上传到 GitHub。

项目当前在 `models/deepseek_api.py` 和 `main.py` 中使用以下模型名称：

```text
deepseek-v4-pro
```

如果账户不可使用该模型，请根据 DeepSeek API 当前支持的模型名称修改配置。

## 运行应用

在项目根目录执行：

```powershell
python app.py
```

Gradio 启动后会在终端显示本地访问地址。打开该地址，上传 PDF 或填写摘要，选择分析模式并点击“开始分析”。

也可以运行最小 API 测试：

```powershell
python main.py
```

该脚本会向 DeepSeek 发送一条简单消息，并在终端打印回复。

## 当前限制

- 文本块选择采用固定位置规则，并非关键词检索、向量检索或语义排序，可能遗漏论文中的重要章节。
- 不同分析模式只会选择部分文本块发送给模型，并不一定覆盖整篇论文。
- PDF 文字提取依赖文档本身的文本层；扫描版 PDF 尚未集成 OCR。
- 当前没有自动化测试、异常处理、请求重试和输入长度控制。
- `agent.py` 与 `tools.py` 目前是空的预留文件。

## 隐私提示

上传的论文文本会通过 API 发送给所配置的模型服务。处理未公开论文、内部资料或其他敏感内容前，请先确认相应的数据使用与保密要求。
