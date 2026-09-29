# 非线性动力系统论文阅读助手

一个面向非线性动力系统论文的本地 AI 辅助阅读工具。项目使用 Gradio 提供 Web 界面，通过阿里云百炼的 OpenAI 兼容接口分析论文 PDF 或用户粘贴的摘要。

项目目前支持单篇论文分析，并已完成面向 RAG 的可追踪 Chunk 基础；论文数据库、Embedding、向量检索和知识库问答仍在开发中。

## 主要功能

- 上传 PDF 并提取文本，或直接粘贴论文摘要
- 对单栏、双栏和部分跨栏 PDF 恢复基本阅读顺序
- 识别中英文标题、章节编号和多级章节结构
- 提供四种分析模式：
  - 快速阅读：梳理研究背景、问题、方法和主要贡献
  - 数学模型分析：提取方程、变量和参数，并分析稳定性与动力学性质
  - 混沌动力学分析：关注 Lyapunov 指数、分岔图、Poincaré 截面、功率谱和吸引子等内容
  - 创新点评价：从已有研究、创新类型、创新程度和不足等角度进行评价
- 对非“快速阅读”模式按章节标题选择相关内容
- 处理模型调用中的认证失败、超时、限流、网络异常、异常状态码和空响应
- 清理模型输出中的部分 Markdown 与 LaTeX 格式问题
- 将章节树转换为带章节路径、页码、来源文本块和内容哈希的 Chunk

## 当前工作流程

### 快速阅读

```text
PDF
  ↓
逐页提取文本
  ↓
按字符长度切分并保留重叠
  ↓
选择前部文本块
  ↓
组装提示词并调用模型
  ↓
清理 Markdown/LaTeX 并展示结果
```

### 其他分析模式

```text
PDF
  ↓
提取文本块和文本行
  ↓
分析版面并恢复阅读顺序
  ↓
识别标题和章节层级
  ↓
构建章节树
  ↓
根据分析模式选择相关章节
  ↓
组装提示词并调用模型
  ↓
清理 Markdown/LaTeX 并展示结果
```

### RAG 基础链路

```text
章节树
  ↓
遍历章节并生成完整章节路径
  ↓
按文本块边界构建 Chunk
  ↓
记录页码、来源块和内容哈希
```

这条 RAG 基础链路目前尚未接入 Gradio，也尚未写入数据库或向量索引。

## 项目结构

```text
.
├── app.py                         # Gradio Web 应用入口
├── main.py                        # 模型 API 连通性测试脚本
├── agent.py                       # 预留文件，当前为空
├── tools.py                       # 预留文件，当前为空
├── requirements.txt               # Python 依赖及已验证版本
├── .env.example                   # 环境变量安全模板
├── domain/
│   └── chunk.py                   # 可追踪 Chunk 数据模型
├── ingestion/
│   └── chunk_builder.py           # 章节树到 Chunk 的转换
├── models/
│   └── deepseek_api.py            # 百炼 OpenAI 兼容客户端及错误处理
├── prompts/
│   ├── paper_summary_prompt.py    # 快速阅读提示词
│   ├── model_analysis_prompt.py   # 数学模型分析提示词
│   ├── chaos_analysis_prompt.py   # 混沌动力学分析提示词
│   └── innovation_prompt.py       # 创新点评价提示词
├── scripts/
│   └── inspect_pdf_layout.py      # PDF 版面诊断脚本
├── tests/
│   └── test_chunk_builder.py      # Chunk 构建器单元测试
├── utils/
│   ├── pdf_reader.py              # 旧的逐页 PDF 文本提取
│   ├── text_splitter.py           # 旧的按字符长度切分逻辑
│   ├── pdf_structure.py           # 文本行、文本块和版面阅读顺序
│   ├── heading_detector.py        # 标题识别和层级判断
│   ├── section_builder.py         # 章节划分和章节树构建
│   ├── paper_parser.py            # 结构化 PDF 解析入口
│   ├── chunk_selector.py          # 固定文本块和章节选择逻辑
│   └── markdown_cleaner.py        # Markdown/LaTeX 输出清理
└── notebooks/
    └── 测试存储位置.ipynb          # 实验 Notebook
```

`models/deepseek_api.py` 和 `ask_deepseek()` 保留了早期命名，以避免影响现有调用；当前实际接入的是由环境变量配置的阿里云百炼 OpenAI 兼容接口。

## 环境要求

- Python 3.10 或更高版本
- 可用的阿里云百炼 OpenAI 兼容接口及 API Key

建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 配置

复制安全模板：

```powershell
Copy-Item .env.example .env
```

然后在本地 `.env` 中填写真实配置：

```dotenv
DASHSCOPE_API_KEY=你的_API_Key
DASHSCOPE_BASE_URL=你的_OpenAI_兼容接口地址
DASHSCOPE_MODEL=你的模型名称
DASHSCOPE_TIMEOUT=180
DASHSCOPE_MAX_TOKENS=5000
DASHSCOPE_ENABLE_THINKING=false
```

| 环境变量 | 作用 | 是否必填 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 模型服务认证密钥 | 是 |
| `DASHSCOPE_BASE_URL` | OpenAI 兼容接口地址 | 是 |
| `DASHSCOPE_MODEL` | 模型名称 | 否，代码中有默认值 |
| `DASHSCOPE_TIMEOUT` | 请求超时秒数 | 否，默认 `180` |
| `DASHSCOPE_MAX_TOKENS` | 最大输出 token 数 | 否，默认 `5000` |
| `DASHSCOPE_ENABLE_THINKING` | 是否请求模型启用思考模式 | 否，默认 `false` |

`.env` 已被 `.gitignore` 排除，`.env.example` 只包含占位值。请勿把真实 API Key 写入提交、日志、截图或公开仓库。如果密钥曾经泄露，应立即轮换。

## 运行应用

在项目根目录执行：

```powershell
python app.py
```

Gradio 启动后会在终端显示本地访问地址。打开该地址，上传 PDF 或填写摘要，选择分析模式并点击“开始分析”。

也可以运行最小模型 API 测试：

```powershell
python main.py
```

该脚本会向当前配置的模型发送一条简单消息，并在终端打印回复。运行它会产生一次真实 API 请求。

## 运行测试

当前自动化测试使用 Python 标准库 `unittest`，不需要额外安装测试框架：

```powershell
python -m unittest discover -s tests -v
```

当前测试主要覆盖：

- 父子章节分别生成 Chunk，避免正文重复
- 完整章节路径
- 精确页码和来源文本块
- 按文本块边界切分与重叠
- Chunk ID 和内容哈希的稳定性
- 非法切分参数校验

## 当前 RAG 开发状态

已经完成：

- 可追踪 Chunk 数据模型
- 章节树深度优先遍历和章节路径生成
- 章节直属正文到 Chunk 的转换
- 页码、来源文本块和内容哈希记录
- Chunk 构建器基础单元测试

尚未完成：

- SQLite 论文、章节和 Chunk 持久化
- 论文文件哈希和重复导入检测
- Embedding 模型接入
- 向量索引和语义检索
- 带论文、章节和页码引用的 RAG 问答
- Gradio 科研知识库问答界面
- RAG 检索与回答质量评估集

## 当前限制

- “快速阅读”仍使用旧的固定文本块选择逻辑，尚未统一到结构化章节流程。
- 非“快速阅读”模式主要依赖章节标题关键词选择内容，并非 Embedding 语义匹配。
- 长论文尚未建立统一的 token 预算、多批次处理和分层汇总机制。
- PDF 文字提取依赖文档本身的文本层；扫描版 PDF 尚未集成 OCR。
- 图片、表格和数学公式尚未进行可靠的结构化提取。
- 版面和标题识别采用启发式规则，对特殊论文模板仍可能判断错误。
- 自动化测试目前主要覆盖 Chunk 构建器，尚未覆盖完整 PDF 解析和 Gradio 应用流程。
- 模型调用已有常见错误处理，但尚未实现自动重试。
- `agent.py` 与 `tools.py` 目前仍是空的预留文件。

## 隐私提示

上传的论文文本会发送到 `.env` 中配置的模型服务。处理未公开论文、内部资料或其他敏感内容前，请确认所使用服务的数据处理条款和所在机构的保密要求。
