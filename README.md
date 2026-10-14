# 非线性动力系统论文智能分析助手

一个面向非线性动力系统研究的本地论文分析与 RAG 知识库应用。

项目使用 PyMuPDF 解析论文结构，使用 SQLite 保存论文、章节和 Chunk，使用本地 Sentence Transformer 模型进行语义编码，并通过阿里云百炼的 OpenAI 兼容接口生成带原文证据的回答。Web 界面由 Gradio 构建。

## 当前功能

### 单篇论文分析

支持上传 PDF 或直接输入论文摘要，并提供四种分析模式：

- 快速阅读
- 数学模型分析
- 混沌动力学分析
- 创新点评价

PDF 处理支持：

- 单栏、双栏和部分跨栏版面
- 阅读顺序恢复
- 中英文标题识别
- 多级章节树构建
- 按章节结构选择分析内容
- Markdown 和 LaTeX 输出清理

### 科研知识库

支持：

- 从网页上传 PDF 并导入知识库
- 使用 SHA-256 检测重复文件
- 将论文、章节和 Chunk 保存到 SQLite
- 查看论文、章节数和 Chunk 数
- 删除论文及其关联内容
- 选择单篇或多篇论文作为问答范围
- 不选择论文时检索整个知识库
- 展示论文标题、章节、页码、相似度和原文证据

### RAG 问答

当前 RAG 链路包括：

```text
PDF
  ↓
版面与章节结构解析
  ↓
生成带来源信息的Chunk
  ↓
保存到SQLite
  ↓
本地Embedding编码
  ↓
向量缓存与内存索引
  ↓
语义检索与多论文证据均衡
  ↓
构建带引用的上下文
  ↓
调用大模型生成回答
  ↓
展示回答和可追踪原文证据
```

RAG 回答要求模型：

- 仅根据检索证据回答
- 证据不足时明确说明
- 使用证据编号标注关键结论
- 区分论文原文、模型推断和研究评价
- 不编造公式、参数或实验结果

## 技术栈

- Python
- Gradio
- PyMuPDF
- SQLite
- NumPy
- Sentence Transformers
- PyTorch
- OpenAI Python SDK
- 阿里云百炼 OpenAI 兼容接口
- unittest

默认本地 Embedding 模型：

```text
intfloat/multilingual-e5-small
```

## 项目结构

```text
.
├── app.py                    # Gradio Web 应用入口
├── main.py                   # 模型接口连通性测试
├── requirements.txt          # 项目依赖
├── .env.example              # 环境变量模板
│
├── database/                 # SQLite连接、表结构和仓库
├── domain/                   # Paper、Section、Chunk等领域模型
├── embedding/                # Embedding编码、向量索引和磁盘缓存
├── ingestion/                # PDF结构化入库管道
├── models/                   # 阿里云百炼模型客户端
├── prompts/                  # 单篇分析提示词
├── rag/                      # 上下文、Prompt和回答服务
├── retrieval/                # 语义检索和内容过滤
├── scripts/                  # 命令行工具与诊断脚本
├── tests/                    # 自动化测试
└── utils/                    # PDF解析、章节识别和文本处理
```

## 环境要求

- Python 3.10 或更高版本
- 可用的阿里云百炼模型服务
- 首次使用 RAG 时需要下载本地 Embedding 模型

建议创建独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果使用 NVIDIA GPU，需要确保 PyTorch 与本机 CUDA 环境兼容。

## 环境变量

复制配置模板：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写：

```dotenv
DASHSCOPE_API_KEY=你的_API_Key
DASHSCOPE_BASE_URL=你的_OpenAI_兼容接口地址
DASHSCOPE_MODEL=你的模型名称

DASHSCOPE_TIMEOUT=180
DASHSCOPE_MAX_TOKENS=5000
DASHSCOPE_ENABLE_THINKING=false
```

| 变量 | 作用 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 模型服务认证密钥 |
| `DASHSCOPE_BASE_URL` | OpenAI 兼容接口地址 |
| `DASHSCOPE_MODEL` | 使用的对话模型 |
| `DASHSCOPE_TIMEOUT` | 请求超时时间 |
| `DASHSCOPE_MAX_TOKENS` | 最大输出 Token 数 |
| `DASHSCOPE_ENABLE_THINKING` | 是否请求启用思考模式 |

`.env` 已被 Git 忽略。不要把真实 API Key 提交到仓库、日志或截图中。

## 启动 Web 应用

在项目根目录运行：

```powershell
python app.py
```

打开终端显示的本地地址后，可以使用两个功能区：

```text
📄 单篇论文分析
🔎 科研知识库问答
```

知识库页面支持：

1. 上传并导入论文
2. 查看已入库论文
3. 选择单篇或多篇论文
4. 提交研究问题
5. 查看回答和原文证据
6. 删除不再需要的论文

## 命令行导入论文

最简单的导入方式：

```powershell
python -m scripts.ingest_pdf "D:\papers\paper.pdf"
```

带元数据导入：

```powershell
python -m scripts.ingest_pdf `
    "D:\papers\paper.pdf" `
    --title "论文标题" `
    --author "作者一" `
    --author "作者二" `
    --year 2026 `
    --doi "10.xxxx/example"
```

不填写标题时使用 PDF 文件名。

## 命令行语义检索

```powershell
python -m scripts.search_papers `
    "论文使用了什么动力学分析方法？" `
    --top-k 5
```

指定 GPU：

```powershell
python -m scripts.search_papers `
    "论文使用了什么动力学分析方法？" `
    --top-k 5 `
    --device cuda
```

## 命令行 RAG 问答

```powershell
python -m scripts.ask_papers `
    "论文使用了哪些动力学分析方法？" `
    --top-k 5
```

## 查看数据库内容

```powershell
python -m scripts.inspect_paper_database
```

也可以指定论文 ID：

```powershell
python -m scripts.inspect_paper_database `
    --paper-id "paper-..."
```

## 本地数据

运行过程中会生成：

```text
data/
├── papers.db            # SQLite论文数据库
├── papers/              # 网页上传的PDF
└── vector_indexes/      # Embedding向量缓存
```

这些内容均属于本地运行数据，不会提交到 Git。

向量缓存用于避免每次启动时重新编码全部 Chunk。首次构建索引可能较慢，后续运行会直接读取磁盘缓存。

## 运行测试

项目使用 Python 标准库 `unittest`：

```powershell
python -m unittest discover -s tests -v
```

当前测试覆盖：

- PDF 章节与 Chunk 构建
- Chunk 来源、页码和哈希
- SQLite 表结构和外键约束
- 论文文件哈希与重复检测
- 论文、章节和 Chunk 仓库
- 入库事务与异常回滚
- Embedding 编码接口
- 向量索引与相似度排序
- 向量磁盘缓存
- 单篇和多篇论文检索
- 参考文献内容过滤
- 多论文证据来源均衡
- RAG 上下文与 Prompt
- 模型客户端错误处理
- Gradio RAG 服务
- 论文统计与安全删除

当前共有 132 项自动化测试。

## 隐私与安全

- Embedding 默认在本地运行。
- PDF 原文和 SQLite 数据库存储在本地。
- 最终 RAG 上下文会发送给 `.env` 中配置的模型服务。
- 处理未公开论文或内部资料前，请确认模型服务的数据处理条款。
- 删除论文时，系统只自动删除 `data/papers/` 中受管理的 PDF。
- 通过外部路径导入的原始 PDF 不会被自动删除。

## 当前限制

- 扫描版 PDF 尚未接入 OCR。
- 图片、表格和公式尚未实现可靠的结构化识别。
- PDF 版面和标题识别仍基于启发式规则。
- 快速阅读仍使用旧的固定文本块策略。
- 当前向量检索为本地内存索引，不适合超大规模论文库。
- 尚未实现关键词与向量混合检索、Reranker 和系统化 RAG 评估。
- 检索效果仍受 PDF 解析质量、Chunk 划分和 Embedding 模型影响。
- 当前系统还不是自主 Agent，尚未接入 Tool Calling 和任务工作流。

## 后续规划

下一阶段将从完整 RAG 应用继续扩展到 AI Agent：

1. 为现有能力定义结构化工具接口
2. 学习并接入 Function Calling / Tool Calling
3. 让模型自主选择论文查询、检索和管理工具
4. 使用 LangGraph 构建有状态 Agent 工作流
5. 增加结构化输出和任务状态
6. 建立 RAG 与 Agent 评估集
7. 使用 FastAPI 提供服务接口
8. 使用 Docker 完成可复现部署

后续质量优化包括：

- OCR 与扫描版 PDF 支持
- 表格和公式提取
- 混合检索
- Reranker
- 查询改写
- 知识卡片
- 多论文对比
- 自动文献综述