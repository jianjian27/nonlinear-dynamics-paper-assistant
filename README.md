# 非线性动力系统论文智能分析助手

一个面向非线性动力系统研究的本地论文分析、RAG 知识库与科研 Agent 应用。

项目使用 PyMuPDF 解析论文结构，使用 SQLite 保存论文、章节和 Chunk，使用本地 Sentence Transformer 模型进行语义编码，并通过阿里云百炼的 OpenAI 兼容接口完成 RAG 问答和 Tool Calling。Web 界面由 Gradio 构建。

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

### 科研 Agent

科研 Agent 可以根据用户任务自主选择并连续调用工具，而不是按照固定流程执行一次检索。当前提供三个只读工具：

- `list_papers`：查看知识库中的论文
- `get_paper_overview`：读取指定论文的元数据和章节结构
- `search_paper_evidence`：检索带论文、章节和页码的原文证据

Agent 运行链路包括：

```text
用户研究任务
  ↓
模型分析任务并选择工具
  ↓
JSON Schema参数验证
  ↓
工具注册表与安全调度器
  ↓
读取论文列表、章节结构或原文证据
  ↓
模型根据工具结果决定继续调用或生成回答
  ↓
展示最终回答和完整工具调用轨迹
```

运行时设置了模型轮数和工具调用次数上限，并对未知工具、重复调用 ID、无效参数和异常模型消息进行处理。

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
- Pydantic
- JSON Schema / Tool Calling
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
├── agent_app/                # Agent组装与Gradio服务
├── agent_runtime/            # 消息协议、工具调度和多轮运行循环
├── agent_tools/              # 科研Agent的结构化工具
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

打开终端显示的本地地址后，可以使用三个功能区：

```text
📄 单篇论文分析
🤖 科研 Agent
🔎 科研知识库问答
```

知识库页面支持：

1. 上传并导入论文
2. 查看已入库论文
3. 选择单篇或多篇论文
4. 提交研究问题
5. 查看回答和原文证据
6. 删除不再需要的论文

科研 Agent 页面支持输入复合研究任务，并展示模型轮数、工具调用数、调用参数、执行状态和工具调用轨迹。

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

## 命令行科研 Agent

查看知识库中的论文：

```powershell
python -m scripts.ask_agent `
    "知识库中有哪些论文？" `
    --show-trace
```

执行需要论文证据的研究任务：

```powershell
python -m scripts.ask_agent `
    "知识库中的论文使用了哪些动力学分析方法？请注明章节和页码。" `
    --device cuda `
    --show-trace
```

`--show-trace` 会显示 Agent 的工具选择、调用参数和执行结果，便于检查回答依据与运行过程。

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
- Agent工具定义、参数验证与注册
- 工具安全调度与结构化错误
- Tool Calling消息协议与模型适配
- 多轮Agent循环、终止条件和运行限制
- Agent命令行入口与Gradio服务
- Agent网页回调与索引重置

## 隐私与安全

- Embedding 默认在本地运行。
- PDF 原文和 SQLite 数据库存储在本地。
- 最终 RAG 上下文会发送给 `.env` 中配置的模型服务。
- Agent 调用工具获得的论文证据和工具结果可能会发送给模型服务。
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
- 当前 Agent 每次执行一个独立任务，尚未实现跨会话记忆和长期任务状态。
- 当前 Agent 工具均为只读查询工具，不会自主修改或删除论文数据。
- 当前多轮 Agent 使用自研轻量运行循环，尚未接入 LangGraph 等有状态工作流框架。

## 后续规划

下一阶段将继续提高 Agent 的可验证性与工程交付能力：

1. 建立 RAG 与 Agent 评估集和质量指标
2. 增加查询改写、混合检索与 Reranker
3. 为多论文比较等任务增加结构化输出
4. 增加会话状态和可恢复的任务执行记录
5. 根据工作流复杂度评估并接入 LangGraph
6. 使用 FastAPI 提供服务接口
7. 使用 Docker 完成可复现部署
8. 增加日志、配置管理和基础可观测性

后续质量优化包括：

- OCR 与扫描版 PDF 支持
- 表格和公式提取
- 混合检索
- Reranker
- 查询改写
- 知识卡片
- 多论文对比
- 自动文献综述
