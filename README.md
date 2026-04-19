# 职引 CareerPilot

职引是一个面向求职场景的智能助手项目，核心目标是把“岗位搜索、简历理解、长期偏好记忆、岗位匹配和模拟面试”串成一个可持续迭代的求职工作台。

项目当前以 FastAPI + Vue 3 为主架构，后端通过 LangGraph / LangChain 编排大模型工具调用，结合 PostgreSQL、pgvector、Redis、Ollama embedding 和通义千问等模型能力，为用户提供自然语言找工作、简历上传解析、长期记忆管理、岗位推荐和面试练习等功能。

## 功能概览

- 自然语言岗位搜索：用户可以直接输入“帮我找北京的大模型算法岗”这类请求，系统会解析意图、补全搜索计划并返回岗位推荐。
- 混合检索：结合 PostgreSQL 全文检索、pgvector 向量检索、规则过滤和 reranker 重排，提高岗位召回和排序质量。
- 长期记忆系统：从对话中抽取求职偏好，例如目标岗位、城市、薪资、经验要求、排除项和技能栈，并在后续搜索中自动复用。
- 用户可编辑记忆：前端提供记忆管理入口，用户可以修改或删除不准确的长期记忆。
- 简历上传与结构化：支持 PDF / 图片简历上传，提取文本并结构化为教育经历、工作经历、项目经历、技能等信息。
- 岗位匹配分析：基于用户简历和岗位 JD，分析匹配度、优势、风险点和优化建议。
- 模拟面试：根据岗位和用户背景生成面试问题，并支持回答评估与总结。
- 账号体系：支持注册、登录、修改密码、会话隔离和用户级数据管理。
- 评测脚本：内置岗位检索评测和记忆系统回归评测，便于迭代时检查质量回退。

## 技术栈

后端：

- Python 3.10+
- FastAPI / Uvicorn
- LangChain / LangGraph
- PostgreSQL + pgvector + pg_trgm
- Redis Streams
- Ollama embeddings
- sentence-transformers reranker
- PyMuPDF / Pillow / 可选 PaddleOCR

前端：

- Vue 3
- TypeScript
- Vite
- Pinia
- Vue Router
- Element Plus
- Axios
- ECharts

模型与服务：

- DashScope / 通义千问兼容模型
- Ollama 本地模型
- 本地 reranker 模型，可按需关闭或替换

## 项目结构

```text
find_a_good_job/
+-- backend/                 # FastAPI 后端
|   +-- config/              # 配置、模型工厂、环境变量读取
|   +-- db/                  # 数据库连接、schema、岗位检索、记忆系统
|   +-- logic/               # 业务流程编排
|   +-- middlewares/         # 中间件
|   +-- models/              # Agent、搜索计划、面试流程等模型逻辑
|   +-- routers/             # API 路由
|   +-- schemas/             # Pydantic 数据结构
|   +-- static/              # 上传文件与静态资源
|   +-- utils/               # 工具函数、简历解析、队列、监控、LangChain tools
|   +-- main.py              # FastAPI 入口
|   +-- requirements.txt
+-- frontend/                # Vue 3 前端
|   +-- src/
|   |   +-- api/             # HTTP API 封装
|   |   +-- assets/          # 前端静态资源
|   |   +-- components/      # 页面组件
|   |   +-- router/          # 路由
|   |   +-- stores/          # Pinia 状态
|   |   +-- views/           # 页面
|   +-- package.json
|   +-- vite.config.ts
+-- eval/                    # 检索与记忆系统评测脚本
+-- docs/                    # 项目设计文档和模块说明
+-- models/                  # 本地模型目录，通常不提交到 Git
+-- logs/                    # 运行日志，通常不提交到 Git
+-- README.md
```

## 环境依赖

本地运行前需要准备：

- Python 3.10 或更高版本
- Node.js 18 或更高版本
- PostgreSQL 14+，并安装 `pgvector` 扩展
- Redis，推荐开启，用于简历解析队列、缓存和限流；不可用时部分队列逻辑会退回本地队列
- Ollama，用于 embedding 和本地轻量分析模型
- DashScope API Key，用于通义千问相关模型调用

推荐提前拉取 Ollama 模型：

```bash
ollama pull nomic-embed-text-v2-moe
ollama pull qwen2.5:3b
```

如果需要启用本地重排模型，请把模型放到 `models/bge-reranker-v2-m3/`，或通过环境变量 `RERANKER_MODEL_PATH` 指向实际路径。

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd find_a_good_job
```

### 2. 创建后端环境

```bash
conda create -n careerpilot python=3.10
conda activate careerpilot
pip install -r backend/requirements.txt
```

如需解析扫描版 PDF 或图片简历，可以额外安装 PaddleOCR 相关依赖。

### 3. 配置环境变量

项目会优先读取以下位置的 `.env` 文件：

```text
config/.env
backend/config/.env
```

可以创建 `config/.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key

DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=boss_zhipin

OLLAMA_URL=http://localhost:11434
REDIS_URL=redis://localhost:6379/0

SESSION_SECRET_KEY=change-me
JWT_SECRET_KEY=change-me-too
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

可选模型配置：

```env
MODEL_ASSISTANT_MAIN=qwen-plus-2025-07-28
MODEL_ASSISTANT_PLANNER=qwen-max-2025-01-25
MODEL_ASSISTANT_REWRITE=qwen-max-2025-01-25
MODEL_OLLAMA_EMBEDDING=nomic-embed-text-v2-moe
MODEL_OLLAMA_MATCH_ANALYSIS=qwen2.5:3b
RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3
```

### 4. 初始化数据库

创建数据库后，启动后端时项目会自动创建主要 schema 和索引，包括 `jobs`、`users`、`resumes`、`memory_facts`、`conversation_states` 等表。

```sql
CREATE DATABASE boss_zhipin;
```

数据库用户需要有创建扩展的权限，因为项目会执行：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

岗位搜索依赖 `jobs` 表中的岗位数据和 embedding。首次运行如果数据库没有岗位数据，搜索接口可以启动，但不会有有效推荐结果。

公开仓库不建议提交真实岗位数据、用户简历或任何未脱敏数据。岗位数据需要自行准备，采集或导入时请遵守数据来源网站的服务条款和 robots 规则。

### 5. 启动后端

```bash
cd backend
python main.py
```

后端默认运行在：

```text
http://127.0.0.1:8000
```

也可以使用：

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端默认运行在：

```text
http://localhost:5173
```

Vite 开发服务器会把 `/api` 和 `/static` 代理到后端 `http://127.0.0.1:8000`。

## 常用命令

后端启动：

```bash
cd backend
python main.py
```

前端开发：

```bash
cd frontend
npm run dev
```

前端构建：

```bash
cd frontend
npm run build
```

前端预览：

```bash
cd frontend
npm run preview
```

记忆系统回归评测：

```bash
python eval/evaluate_memory_regression.py --no-fail-exit
```

岗位检索评测：

```bash
python eval/evaluate_retrieval.py
```

生成或更新合成检索评测集：

```bash
python eval/generate_dataset.py
```

## 记忆系统说明

当前长期记忆系统以 `memory_facts` 作为唯一事实源，不再维护独立的 profile 表。用户画像不是单独存储的一份冗余数据，而是由 `memory_facts` 动态聚合出来的读模型。

记忆写入流程：

1. 对话或搜索过程中识别用户表达的长期偏好。
2. 规则抽取和 LLM 抽取同时工作，规则负责稳定兜底，LLM 负责语义泛化。
3. 抽取结果会被标准化为统一事实结构。
4. 根据 `fact_key`、`fact_value`、极性、来源、置信度、重要性和 TTL 写入或合并到 `memory_facts`。
5. 后续搜索计划解析时，系统会读取记忆画像并补全岗位、城市、薪资、经验、排除项等字段。

记忆读取流程：

- 对话上下文会读取用户长期记忆，生成更贴合用户偏好的回复。
- 搜索计划会使用记忆补全缺失条件，但当前轮明确表达的条件优先级更高。
- 前端记忆管理可以直接修改或删除用户认为不准确的事实。

项目已内置记忆回归测试集，用于验证抽取、画像聚合、搜索计划融合、记忆修改和删除等行为。

## API 概览

主要接口挂载在 `/api` 下：

- `/api/auth`：注册、登录、鉴权、修改密码
- `/api/user`：用户信息、简历上传、简历任务状态、结构化简历
- `/api/chat`：对话、会话、流式响应、岗位推荐
- `/api/interview`：模拟面试、回答评估、面试报告
- `/api/health`：健康检查
- `/api/metrics`：运行指标

## 上传 GitHub 前建议

不要提交以下内容：

- `.env` 和任何真实密钥
- `logs/`
- `frontend/node_modules/`
- `frontend/dist/`，除非你希望仓库直接携带前端构建产物用于部署展示
- `backend/static/resumes/` 中的真实简历
- `models/` 中的大模型文件，除非明确使用 Git LFS
- `eval/results/` 中的本地评测输出

如果希望把 `docs/` 项目文档一起上传，请确认 `.gitignore` 没有继续忽略该目录。

## 许可证

当前项目尚未指定开源许可证。正式公开前建议根据用途选择合适的 License，例如 MIT、Apache-2.0 或仅作为个人作品展示。
