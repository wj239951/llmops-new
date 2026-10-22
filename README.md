# LLMOps 平台 · 工业知识库 RAG 问答

> 一个可本地部署的一站式 LLMOps 平台：**Prompt 管理 + 知识库 RAG + 多模型对话 + 会话记忆 + 调用日志**。
> 后端 FastAPI，前端 Vue3 + TypeScript，向量检索 Chroma（bge-m3），元数据 MySQL 8。

`后端 pytest 17 passed` · `前端 vue-tsc 零报错` · `Docker Compose 一键启动` · `支持 Windows 原生部署`

---

## 这是什么

这不是一个「调通大模型 API 就算完成」的 Demo。它把一条完整的 RAG 链路拆开、测量、再一步步修：

- **知识怎么进来**：手动录入 / 文件上传（txt、md、pdf、py、json）→ 解析 → 按中文标点切块 → bge-m3 向量化 → 写入 Chroma；
- **怎么被检索到**：余弦相似度取宽候选 → 阈值过滤 → 截断 → 拼上下文 → 交给模型；
- **怎么判断「命中了但不可靠」**：top1 只是勉强过线时降级为 `weak_hit`，前端明确告警，而不是假装命中；
- **怎么保证两套存储不打架**：MySQL 与 Chroma 的写/删同步，孤儿向量会被记录而不是静默吞掉。

阈值、余量、候选池这些数字不是拍出来的，是在本机语料上做阈值扫描标定，并用 12 组实验（换机器、换语料、换阈值）验证过稳定性。相关结论已沉淀为代码里的注释和常量。

## 架构

```
┌──────────────────────────────────────┐
│ 浏览器  Vue3 + TypeScript + Element UI│
└──────────────────┬───────────────────┘
                   │ /api/*  Authorization: Bearer <DEMO_PASSWORD>
┌──────────────────▼───────────────────┐
│ FastAPI (uvicorn :8000)              │
│   ChatService  RAGService            │
│   PromptService  FileParser          │
└──┬───────────────┬───────────────┬───┘
   │               │               │
   ▼               ▼               ▼
 MySQL 8        Chroma            LLMService
 4 张表         cosine 空间        ├─ Ollama  deepseek-r1:7b（本地）
                bge-m3 · 1024 维   └─ DeepSeek API（云端）
```

## 核心工程问题与解法

这是本项目真正花时间的地方，也是它区别于普通 CRUD 的地方：

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| 中文整句提问检索 **0 命中** | 关键词方案按空格切词，`润滑脂型号是什么` 被切成 1 个 token，`LIKE` 要求整句连续出现 | 改用 **bge-m3 向量检索**，整句编码，不再依赖分词 |
| 写入 384 维、读取 1024 维 | 直接调 `_collection.add` 会走 chromadb 自带的默认 embedding | 统一走 LangChain `add_texts`，锁定 bge-m3 |
| 所有相似度整体偏高、区分度差 | 向量化时拼了标题前缀，同文档各 chunk 共享同一段开头 | **只对正文向量化**，标题放 metadata |
| 正确片段永远进不了候选集 | 先 `k=3` 截断，再按阈值过滤，第 4 名及以后被提前丢掉 | 先取 `RAG_CANDIDATE_K=6` 宽候选 → 阈值过滤 → 截 `RAG_FEED_TOPK=3` |
| 阈值没有依据 | 拍脑袋设一个数，换个问法就崩 | 31 行阈值 × 6 指标扫描标定 **0.53**；top1 过线但余量 < `0.06` 判 **`weak_hit`**，前端告警 |
| MySQL 与向量库不一致 | 两套存储各自写，删了文档向量还在 | 更新/删除先落 MySQL 再同步删向量；反查不到文档的孤儿向量记 `warning` |
| 删除不存在的文档却返回 200 | 只做 `query().delete()`，不校验受影响行 | 删除前查存在，不存在返回 **404**，与 `PUT` 语义对齐 |

## 功能

- **对话**：本地 Ollama（deepseek-r1:7b）与 DeepSeek 云端双通道，可选 Prompt 模板、温度 / top_p、思考过程、Token 与耗时统计
- **知识库**：手动录入、文件上传解析、增删改查，切块与向量同步，来源与命中分数回显
- **RAG**：余弦检索、阈值过滤、宽候选池、弱命中降级（`hit` / `weak_hit` / `no_hit` / `off` 四态）
- **会话记忆**：按 `conversation_id` 取最近 `MEMORY_MAX_TURNS` 轮成功对话，支持会话列表与历史消息
- **调用日志**：分页查询，带 token、耗时、状态、错误信息

## 快速开始

### 方式 A：Docker Compose（推荐）

前置：Docker Desktop；若走本地模型还需 Ollama 并拉好模型。

```bash
ollama pull deepseek-r1:7b   # 对话模型（约 4.7GB）
ollama pull bge-m3           # 向量化模型（约 1.2GB，缺了检索直接 0 命中）

cp .env.example .env         # Windows: copy .env.example .env
# 编辑 .env，至少改掉 MYSQL_PASSWORD（只用大小写字母+数字）

docker compose up -d --build
docker compose exec backend python tools/reindex_knowledge.py   # 首次必做：重建向量库
```

打开 **http://localhost:3000**，演示口令默认为 `demo2026`；接口文档在 **http://localhost:8000/docs**。

不用 Ollama 也可以：`.env` 里设 `EMBEDDING_PROVIDER=siliconflow` + 填 `SILICONFLOW_API_KEY`，对话通道填 `DEEPSEEK_API_KEY`。两者同为 bge-m3 权重，向量互通。

### 方式 B：Windows 原生（无需 Docker）

前置：MySQL 80 服务、Ollama（同上两个模型）、Python 3.11 虚拟环境、Node.js。

```powershell
# 1) 导入库结构与数据（覆盖空库）
mysql -uroot -p < mysql-init/llmops_db.sql

# 2) 装依赖
cd llmops-backend
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
cd ..\llmops-frontend; npm ci; npm run build-only

# 3) 一键启停
.\start.ps1     # backend :8000 + frontend :3000
.\stop.ps1
```

## 配置

`.env.example` 是完整模板；`RAG_*` 三项在代码里有默认值，不写也能跑。

| 变量 | 默认 / 说明 |
| --- | --- |
| `MYSQL_HOST` `MYSQL_PORT` `MYSQL_DATABASE` `MYSQL_USER` `MYSQL_PASSWORD` | 数据库连接，`MYSQL_PASSWORD` 必填 |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_EMBEDDING_MODEL` | `127.0.0.1:11434` / `deepseek-r1:7b` / `bge-m3` |
| `EMBEDDING_PROVIDER` | `ollama`（本地）或 `siliconflow`（云端，无需 Ollama） |
| `SILICONFLOW_API_KEY` | 仅云端向量化时必填 |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | 云端对话通道，不填则只有本地 Ollama |
| `DEMO_PASSWORD` | 前后端共用的演示口令，默认 `demo2026` |
| `RAG_SCORE_THRESHOLD` | 余弦命中阈值，默认 `0.53` |
| `RAG_CANDIDATE_K` / `RAG_FEED_TOPK` / `RAG_WEAK_MARGIN` | 代码默认 `6` / `3` / `0.06` |
| `MEMORY_MAX_TURNS` | 会话记忆加载轮数，默认 `10` |

## 目录结构

```
.
├── docker-compose.yml          # mysql + backend + frontend 三服务
├── .env.example                # 环境变量模板
├── start.ps1 / stop.ps1        # Windows 原生部署启停
├── DEPLOY.md                   # 完整部署 / 自检 / 排障手册
├── docs/                       # 开发顺序与主链路笔记
├── mysql-init/                 # MySQL 镜像 + 全量初始化 SQL
├── llmops-backend/
│   ├── app/
│   │   ├── api/                # chat / prompt / knowledge / logs / conversations
│   │   ├── core/               # config / database / security（Bearer 口令鉴权）
│   │   ├── models/             # prompt / knowledge / chat_log / conversation
│   │   ├── schemas/
│   │   ├── services/           # rag / chat / llm / ollama / deepseek / prompt / file_parser
│   │   └── main.py
│   ├── tests/                  # pytest：Prompt CRUD、RAG 过滤、chat 流程
│   ├── tools/reindex_knowledge.py   # 从 MySQL 重建向量库
│   └── Dockerfile
└── llmops-frontend/
    ├── src/views/              # ChatStudio / PromptManager / KnowledgeBase / runlogs
    ├── src/api/                # axios 封装 + 资源接口
    ├── src/router/
    ├── vite.config.ts          # 含 preview.proxy 反代
    └── Dockerfile + nginx.conf
```

## API

除 `GET /` 外，`/api/*` 均需请求头 `Authorization: Bearer <DEMO_PASSWORD>`。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 服务信息（免鉴权） |
| `POST` | `/api/chat` | 对话：RAG / 会话记忆 / 模型选择 |
| `GET` `POST` | `/api/prompts` | Prompt 列表 / 新建 |
| `PUT` `DELETE` | `/api/prompts/{id}` | 更新 / 删除（不存在 404） |
| `GET` `POST` | `/api/knowledge` | 文档列表 / 新建（切块 + 向量化） |
| `POST` | `/api/knowledge/upload` | 上传文件解析入库 |
| `PUT` `DELETE` | `/api/knowledge/{id}` | 更新（重建向量）/ 删除（同步删向量，不存在 404） |
| `GET` | `/api/logs` | 调用日志，支持 `page` / `page_size` / `status` |
| `GET` `POST` | `/api/conversations` | 会话列表 / 新建 |
| `GET` | `/api/conversations/{id}/messages` | 会话历史消息 |
| `DELETE` | `/api/conversations/{id}` | 删除会话及其记录 |

## 部署形态与成本权衡

- **Windows 原生（当前主用）**：MySQL 80 + Ollama + `.venv` + vite preview，`start.ps1` / `stop.ps1` 一键启停，细节见 [DEPLOY.md](DEPLOY.md) 第〇节。
- **Docker Compose**：三容器，向量库挂 `chroma_data` 卷，容器重建不丢向量。
- **2 核 4G 云服务器（真实上线过，后因预算下线）**：本地 bge-m3 与 7B 对话模型同时驻留会打满内存，因此把 `EMBEDDING_PROVIDER` 切到 `siliconflow`（同权重 bge-m3，向量兼容），对话走 DeepSeek 云 API。代价是网络延迟和按量调用费，换来的是不需要 GPU、不需要大内存也能跑完整 RAG。这条链路在本项目里是一等公民，不是事后补的开关。

## 测试

```bash
cd llmops-backend
.venv\Scripts\python.exe -m pytest -q     # 17 passed
```

覆盖：Prompt CRUD、RAG 检索的宽候选 + 阈值过滤、chat 流程的 `rag_status` 四态判定。前端类型检查：

```bash
cd llmops-frontend
npm run type-check   # vue-tsc，零报错
```

## 已知边界与后续计划

- **未做 rerank 精排**：问法与文档措辞差异较大时，正确片段仍可能排在第 3–4 位；当前只做到「召回 + 阈值 + 弱命中提示」。
- **阈值与语料强相关**：`0.53` 和 `0.06` 是在本机工业语料上标定的，换一批语料需要重新扫描。
- **鉴权是演示口令**，不是生产级认证方案。
- 单机部署，无水平扩展；尚未接入 CI。

后续计划：

- [ ] rerank 精排（交叉编码器），把检索做成「召回 + 重排」两级
- [ ] 阈值 / 余量按语料自动标定的脚本，替代手工扫描
- [ ] GitHub Actions：pytest + vue-tsc
- [ ] 补充界面截图与在线 Demo

## 文档

- [DEPLOY.md](DEPLOY.md) —— 部署步骤、自检清单、故障排查
- [docs/02.代码编写顺序](docs/02.代码编写顺序) —— 各文件的编写顺序与职责
- [docs/03.核心开发流程](docs/03.核心开发流程) —— 主链路图

---

本项目用于学习与实践；仓库内不含任何真实密钥（`.env` 已在 `.gitignore` 中，仅保留 `.env.example` 占位模板）。
