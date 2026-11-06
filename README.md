# LLMOps 平台 · 工业知识库 RAG 问答

> 一个可本地部署的一站式 LLMOps 平台：**Prompt 管理 + 知识库 RAG + 多模型对话 + 会话记忆 + 调用日志**。
> 后端 FastAPI，前端 Vue3 + TypeScript，向量检索 Chroma（bge-m3），元数据 MySQL 8。

`后端 pytest 52 passed` · `前端 vue-tsc / vite build 通过` · `对话 SSE 流式输出` · `精排请求级开关` · `三模式检索评测一条命令复现` · `Docker Compose 一键启动` · `支持 Windows 原生部署`

---

## 这是什么

这不是一个「调通大模型 API 就算完成」的 Demo。它把一条完整的 RAG 链路拆开、测量、再一步步修：

- **知识怎么进来**：手动录入 / 文件上传（txt、md、pdf、py、json）→ 解析 → 按中文标点切块 → bge-m3 向量化 → 写入 Chroma；
- **怎么被检索到**：余弦相似度取宽候选 →（可选）jieba/BM25 关键词召回 + RRF 融合 →（可选）bge-reranker 交叉编码器精排 → 阈值过滤 → 截断 → 拼上下文 → 交给模型；
- **怎么判断「命中了但不可靠」**：top1 只是勉强过线时降级为 `weak_hit`，前端明确告警，而不是假装命中；
- **怎么保证两套存储不打架**：MySQL 与 Chroma 的写/删同步，孤儿向量会被记录而不是静默吞掉。

阈值、余量、候选池不是拍出来的数字：仓库内置 75 条评测集（45 条自然语言 + 15 条精确编号 + 15 条搜索框短查询）、可复现的检索评测脚本与三级检索实验（见 [docs/04.评测报告](docs/04.评测报告.md)、[docs/05.Rerank实验报告](docs/05.Rerank实验报告.md)、[docs/06.混合检索实验报告](docs/06.混合检索实验报告.md)），基线数字、阈值扫描与局限都写在报告里。

## 量化结果（一条命令复现）

| 检索模式 | Recall@6 | 命中率 | 无答案误放行 | P95 延迟 |
| --- | --- | --- | --- | --- |
| 纯向量 cosine | 100% | 100% | **100%** | 45 ms |
| + BM25 混合检索（RRF） | 100% | 100% | **100%** | 75 ms |
| + bge-reranker 精排 | 100% | 97.3% | **0%** | 8.7 s |

- 45 题（37 可答 + 8 无答案）；答案相关性用 LLM-as-judge 抽样 10 题，均分 **4.6 / 5**，4 分以上占 90%。
- 结论：**召回不是瓶颈（三种模式 Recall@6 都是 100%），闸门才是**——纯余弦把 8 条无答案问题全部放行；精排用 2.7 个点的命中率换掉 100% 的误放行，代价是 P95 从 45 ms 涨到 8.7 s，因此做成请求级开关（默认关）。
- 复现：`cd llmops-backend && python -m eval run --modes dense,hybrid,rerank`；指标退化则退出码非 0：`--gate eval_results/baseline/summary.json`。

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

对话输出有两条路径：`POST /api/chat` 一次性返回完整回答，`POST /api/chat/stream` 用 SSE 增量返回状态、`reasoning`、回答分片与最终汇总；两者共用 `ChatService` 的 Prompt 解析、会话记忆、RAG 检索与调用日志，前端默认走流式，首包前失败自动回退非流式。Chat Studio 还提供「精排」开关：请求体 `use_rerank`（`null` 跟随服务端、`true`/`false` 本次强制），响应与 SSE `meta` 回传 `score_mode` / `rerank_requested` / `rerank_available`，页面直接显示本次打分方式，服务端精排不可用时提示已回退向量余弦。

## 核心工程问题与解法

这是本项目真正花时间的地方，也是它区别于普通 CRUD 的地方：

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| 中文整句提问检索 **0 命中** | 关键词方案按空格切词，`润滑脂型号是什么` 被切成 1 个 token，`LIKE` 要求整句连续出现 | 改用 **bge-m3 向量检索**，整句编码，不再依赖分词 |
| 写入 384 维、读取 1024 维 | 直接调 `_collection.add` 会走 chromadb 自带的默认 embedding | 统一走 LangChain `add_texts`，锁定 bge-m3 |
| 所有相似度整体偏高、区分度差 | 向量化时拼了标题前缀，同文档各 chunk 共享同一段开头 | **只对正文向量化**，标题放 metadata |
| 正确片段永远进不了候选集 | 先 `k=3` 截断，再按阈值过滤，第 4 名及以后被提前丢掉 | 先取 `RAG_CANDIDATE_K=6` 宽候选 → 阈值过滤 → 截 `RAG_FEED_TOPK=3` |
| 阈值没有依据 | 拍脑袋设一个数，换个问法就崩 | 45 条评测集 + 阈值 / 余量扫描（[docs/04](docs/04.评测报告.md)）；当前语料下 `0.53` 保召回，top1 余量 < `0.06` 判 **`weak_hit`**，前端告警 |
| 只输型号/编号（`E-073`、`25m 水带`）时正确片段召回不到 | embedding 对稀有 token 不敏感，精确字符在向量里权重极低 | 加 **可选 jieba/BM25 关键词召回 + RRF 融合**：搜索框短查询 recall@6 **86.7% → 100%**，开精排后 hit@3 **26.7% → 46.7%**（[docs/06](docs/06.混合检索实验报告.md)） |
| 无答案问题被强行放行 | 双塔向量只衡量主题相似：8 条无答案题 top1 分数（0.5385–0.6790）与正确题（0.5808 起）重叠 | 加 **bge-reranker-v2-m3 交叉编码器精排**：45 题上无答案误接受 100% → **0%**，代价是 1 条改写题漏答（[docs/05](docs/05.Rerank实验报告.md)） |
| MySQL 与向量库不一致 | 两套存储各自写，删了文档向量还在 | 更新/删除先落 MySQL 再同步删向量；反查不到文档的孤儿向量记 `warning` |
| 删除不存在的文档却返回 200 | 只做 `query().delete()`，不校验受影响行 | 删除前查存在，不存在返回 **404**，与 `PUT` 语义对齐 |

## 功能

- **对话**：本地 Ollama（deepseek-r1:7b）与 DeepSeek 云端双通道，可选 Prompt 模板、温度 / top_p、思考过程、Token 与耗时统计；**SSE 流式输出**（`reasoning` 分片 + RAG 阶段进度），流式失败自动降级非流式；**精排开关**可请求级强制开/关，并显示本次实际打分方式
- **知识库**：手动录入、文件上传解析、增删改查，切块与向量同步，来源与命中分数回显
- **RAG**：三级检索（余弦召回 + 可选 jieba/BM25 关键词召回 RRF 融合 + 可选 bge-reranker 精排）、阈值过滤、宽候选池、弱命中降级（`hit` / `weak_hit` / `no_hit` / `off` 四态）；支持请求级 `use_rerank` 覆盖服务端配置
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

前置：MySQL 80 服务、Ollama（同上两个模型）、Python 3.11+ 虚拟环境、Node.js 18+。

```powershell
# 1) 导入库结构与数据（覆盖空库）
mysql -uroot -p < mysql-init/llmops_db.sql

# 2) 生成后端配置（注意：读的是 llmops-backend/.env；根目录 .env 只给 docker compose 用）
cd llmops-backend
copy ..\.env.example .env
# 编辑 .env：填 MYSQL_PASSWORD，并把 OLLAMA_BASE_URL 改成 http://127.0.0.1:11434
cd ..

# 3) 装依赖
cd llmops-backend
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
cd ..\llmops-frontend; npm ci; npm run build-only
cd ..

# 4) 一键启停
.\start.ps1     # backend :8000 + frontend :3000
.\stop.ps1
```

### 可选：开启 RAG 精排（CPU 可跑）

默认关闭，不装依赖也不影响运行；开启后若依赖/权重缺失会自动回退纯 cosine 检索，服务不会崩。

```powershell
cd llmops-backend
.venv\Scripts\pip install -r requirements-rerank.txt        # torch + transformers，约 150MB

# 权重下载到本地目录（约 2.17GB；国内用 hf-mirror）
$env:HF_ENDPOINT = "https://hf-mirror.com"
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', local_dir=r'D:\models\bge-reranker-v2-m3', allow_patterns=['*.json','*.safetensors','*.model','*.txt'])"
```

然后在 `llmops-backend/.env` 里设置：

```dotenv
RAG_RERANK_ENABLED=true
RAG_RERANK_MODEL_PATH=D:\models\bge-reranker-v2-m3
RAG_RERANK_CANDIDATE_K=6
RAG_RERANK_SCORE_THRESHOLD=0.76
```

重启后端即可。CPU 上每次检索约 +4.3–6.2s（K=6；K=12 约 10.4s），首次请求需 13–16s 加载模型。标定过程、误判与代价见 [docs/05.Rerank实验报告](docs/05.Rerank实验报告.md)。

### 可选：开启混合检索（无需额外依赖）

默认关闭。开启后向量召回之外并行一路 jieba/BM25 关键词召回，用 RRF 融合候选；不装新依赖、不加新存储，索引在首次检索时从 Chroma 懒加载。

```dotenv
# llmops-backend/.env
RAG_HYBRID_ENABLED=true
```

重启后端生效。代价是纯召回模式每次检索 +8~12ms；BM25 索引不可用时自动退回纯向量检索。短查询/编号类问题的收益与局限见 [docs/06.混合检索实验报告](docs/06.混合检索实验报告.md)。

## 配置

`.env.example` 是完整模板；`RAG_*`、`RAG_HYBRID_*` 与 `RAG_RERANK_*` 在代码里都有默认值，不写也能跑（混合检索与精排默认关闭）。

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
| `RAG_RERANK_ENABLED` / `RAG_RERANK_MODEL_PATH` | 精排开关（默认 `false`）/ 本地权重目录；开启后依赖或权重缺失时自动回退纯 cosine；请求级 `use_rerank` 可临时覆盖 |
| `RAG_RERANK_CANDIDATE_K` / `RAG_RERANK_SCORE_THRESHOLD` / `RAG_RERANK_WEAK_MARGIN` | 代码默认 `6` / `0.76` / `0.06`（45 题标定，见 [docs/05](docs/05.Rerank实验报告.md)） |
| `RAG_HYBRID_ENABLED` / `RAG_HYBRID_DENSE_K` / `RAG_HYBRID_SPARSE_K` / `RAG_HYBRID_RRF_K` | 混合检索开关（默认 `false`）/ 向量候选 `12` / BM25 候选 `12` / RRF 常数 `60`；无额外依赖 |

## 目录结构

```
.
├── docker-compose.yml          # mysql + backend + frontend 三服务
├── .env.example                # 环境变量模板
├── start.ps1 / stop.ps1        # Windows 原生部署启停
├── DEPLOY.md                   # 完整部署 / 自检 / 排障手册
├── docs/                       # 开发顺序、主链路笔记、cosine/rerank/混合检索评测、流式改造报告
├── experiments/                # 75 条评测集 + cosine/rerank/混合检索结果
├── mysql-init/                 # MySQL 镜像 + 全量初始化 SQL
├── llmops-backend/
│   ├── app/
│   │   ├── api/                # chat（含 /chat/stream SSE）/ prompt / knowledge / logs / conversations
│   │   ├── core/               # config / database / security（Bearer 口令鉴权）
│   │   ├── models/             # prompt / knowledge / chat_log / conversation
│   │   ├── schemas/
│   │   ├── services/           # rag / hybrid / rerank / chat / llm / ollama / deepseek / prompt / file_parser
│   │   └── main.py
│   ├── tests/                  # pytest：Prompt CRUD、RAG 过滤、chat 流程、流式 SSE 事件顺序、rerank 精排/回退、混合检索分词/BM25/RRF
│   ├── tools/reindex_knowledge.py   # 从 MySQL 重建向量库
│   ├── tools/evaluate_rag.py        # 检索评测 / 阈值与余量标定
│   ├── tools/evaluate_rerank.py     # 精排评测 / 精排阈值与余量标定
│   ├── tools/evaluate_hybrid.py     # 混合检索四模式对照评测
│   ├── requirements-rerank.txt      # 可选精排依赖（torch + transformers）
│   └── Dockerfile
└── llmops-frontend/
    ├── src/views/              # ChatStudio / PromptManager / KnowledgeBase / runlogs
    ├── src/api/                # axios 封装 + 资源接口（含 SSE 流式客户端）
    ├── src/router/
    ├── vite.config.ts          # 含 preview.proxy 反代
    └── Dockerfile + nginx.conf
```

## API

除 `GET /` 外，`/api/*` 均需请求头 `Authorization: Bearer <DEMO_PASSWORD>`。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 服务信息（免鉴权） |
| `GET` | `/api/chat/options` | 页面可选项：精排是否可用、模型名、两套阈值（只读配置，不加载模型） |
| `POST` | `/api/chat` | 对话：RAG / 会话记忆 / 模型选择（非流式）；`use_rerank` 可覆盖精排开关 |
| `POST` | `/api/chat/stream` | 流式对话（SSE）：`status` / `meta` / `reasoning` / `delta` / `done` / `error`；`use_rerank` 覆盖精排开关 |
| `GET` `POST` | `/api/prompts` | Prompt 列表 / 新建 |
| `PUT` `DELETE` | `/api/prompts/{id}` | 更新 / 删除（不存在 404） |
| `GET` `POST` | `/api/knowledge` | 文档列表 / 新建（切块 + 向量化） |
| `POST` | `/api/knowledge/upload` | 上传文件解析入库 |
| `PUT` `DELETE` | `/api/knowledge/{id}` | 更新（重建向量）/ 删除（同步删向量，不存在 404） |
| `GET` | `/api/logs` | 调用日志，支持 `page` / `page_size` / `status` |
| `GET` `POST` | `/api/conversations` | 会话列表 / 新建 |
| `GET` | `/api/conversations/{id}/messages` | 会话历史消息 |
| `DELETE` | `/api/conversations/{id}` | 删除会话及其记录 |

流式接口返回 `text/event-stream`：`status`（`preparing` / `retrieving` / `generating`）→ `meta`（模型、`rag_status`、`sources`）→ `reasoning` / `delta` 增量 → `done`（最终 `answer`、`first_token_ms` / `latency_ms` / token）；异常时以 `error` 结束。旧 `/api/chat` 保留，作为非流式回退。精排开关：请求可带 `use_rerank`（`null`/省略 = 跟随服务端 `RAG_RERANK_ENABLED`，`true`/`false` = 本次强制），`meta` / `done` / 非流式响应中的 `score_mode` 为 `off` / `cosine` / `rerank`，`rerank_available=false` 表示已回退向量。

## 部署形态与成本权衡

- **Windows 原生（当前主用）**：MySQL 80 + Ollama + `.venv` + vite preview，`start.ps1` / `stop.ps1` 一键启停，细节见 [DEPLOY.md](DEPLOY.md) 第〇节。
- **Docker Compose**：三容器，向量库挂 `chroma_data` 卷，容器重建不丢向量。
- **2 核 4G 云服务器（真实上线过，后因预算下线）**：本地 bge-m3 与 7B 对话模型同时驻留会打满内存，因此把 `EMBEDDING_PROVIDER` 切到 `siliconflow`（同权重 bge-m3，向量兼容），对话走 DeepSeek 云 API。代价是网络延迟和按量调用费，换来的是不需要 GPU、不需要大内存也能跑完整 RAG。这条链路在本项目里是一等公民，不是事后补的开关。

## 测试

```bash
cd llmops-backend
.venv\Scripts\python.exe -m pytest -q     # 36 passed
```

覆盖：Prompt CRUD、RAG 检索的宽候选 + 阈值过滤、chat 流程的 `rag_status` 四态判定、流式 SSE 的事件顺序 / `meta` + `sources` / 成功与失败日志、精排重排 / 阈值过滤 / 模型不可用回退、混合检索的分词 / BM25 / RRF / 索引不可用回退、精排请求级开关（三态 / 不可用回退 / 响应报告）。前端类型检查与构建：

```bash
cd llmops-frontend
npm run type-check   # vue-tsc，零报错
npm run build-only   # vite 构建
```

## 评测

检索层评测已固化为「题库 + 脚本 + 结果」，cosine 与 rerank 两级都有：

```bash
cd llmops-backend
.venv\Scripts\python.exe tools/evaluate_rag.py        # cosine 基线
.venv\Scripts\python.exe tools/evaluate_rerank.py    # rerank 精排（需先按上文准备权重）
.venv\Scripts\python.exe tools/evaluate_hybrid.py    # 混合检索四模式 × 三题库（需精排权重）
```

- 题库：`experiments/eval_qa.jsonl` 45 条自然语言（21 exact / 16 paraphrase / 4 远域无答案 / 4 近域无答案）+ `eval_qa_hard.jsonl` 15 条精确编号长问句 + `eval_qa_token.jsonl` 15 条搜索框短查询
- cosine 基线（21 条向量，`threshold=0.53`）：recall@6 **100%**、MRR **1.0000**、hit@3 **100%**、top1 准确率 **100%**；8 条无答案问题全部被阈值放行
- rerank（`bge-reranker-v2-m3`，K=6，`threshold=0.76`）：recall@6 **100%**、MRR **1.0000**、hit@3 **97.3%**（36/37）、**无答案误接受 0%**（8/8 全拒）；代价是 1 条改写题漏答、CPU 单次检索 +4.8s
- hybrid 对照（[docs/06](docs/06.混合检索实验报告.md)）：自然语言题库上 dense 召回已 100%，hybrid 与 dense 指标完全一致；**搜索框短查询**题库上 recall@6 **86.7% → 100%**，开精排后 hit@3 **26.7% → 46.7%**、MRR **0.8222 → 1.0000**，耗时与 dense+rerank 基本持平
- 产物 `experiments/results/`：`per_query.csv`、`threshold_sweep.csv`、`margin_sweep.csv`、`summary.json`（cosine），`rerank_*.csv`、`rerank_summary.json`（rerank），以及 `hybrid_compare.csv`、`hybrid_per_query.csv`、`hybrid_summary.json`（四模式对照）
- 结论与局限见 [docs/04.评测报告](docs/04.评测报告.md)、[docs/05.Rerank实验报告](docs/05.Rerank实验报告.md)、[docs/06.混合检索实验报告](docs/06.混合检索实验报告.md)

## 已知边界与后续计划

- **rerank 只在小样本上标定**：`0.76` 是 45 题上的刀刃值——距最高误接受样本（q43=0.7509）仅 **0.009**，稳定平台只有 0.76–0.78 三步；同时漏掉 1 条改写题（q02，精排分 0.2368）。换语料或扩题库必须重扫，不能当常数。
- **精排是 CPU 推理**：K=6 时单次检索 +4.3–6.2s，首次请求还要 13–16s 加载模型；未启用 CUDA / ONNX 加速。
- **精排开关在服务器上默认不可用**：2GB 云服务器未装 torch 与权重，页面打开开关会请求精排但后端回退 cosine（`meta.rerank_available=false`，前端提示已回退）；本地要看到 `score_mode=rerank` 需按上文安装依赖与权重。
- **cosine 单阈值无法可靠拒答**：45 条上正确题 top1 全部命中，但无答案题与正确题 top1 分数重叠（0.5808–0.6790）；`0.53` 是保召回选择，换语料需重扫。rerank 在本题库可 100% 拒掉无答案，但阈值同样是刀刃值（见上）。
- **评测集规模有限**：75 条合成题、11 篇语料、8 条无答案问题；端到端「模型是否拒答」尚未评测（RAG 提示词允许模型在知识不足时结合通用知识补充）。
- **混合检索收益有前提**：只在 dense 漏召回时生效。当前 11 篇语料上自然语言题库 dense 召回已 100%，hybrid 与 dense 指标完全相同；15 条搜索框短查询上才体现收益（recall@6 86.7%→100%），代价是纯召回模式 +8~12ms。BM25 索引建在进程内存，语料上万级需换持久化倒排。
- **精排阈值对编号类问题偏严**：阈值 `0.76` 标定自自然语言题库，hard 题库（精确编号）精排覆盖率只有 **73.3%**、短查询题库 **46.7%**——不少题排名第 1 但分数低于阈值被过滤，换题型需重扫（[docs/06](docs/06.混合检索实验报告.md)）。
- **流式依赖代理不缓冲**：SSE 已带 `X-Accel-Buffering: no`，nginx 默认 `proxy_buffering on` 会把增量输出压成一次性返回；服务器经 Caddy 反代，自建其它网关需关闭响应缓冲。当前也没有 SSE 心跳，开 RAG 精排时 `retrieving` 到 `meta` 之间可能数秒无输出，代理空闲超时过短会断流。
- **流式不做断线续传**：没有 `Last-Event-ID`，刷新或断网后只能重发请求（会重新生成、重复计费）；客户端主动取消不写调用日志，生成器被取消时已生成部分不落库。
- **流式时延尚未上报**：`first_token_ms` / `latency_ms` 只写进 `done` 与服务端日志，没有指标面板；本地一次联调的数值与冷启动说明见 [docs/07](docs/07.流式改造报告.md) 7.4。`deepseek-reasoner` 的思维链与回答分开推送，老版本 Ollama 内联的 `<think>…</think>` 只能在流结束后剥离。
- **鉴权是演示口令**，不是生产级认证方案。
- 单机部署，无水平扩展；尚未接入 CI。

后续计划：

- [x] rerank 精排（交叉编码器），把检索做成「召回 + 重排」两级 —— 见 [docs/05](docs/05.Rerank实验报告.md)
- [x] 阈值 / 余量按语料自动标定的脚本，替代手工扫描
- [x] 混合检索（jieba/BM25 + 向量 RRF 融合），搜索框短查询 recall@6 86.7%→100% —— 见 [docs/06](docs/06.混合检索实验报告.md)
- [x] SSE 流式输出（reasoning 分片 + RAG 阶段进度），首包前失败自动降级非流式 —— 见 [docs/07](docs/07.流式改造报告.md)
- [x] 精排请求级开关 + 打分方式可视化（Chat Studio「精排」开关，`GET /api/chat/options`）
- [ ] 扩充评测集（150+ 条、跨语料）验证 rerank 阈值泛化性
- [ ] 精排加速：CUDA / ONNX Runtime / 更小模型（CPU 上目前约 4.8s/次）
- [ ] 按题型分别标定精排阈值（自然语言 / 编号 / 短查询），或用 top1-top2 相对间隔替代绝对阈值
- [ ] SSE 心跳与断线重连（Last-Event-ID 或明确的「连接已断开」提示）
- [ ] TTFT / 分片间隔的指标上报
- [ ] GitHub Actions：pytest + vue-tsc
- [ ] 补充界面截图与在线 Demo

## 文档

- [DEPLOY.md](DEPLOY.md) —— 部署步骤、自检清单、故障排查
- [docs/02.代码编写顺序](docs/02.代码编写顺序) —— 各文件的编写顺序与职责
- [docs/03.核心开发流程](docs/03.核心开发流程) —— 主链路图
- [docs/04.评测报告](docs/04.评测报告.md) —— 检索层评测、阈值 / 余量扫描与结论
- [docs/05.Rerank实验报告](docs/05.Rerank实验报告.md) —— 两级检索、精排阈值扫描与代价
- [docs/06.混合检索实验报告](docs/06.混合检索实验报告.md) —— BM25 + 向量 RRF 混合检索、三题库四模式对照与局限
- [docs/07.流式改造报告](docs/07.流式改造报告.md) —— SSE 事件协议、双 provider 流式、前端增量渲染与降级

---

本项目用于学习与实践；仓库内不含任何真实密钥（`.env` 已在 `.gitignore` 中，仅保留 `.env.example` 占位模板）。
