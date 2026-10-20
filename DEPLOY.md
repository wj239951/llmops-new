# LLMOps 平台迁移部署说明

> 本文件写给**目标电脑上的 AI / 部署人**。按顺序执行即可跑起来，每步都给了"应该看到什么"。
> 项目：工业 LLMOps 平台（FastAPI + Vue3 + MySQL + Ollama RAG）。
> 打包来源机：`C:\Users\lenovo\Desktop\py\first_project\backend`（后端）+ `C:\Users\lenovo\Desktop\LLOPMS\frontend`（前端），打包日期 2026-10-17。
> **本包已在源机上用 Docker 完整跑通一遍**（含数据库导入、向量重建、RAG 问答），实测记录见文末第八节。

---

## 〇、本机（Administrator / 2026-10-17）实际采用的部署方式：原生，不用 Docker

本机未装 Docker Desktop，且 WSL2 功能为 Disabled（启用需重启），所以第三节的 `docker compose` 路线没有执行。
改为直接用宿主机已有组件，功能与 Docker 版等价，差异如下：

| 第三节（Docker） | 本机实际 |
| --- | --- |
| `llmops-mysql` 容器 | 复用已运行的 MySQL80 服务（127.0.0.1:3306，root），`mysql-init/llmops_db.sql` 已手工导入，覆盖原有空库 |
| `llmops-backend` 容器 :8000 | `llmops-backend/.venv`（uv 装的 CPython 3.11.17）里跑 `uvicorn app.main:app --host 127.0.0.1 --port 8000` |
| `llmops-frontend` 容器（nginx :80→:3000） | `npm run build-only` 产出 `dist/`，再用 `npm run preview`（vite 静态服务 :3000）；`/api` 反代写在 `llmops-frontend/vite.config.ts` 的 `preview.proxy` 里 |
| 容器内 `host.docker.internal:11434` | 直连 `127.0.0.1:11434`（见 `llmops-backend/.env`） |
| `chroma_data` 卷 | `llmops-backend/chroma_db/` 目录（已重建，21 条向量） |
| `docker compose up -d` / `down` | `powershell -ExecutionPolicy Bypass -File .\start.ps1` / `.\stop.ps1` |

启动前需要 MySQL80 服务和 Ollama（`deepseek-r1:7b` + `bge-m3`）在跑。
`docker-compose.yml`、`mysql-init/Dockerfile` 在本机不参与运行，保留备查。

---

## 一、这个 U 盘包里有什么

```
LLMOps_move/
├── docker-compose.yml          # 三个服务：mysql + backend + frontend（已写好，直接用）
├── .env.example                # 环境变量模板 → 复制成 .env 再填
├── DEPLOY.md                   # 本文件
├── mysql-init/
│   ├── Dockerfile              # FROM mysql:8.0 + 把备份烘进 initdb 目录
│   └── llmops_db.sql           # 源机 MySQL 全量备份（4 张表 + 数据，mysqldump 导出）
├── docs/
│   ├── 02.代码编写顺序          # 原作者的开发顺序笔记（了解设计意图用）
│   └── 03.核心开发流程          # 主链路图：Vue3 → /api/chat → ChatService → 模型 → 日志
├── llmops-backend/             # FastAPI 后端源码 + Dockerfile
│   └── tools/reindex_knowledge.py   # 【迁移包新增】从 MySQL 重建向量库
└── llmops-frontend/            # Vue3 前端源码 + Dockerfile + nginx.conf
```

源码里**不包含**：`.git`、`.venv`、`node_modules`、`__pycache__`、`chroma_db`（向量库数据）、`.env`（真实密钥）。见第七节的取舍说明。

---

## 二、目标机前置要求

| 依赖 | 要求 | 检查命令 |
| --- | --- | --- |
| Docker Desktop | 已安装并启动（Windows 需 WSL2 后端） | `docker info` 不报错 |
| Ollama | 装在**宿主机**，提供对话与向量化模型 | `ollama list` |
| 网络 | 能拉 Docker 基础镜像、能 `pip install` / `npm ci` | — |
| 端口 | 宿主机 `3000`、`8000` 未被占用 | `netstat -ano \| findstr "LISTENING" \| findstr ":3000 :8000"` |

**模型必须提前拉好**（源机实测体积：`deepseek-r1:7b` 4.7GB + `bge-m3` 1.2GB，合计约 6GB）：

```bash
ollama pull deepseek-r1:7b   # 对话模型
ollama pull bge-m3           # RAG 向量化模型，缺了第五步直接失败
```

> 不想装 Ollama？可以只走云端：`.env` 里设 `EMBEDDING_PROVIDER=siliconflow` + 填 `SILICONFLOW_API_KEY`，对话通道填 `DEEPSEEK_API_KEY`。两者同为 bge-m3 权重，向量互相兼容。

---

## 三、部署步骤

### 1. 放置项目

把整个 `LLMOps_move` 拷到目标机任意路径（**建议放本机 SSD，别在 U 盘上直接构建**，反复读写很慢）。下文所有命令都在这个含 `docker-compose.yml` 的目录里执行。

```bash
cd /d E:\LLMOps_move
```

### 2. 生成 `.env`

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
```

然后**编辑 `.env`，至少改掉 `MYSQL_PASSWORD`**（不需要和源机一致，这是容器里的全新 MySQL 实例）。
只用大小写字母 + 数字：后端把密码拼进 `mysql+pymysql://` 连接串，出现 `@ : / #` 会解析错位。值不要加引号。

### 3. 构建并启动

```bash
docker compose up -d --build
```

首次约 5–15 分钟（装 Python 依赖 + `npm ci` 构建前端 + 拉基础镜像）。结束标志：

```bash
docker compose ps
```

三个服务，`llmops-mysql` 显示 `healthy`，另两个 `Up`。

### 4. 确认数据库真的导进去了（别只看 healthy）

备份 `mysql-init/llmops_db.sql` 已烘进镜像的 `/docker-entrypoint-initdb.d/`，**仅当 `mysql_data` 卷为空时（即第一次启动）自动执行**。
⚠️ MySQL 容器在"初始化失败"后重启，会因为数据目录已非空而**跳过导入并照常变 healthy**，所以必须显式核对：

```bash
docker compose exec -T mysql sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW DATABASES; SELECT COUNT(*) FROM llmops_db.llm_knowledge_documents;"'
# → 应列出 llmops_db，且文档数为 11
```

若这里报 `Unknown database 'llmops_db'`，说明卷是在备份就位之前初始化的：
`docker compose down -v`（⚠️ 清空数据卷，本包首次部署无数据可丢）后重新 `docker compose up -d --build`。

### 5. 重建向量库（**必做，否则知识库检索 0 命中**）

备份带回了知识库文档（MySQL），但向量库 `chroma_db` 没随包迁移，需重新向量化一次：

```bash
docker compose exec backend python tools/reindex_knowledge.py
# → 待重新向量化：11 篇文档 … 完成，共写入 21 条向量。
```

若报连不上 Ollama，先回第二节确认 `ollama list` 里有 `bge-m3`。

### 6. 打开页面

浏览器访问 **http://localhost:3000**，输入演示口令（`.env` 里 `DEMO_PASSWORD`，默认 `demo2026`）。
接口文档：http://localhost:8000/docs

---

## 四、部署自检清单

逐条跑，右边是**期望结果**（下列值取自源机实测）。任何一条不符先看第五节。

```bash
# 1. 后端活着
curl http://localhost:8000/
# → {"name":"Industrial LLMOps Ollama Local","mode":"local",...}

# 2. 鉴权生效：不带口令必须 401
curl -s -o nul -w "%{http_code}\n" http://localhost:8000/api/prompts
# → 401

# 3. 数据库连通（鉴权是请求头带演示口令，见 app/core/security.py）
curl -H "Authorization: Bearer demo2026" http://localhost:8000/api/prompts
# → JSON 数组，非 401、非 500

# 4. 向量库真的有条数
docker compose exec backend python -c "from app.services.rag_service import vectorstore as v; print(v._collection.count())"
# → 21

# 5. RAG 端到端（走前端 nginx 反代，顺便验证代理）
curl -X POST http://localhost:3000/api/chat -H "Content-Type: application/json" -H "Authorization: Bearer demo2026" -d "{\"query\":\"注塑机ZX-2026的巡检周期是多久？\",\"use_rag\":true,\"model_provider\":\"ollama\"}"
# → HTTP 200；sources 有命中（首位应为《01_注塑机ZX-2026巡检手册.txt》，score 约 0.70）；answer 引用了手册内容
# 注：本地 7b 模型首次加载权重，这一步 20–30 秒属正常

# 6. 调用日志落库（跑完第 5 条后总数会 +1）
curl -H "Authorization: Bearer demo2026" "http://localhost:8000/api/logs?page=1&page_size=1"
# → total 递增，最新一条 status=success、带 total_tokens 与 latency_ms
```

---

## 五、故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 启动即报 `缺少 MYSQL_PASSWORD` | 没复制 `.env.example` 为 `.env`，或该行为空 | 见第三步第 2 小节 |
| backend 日志 `Unknown database 'llmops_db'` | 卷先于备份初始化，或改过 compose 后带着旧卷启动 → initdb 被跳过 | `docker compose down -v` 再 `up -d --build`（会清空数据卷） |
| backend 反复重启，日志 `Can't connect to MySQL server on 'mysql'` | MySQL 未 healthy，或 `.env` 密码与已有卷的 root 密码不一致 | `docker compose logs mysql`；改密码后需 `down -v` 重建 |
| 页面能开但 `/api` 全 502 | backend 容器没起来 | `docker compose logs backend` |
| 聊天报 `connection refused ... host.docker.internal:11434` | Ollama 未启动，或只监听 127.0.0.1 被容器网络隔离 | 启动 Ollama；仍不行则设 `OLLAMA_HOST=0.0.0.0:11434` 后重启 Ollama |
| RAG 恒 0 命中、`sources` 为空 | 第五步没跑；或换了 embedding 模型导致维度不一致 | 跑 `reindex_knowledge.py`；embedding 模型名一旦定下别再改 |
| 命中分数低于阈值被丢弃 | 源机阈值 `RAG_SCORE_THRESHOLD=0.53` 是针对那批文档调出来的（实测命中 0.70/0.55/0.53） | 换自己的文档后如召回过严，在 `.env` 改成 `0.45` 再 `docker compose up -d backend` |
| 启动日志满屏 `LangChainDeprecationWarning: OllamaEmbeddings / Chroma was deprecated` | langchain-community 0.4 / langchain-core 1.x 的弃用提示 | **无害，不影响功能**，实测整条链路仍正常。想消掉就装 `langchain-ollama`、`langchain-chroma` 并改 import |
| 端口 3000/8000 被占 | 与宿主机已有服务冲突 | 改 `docker-compose.yml` 里 `ports` 的**冒号左边** |
| 上传 PDF 报 413 | nginx 体积限制 | 已放宽到 20m（`llmops-frontend/nginx.conf`），自行改小会复现 |

---

## 六、访问地址与端口

| 用途 | 地址 | 说明 |
| --- | --- | --- |
| 平台页面 | http://localhost:3000 | 前端 nginx，`/api` 由它反代到 backend |
| 后端 API | http://localhost:8000/api | 直连后端，仅调试用 |
| Swagger | http://localhost:8000/docs | |
| MySQL | 未对外映射 | 容器内网 `mysql:3306`，避免和目标机本机 MySQL 抢端口；要客户端连就加 `ports: - "13306:3306"` |

---

## 七、打包时做的取舍与改动（迁移后要知道的差异）

1. **依赖版本已锁死**。`llmops-backend/requirements.txt` 原来只写包名不带版本，已按源机 `.venv` 实测版本改成 `==` 精确锁定（18 个包全部核对，源码 import 与清单一致，无遗漏）。源机 venv 是 Python 3.14.6、镜像是 `python:3.11-slim`，代码只用到 3.10+ 特性，已在 3.11 镜像里构建并跑通。**建议把这份 requirements.txt 回填进 git 仓库**，否则下次换机器装出来的版本可能漂移。
2. **没带 `.env`**：源机真实密码与 API Key 不随 U 盘走，目标机自己填。`DEEPSEEK_API_KEY` / `SILICONFLOW_API_KEY` 需重新提供。
3. **没带 `chroma_db`**（源机仅 9.2MB）：文档在、向量不在，所以必须执行第三步第 5 小节。想让向量也随包走，就把源机 `backend/chroma_db/` 整个目录拷进包里命名为 `chroma_db/`，再把 compose 里 backend 的 `- chroma_data:/app/chroma_db` 改成 `- ./chroma_db:/app/chroma_db`——前提是该目录在目标机上以本机磁盘路径挂载（U 盘/exFAT 上的绑定挂载不可靠，见第 4 条）。
4. **备份改成"烘进镜像"而非 bind mount**——这是实测逼出来的改动：Docker Desktop 把 U 盘（exFAT）上的**单文件绑定挂载静默挂成了空目录**，MySQL 读到目录后跳过导入，容器照样 healthy，后端却因缺库启动失败。build 上下文由 docker client 读取后传给守护进程，不受文件共享/exFAT 影响，所以放进 `mysql-init/` 用 `COPY`。
5. **`C:\Users\lenovo\Desktop\数据库\llmops.sql` 已过时，不要拿它建库**：它缺 `conversations` 表和 `llm_chat_logs.conversation_id` 字段（会话记忆功能依赖这两者）。本包的 `mysql-init/llmops_db.sql` 是从源机**运行中的库**直接 dump 的，4 张表全量数据，以它为准。
6. **`DEEPSEEK_MODEL` 默认值改为 `deepseek-chat`**：源机 `.env.example` 写的 `deepseek-v4-pro` 不是官方模型名，走云端通道会报模型不存在；可选只有 `deepseek-chat` / `deepseek-reasoner`。
7. **`.env` 变量必须在这里声明才会进容器**：compose 只转发 `docker-compose.yml` 里引用到的键。`RAG_SCORE_THRESHOLD`、`MEMORY_MAX_TURNS` 已补进 backend 的 `environment`；以后新加配置项，两处都要写。
8. **鉴权仍是演示级**：`Authorization: Bearer <DEMO_PASSWORD>` 明文口令，README 里也已标注"仅适用本地开发"。别把它直接挂公网。

---

## 八、源机验证记录（2026-10-17，Docker Desktop 29.8.2 / Compose v5.5.1）

| 检查项 | 结果 |
| --- | --- |
| `docker compose build` 前后端镜像 | exit 0，backend 1.16GB / frontend 104MB |
| `docker compose up -d` | 三服务起，`llmops-mysql` healthy |
| initdb 自动导入 | `llmops_db` + 4 张表；prompts 2 / knowledge 11 / conversations 4 / chat_logs 151 |
| 后端启动 | `Application startup complete`，Uvicorn :8000；自动创建 cosine collection |
| 鉴权 | 无口令 401，带口令 200 |
| `reindex_knowledge.py` | 11 篇文档 → 21 条向量（经 `host.docker.internal` 调到宿主机 bge-m3） |
| RAG 问答（走 :3000 反代） | HTTP 200，26.5s；命中 3 条，首位《01_注塑机ZX-2026巡检手册.txt》score 0.702；回答按手册内容给出滤芯 500 小时、螺杆 2000 小时等条目 |
| 日志落库 | total 151 → 153，最新一条 `status=success`、total_tokens 1211、latency_ms 26471 |

验证完成后源机已 `docker compose down`，U 盘包内不含 `.env`，也不含任何运行数据。

---

## 九、日常运维命令

```bash
docker compose ps                      # 看状态
docker compose logs -f backend         # 跟后端日志
docker compose restart backend         # 重启单个服务
docker compose up -d --build backend   # 改了后端代码后重新构建
docker compose down                    # 停止并删容器（数据卷保留，下次 up 数据还在）
docker compose down -v                 # ⚠️ 连数据卷一起删：MySQL 数据 + 向量库全部清空
```

备份目标机当前数据：

```bash
docker compose exec -T mysql sh -c 'mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" --databases llmops_db' > backup.sql
```

改了 `.env` 里的值（如 `RAG_SCORE_THRESHOLD`）要重建容器才生效：`docker compose up -d backend`。
