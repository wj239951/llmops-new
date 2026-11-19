# 读取 .env 配置，统一管理数据库地址、 Ollama 地址、 DeepSeek Key 、向量化通道 。

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 一律按「本文件所在的后端根目录」解析，而不是按进程 cwd 解析。
# 否则在别的目录启动 uvicorn（systemd、计划任务、IDE 配置错一个工作目录）时会读不到 .env，
# 静默退化成空密码 / 默认地址，表现为 MySQL 1045 或 RAG 整条链路降级，排查成本极高。
BACKEND_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    APP_NAME: str = "Industrial LLMOps Ollama Local"
    APP_ENV: str = "local"
    API_PREFIX: str = "/api"
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = "llmops_db"
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    DEFAULT_MODEL_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "deepseek-r1:7b"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"  # 官方模型名只有 deepseek-chat / deepseek-reasoner
    DEMO_PASSWORD: str = "demo2026"  # 演示口令（前后端一致）
    MEMORY_MAX_TURNS: int = 10  # 内存最大轮数
    # RAG 分数阈值（cosine）。2026-11-17 在 196 篇 / 338 片段 / 183 题上重扫（docs/13）：
    # 0.53 覆盖率 100% 但无答案误接受 80%；0.56 覆盖仍 100%、误接受降到 71.4%，是"不丢正确片段"的上限；
    # 再往上（0.60 起）开始丢 2% 以上的正确片段。真正治拒答靠精排，不靠这个数。
    RAG_SCORE_THRESHOLD: float = 0.56
    # 候选池宽度：先取这么多条，再由阈值过滤、截断。必须大于 RAG_FEED_TOPK，
    # 否则排在 topk 之后的正确片段永远进不了候选集，阈值调多高都救不了（E10 实测）。
    RAG_CANDIDATE_K: int = 6
    # 最终喂给模型 / 写进 sources 的条数
    RAG_FEED_TOPK: int = 3
    # 上下文按文档去重：同一篇文档只保留最相关的一个片段（docs/13 实测出现过一篇占满 3 条）。
    # 关掉则退回"纯按分数截断"，多文档对比类问题可能只剩一份材料。
    RAG_DEDUP_BY_DOC: bool = True
    # top1 高于阈值但不足此余量时判为「弱命中」，前端降级显示，不当可靠依据。
    # 0.06 是 2026-10-20 在本机 21 向量库上实测出来的：闸门放行的 19 条查询里
    # top1 错的有 2 条（0.5733 螺杆链条 / 0.5858 氧气含量），top1 对的最低是 0.5831（住宿标准）。
    # 两组交错，故不存在零误报的余量；0.06（判定线 0.5900）是「拦住全部已知真错」的最小值，
    # 代价是 1 条误报。样本只有 2 条正例，这个数不能当常数——换语料后须重扫。
    RAG_WEAK_MARGIN: float = 0.06
    # ── 第二阶段精排（bge-reranker-v2-m3，可选）──
    # 默认关闭：需先 pip install -r requirements-rerank.txt 并准备权重。
    # 开启后若依赖/权重缺失，RAGService.search 会自动回退纯 cosine 检索并打 warning。
    RAG_RERANK_ENABLED: bool = False
    RAG_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # 本地权重目录（HF snapshot 下载目录）；留空则按 RAG_RERANK_MODEL 从 HF 缓存/网络加载
    RAG_RERANK_MODEL_PATH: str = ""
    # 进入精排的候选数（bge-m3 召回池）；最终仍只喂 RAG_FEED_TOPK 条。
    # 45 题上 K=6 已 100% 召回；K=12 质量相同但 CPU 精排耗时约 2.4 倍（实测见 docs/05）
    RAG_RERANK_CANDIDATE_K: int = 6
    # 精排分数 = sigmoid(logit)，与 cosine 不同尺度。
    # 0.76 为 45 题扫描推荐值：8 条无答案全部拒掉，代价 1 条改写题漏答；标定细节与局限见 docs/05
    # 精排阈值（sigmoid）同样在 docs/13 重扫：0.76 在 338 片段上留下 30% 近域误接受，
    # 0.90 覆盖 96.6% / 误接受 5.7%，0.95 覆盖 94.6% / 误接受 0%。默认取 0.90 做平衡点。
    RAG_RERANK_SCORE_THRESHOLD: float = 0.90
    RAG_RERANK_WEAK_MARGIN: float = 0.06
    RAG_RERANK_MAX_LENGTH: int = 512
    # 精排推理的 torch 线程数。start.ps1 给 numpy 设了 OMP_NUM_THREADS=1（避免 16 线程内存分配失败），
    # 但 torch 会继承这个环境变量，导致精排 6 个片段要 20 s；显式设 4 线程可降到 ~6 s，
    # 又不会回到 16 线程的内存风险。0 表示不干预（沿用环境变量）。
    RAG_RERANK_TORCH_THREADS: int = 4
    # 启动时后台预热精排模型（2 GB 权重，CPU 首次加载 15–40 s）
    RAG_RERANK_WARMUP: bool = True
    # auto | cpu | cuda
    RAG_RERANK_DEVICE: str = "auto"
    # 精排运行模式：local = 进程内 torch 加载权重；cloud = 调云端 /rerank API。
    # 2核2G 这类小内存服务器装不下 2.27GB 权重，用 cloud；cloud 不依赖 torch/transformers，
    # 请求失败/缺 key 时与 local 缺权重一样回退纯 cosine。
    # 注意：cloud 分数是服务商的 relevance_score，与 local 的 sigmoid(logit) 尺度不同，
    # 切换后 RAG_RERANK_SCORE_THRESHOLD 须按 docs/05 的方法重扫标定。
    RAG_RERANK_PROVIDER: str = "local"
    # cloud 模式 /rerank 端点前缀（含 /v1 或 /v2）；SiliconFlow / Jina / Cohere v2 请求响应同构
    RAG_RERANK_API_BASE_URL: str = "https://api.siliconflow.cn/v1"
    # cloud 模式 key；留空复用 SILICONFLOW_API_KEY（同一家硅基流动账号时不用填两遍）
    RAG_RERANK_API_KEY: str = ""
    RAG_RERANK_API_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RAG_RERANK_API_TIMEOUT: float = 10.0
    # 检索查询分解：复合问题拆主题子问句，各自过精排闸门后合并去重（query_decompose）。
    # 默认关闭；开启后每问多一次 LLM 调用（走 DEFAULT_MODEL_PROVIDER），失败回退原句。
    RAG_QUERY_DECOMPOSE_ENABLED: bool = False
    # 多轮检索查询改写（contextualization）：把「那它的更换周期呢」补成可独立检索的完整问句。
    # off（默认）= 关闭；rule = 规则改写（零延迟，从最近轮次抽型号回填指代，失败退原句）；
    # llm = 模型改写（更准，每轮多一次调用，失败按 RAG_QUERY_REWRITE_FALLBACK 退 rule）。
    # 只改「检索查询」，用户原文仍用于展示、写库与生成；单轮/自带型号/换话题的问题不改。
    RAG_QUERY_REWRITE_MODE: str = "off"
    # llm 模式的硬超时；超时不阻塞，直接按 fallback 回退（线程跑完结果丢弃）
    RAG_QUERY_REWRITE_TIMEOUT: float = 1.5
    RAG_QUERY_REWRITE_FALLBACK: str = "rule"
    # ── 拒答策略（docs/17）──
    # 有命中时的提示词收紧为「只依据片段回答，片段没有就说未收录」，不再允许用常识补充。
    RAG_STRICT_GROUNDING: bool = True
    # 无命中时不再交给模型自由发挥，直接返回固定话术（不调用模型，省掉一次生成）。
    # 关掉退回旧行为：把原问题直接给模型，由它结合通用知识回答。
    RAG_NO_HIT_REPLY_ENABLED: bool = True
    RAG_NO_HIT_REPLY_TEXT: str = (
        "知识库里没有检索到与这个问题相关的内容。"
        "可以补充设备型号、文件名称，或换个说法再问一次。"
    )
    # 数字级 grounding：回答里出现、但喂进去的片段里找不到的数字，会在回答末尾追加提示（docs/19）。
    # 只查"数字来源"，不查"语义归属"（把载重答成模具重量这类错配查不出来）。
    RAG_NUMERIC_GUARD_ENABLED: bool = True
    RAG_NUMERIC_GUARD_TEXT: str = "（提示：回答中的 {nums} 未在知识库片段中找到依据，请核实。）"
    # ── 混合检索（jieba/BM25 关键词召回 + 向量召回 RRF 融合，可选）──
    # 默认关闭：向量检索仍是主链路；开启后若 jieba/索引不可用，search 自动回退纯向量。
    # 关键词通道专治「精确编号 / 型号 / 数字」类查询（RL-2077、E-073、25m…），
    # 与精排互补：两者可同时开启，顺序为 混合候选 → 精排 → 阈值截断。
    RAG_HYBRID_ENABLED: bool = False
    # 稀疏/稠密两路各自召回的候选数（融合后池子不超过两者之和）
    RAG_HYBRID_DENSE_K: int = 12
    RAG_HYBRID_SPARSE_K: int = 12
    # RRF 平滑常数：score = Σ 1/(k + rank)，k 越大越偏向「两路都靠前」
    RAG_HYBRID_RRF_K: int = 60
    # 精确编号豁免：查询里带型号/编号（ZX-2026、GA-90、WNS2-1.25-Q）且候选片段原文包含该编号时，
    # 不再要求它为 cosine 阈值让路。这类片段的余弦分普遍只有 0.4 左右，会被 0.53 的闸门误杀，
    # 导致混合检索召回的片段到不了回答层（2026-11-05 实测：ZX-2026 候选分 0.4207，最终 0 条）。
    RAG_HYBRID_EXACT_BYPASS: bool = True
    # ── 输入清洗（可选）：把用户粘贴的脏文本规范化后再进检索 / 生成 ──
    # off = 关闭；rule = 纯规则清洗（毫秒级、零依赖）；model = 调独立清洗服务（蒸馏 0.8B 模型）。
    # 默认关闭：清洗会改写用户原文，属于需要显式开启的行为。
    # 开启后任何失败都不影响主链路，按 INPUT_CLEAN_FALLBACK 回退。
    INPUT_CLEAN_MODE: str = "off"
    # model 模式的清洗服务地址（tools/clean_server.py 默认监听 8010）
    INPUT_CLEAN_SERVER_URL: str = "http://127.0.0.1:8010"
    # 单次清洗超时：模型通道在 CPU 上约 0.3–1s，留 5s 兜冷启动
    INPUT_CLEAN_TIMEOUT: float = 5.0
    # 清洗失败兜底：rule = 退回规则清洗；off = 直接用原文
    INPUT_CLEAN_FALLBACK: str = "rule"
    # 超过这个长度不清洗：模型通道会拖慢首字，长文清洗收益递减
    INPUT_CLEAN_MAX_CHARS: int = 4000
    # 是否允许「请求级精细清洗」（ChatRequest.use_clean_model=true 时走模型通道）。
    # 默认 true：主链路仍按 INPUT_CLEAN_MODE 走规则清洗，只有前端显式点「精细清洗」才升级到模型。
    # 小机器上没有 GPU（模型通道 CPU 上 0.3–1s/次）或不想让人滥用时，置 false 即可整体关掉，
    # 前端按钮随之隐藏，请求里的 true 会被忽略并退回规则清洗。
    INPUT_CLEAN_MODEL_ALLOWED: bool = True
    # embedding provider 开关:"ollama" | "siliconflow"
    EMBEDDING_PROVIDER: str = "ollama"
    # ollama embedding(本地)
    OLLAMA_EMBEDDING_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "bge-m3"
    # siliconflow embedding(云上)
    SILICONFLOW_EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    SILICONFLOW_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    SILICONFLOW_API_KEY: str = ""
    # ---- 图片输入：用户发图 -> OCR -> 与提问合并后检索（docs/22）----
    IMAGE_INPUT_ENABLED: bool = True
    # 单张图片上限（解码后字节），超过直接忽略，避免大图打爆内存
    IMAGE_INPUT_MAX_BYTES: int = 8 * 1024 * 1024
    # 提取文本上限，超出截断（检索上下文不需要整页）
    IMAGE_INPUT_MAX_CHARS: int = 1500
    # OCR 置信度过低的框直接丢弃
    IMAGE_INPUT_MIN_SCORE: float = 0.5

    @property
    def SQLALCHEMY_DATABASE_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            "?charset=utf8mb4"
        )

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8"
    )


settings = Settings()