# 读取 .env 配置，统一管理数据库地址、 Ollama 地址、 DeepSeek Key 、向量化通道 。

from pydantic_settings import BaseSettings, SettingsConfigDict

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
    RAG_SCORE_THRESHOLD: float = 0.53  # RAG分数阈值，由10片测试文档得来
    # 候选池宽度：先取这么多条，再由阈值过滤、截断。必须大于 RAG_FEED_TOPK，
    # 否则排在 topk 之后的正确片段永远进不了候选集，阈值调多高都救不了（E10 实测）。
    RAG_CANDIDATE_K: int = 6
    # 最终喂给模型 / 写进 sources 的条数
    RAG_FEED_TOPK: int = 3
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
    RAG_RERANK_SCORE_THRESHOLD: float = 0.76
    RAG_RERANK_WEAK_MARGIN: float = 0.06
    RAG_RERANK_MAX_LENGTH: int = 512
    # auto | cpu | cuda
    RAG_RERANK_DEVICE: str = "auto"
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
    # embedding provider 开关:"ollama" | "siliconflow"
    EMBEDDING_PROVIDER: str = "ollama"
    # ollama embedding(本地)
    OLLAMA_EMBEDDING_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "bge-m3"
    # siliconflow embedding(云上)
    SILICONFLOW_EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    SILICONFLOW_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    SILICONFLOW_API_KEY: str = ""
    @property
    def SQLALCHEMY_DATABASE_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            "?charset=utf8mb4"
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()