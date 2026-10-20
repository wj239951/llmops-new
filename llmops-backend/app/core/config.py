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
    DEEPSEEK_MODEL: str = "deepseek-v4-pro"
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