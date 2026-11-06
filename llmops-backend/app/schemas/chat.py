# 聊天请求、返回数据结构。
from pydantic import BaseModel, field_validator

class ChatRequest(BaseModel):
    query: str
    prompt_id: int | None = None
    system_prompt: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    use_rag: bool = False
    # None=跟随服务端配置；True/False=本次请求强制开启/关闭精排（前端按钮开关用）
    use_rerank: bool | None = None
    ##
    conversation_id: str | None = None# 聊天会话 ID
    memory_enabled: bool = False# 是否启用内存
    ##

    @field_validator("conversation_id", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: str | None) -> str | None:
        return v or None

class ChatResponse(BaseModel):
    answer: str
    reasoning: str | None = None
    thought: str
    model_provider: str
    model_name: str
    total_tokens: int
    latency_ms: int
    ##
    conversation_id: str | None = None
    rag_status: str = "off"  # off / no_hit / hit —— 不要让前端靠 sources 是否为空反推
    sources: list[dict] = []  # 每个元素带 score
    # 本次实际生效的打分方式：off / cosine / rerank；rerank_requested 是用户意图，
    # rerank_available 表示服务端精排是否真的可用（不可用时已回退余弦）
    score_mode: str | None = None
    rerank_requested: bool | None = None
    rerank_available: bool | None = None
    # 阶段耗时（ms）：prepare / retrieve / rerank / llm / first_token / total
    timings: dict | None = None
    ##
