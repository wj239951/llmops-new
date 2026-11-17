# 聊天接口：POST /api/chat（非流式）与 POST /api/chat/stream（SSE 流式）。
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])
service = ChatService()

# SSE 需要关掉各级代理的缓冲，否则浏览器要等整段生成完才收到数据。
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(item: dict) -> str:
    """把事件字典序列化为 SSE 帧。"""
    data = json.dumps(item.get("data") or {}, ensure_ascii=False)
    return f"event: {item.get('event', 'message')}\ndata: {data}\n\n"


@router.get("/options")
def chat_options():
    """前端精排 / 输入清洗开关的初始状态：只读服务端配置，不触发模型加载。"""
    return {
        "rerank_enabled": bool(settings.RAG_RERANK_ENABLED),
        "rerank_model": settings.RAG_RERANK_MODEL,
        "hybrid_enabled": bool(settings.RAG_HYBRID_ENABLED),
        "score_threshold_cosine": settings.RAG_SCORE_THRESHOLD,
        "score_threshold_rerank": settings.RAG_RERANK_SCORE_THRESHOLD,
        # 输入清洗：mode=off 时前端开关默认收起；fallback=rule 表示模型通道失败会退回规则清洗
        "clean_mode": settings.INPUT_CLEAN_MODE,
        "clean_enabled": settings.INPUT_CLEAN_MODE in ("rule", "model"),
        "clean_fallback": settings.INPUT_CLEAN_FALLBACK,
        # 请求级「精细清洗」是否被允许：false 时前端不显示该按钮，请求里的 true 会被忽略
        "clean_model_allowed": bool(settings.INPUT_CLEAN_MODEL_ALLOWED),
    }


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    return service.chat(db, req)


@router.post("/stream")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """流式对话（SSE）：status → meta → reasoning/delta* → done，异常时发 error。"""

    def event_source():
        try:
            for item in service.stream(db, req):
                yield _sse(item)
        except Exception as exc:  # 生成器内部已兜底，这里再防一层，避免连接裸断
            yield _sse({"event": "error", "data": {"message": str(exc)}})

    return StreamingResponse(event_source(), media_type="text/event-stream", headers=SSE_HEADERS)
