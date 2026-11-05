# FastAPI 主入口，注册路由、配置 CORS、启动时初始化数据库。
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings, BACKEND_DIR
from app.core.database import init_db
from app.core.security import require_auth
from app.api.chat import router as chat_router
from app.api.prompt import router as prompt_router
from app.api.knowledge import router as knowledge_router
from app.api.logs import router as logs_router
from app.api.conversations import router as conversations_router

logger = logging.getLogger(__name__)


def _check_config() -> None:
    """启动自检：把「配置没读到」这种静默故障变成一条显眼的 warning。"""
    env_file = BACKEND_DIR / ".env"
    if not env_file.exists():
        logger.warning("未找到配置文件 %s，将只使用环境变量与代码默认值。", env_file)
    if not settings.MYSQL_PASSWORD:
        logger.warning(
            "MYSQL_PASSWORD 为空：若数据库设置了密码，连接会报 1045。"
            "请确认 %s 中的 MYSQL_PASSWORD，或改用环境变量注入。", env_file,
        )


def _warmup_rerank() -> None:
    """精排模型预热：2 GB 权重在 CPU 上首次加载要 15–40 s，
    放到后台线程里加载，避免第一个带精排的请求卡近一分钟。
    """
    if not settings.RAG_RERANK_ENABLED or not settings.RAG_RERANK_WARMUP:
        return

    def _run() -> None:
        try:
            from app.services import rerank_service
            ok = rerank_service.get_reranker().load()
            logger.info("精排预热%s", "完成" if ok else "失败（已回退 cosine，原因见上一条 warning）")
        except Exception as exc:  # pragma: no cover - 预热失败不应影响服务
            logger.warning("精排预热异常：%s", exc)

    threading.Thread(target=_run, name="rerank-warmup", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _check_config()
    _warmup_rerank()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "name": settings.APP_NAME,
        "mode": "local",
        "database": "MySQL 8.x",
        "local_model": settings.OLLAMA_MODEL,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "deepseek_mode": "DeepSeek API",
        "docker": False,
        "virtual_machine": False,
    }

app.include_router(chat_router, prefix=settings.API_PREFIX, dependencies=[Depends(require_auth)])
app.include_router(prompt_router, prefix=settings.API_PREFIX, dependencies=[Depends(require_auth)])
app.include_router(knowledge_router, prefix=settings.API_PREFIX, dependencies=[Depends(require_auth)])
app.include_router(logs_router, prefix=settings.API_PREFIX, dependencies=[Depends(require_auth)])
app.include_router(conversations_router, prefix=settings.API_PREFIX, dependencies=[Depends(require_auth)])
