# 创建 MySQL 连接、 Session 会话、初始化数据库表。
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import settings

class Base(DeclarativeBase):
    pass

engine = create_engine(settings.SQLALCHEMY_DATABASE_URL,
                       pool_pre_ping=True,
                       pool_recycle=3600,
                       echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _ensure_schema() -> None:
    """轻量迁移：给已存在的表补新列。

    `create_all` 只建缺失的表，不会给老表加列；这里对已知的新增列做一次
    「查 information_schema → 没有就 ALTER」，避免老库升级后启动即报 Unknown column。
    只对 MySQL 生效，测试用的 SQLite 由 create_all 直接建全。
    """
    from sqlalchemy import inspect, text

    if engine.dialect.name != "mysql":
        return
    insp = inspect(engine)
    if "llm_chat_logs" not in insp.get_table_names():
        return
    columns = {c["name"] for c in insp.get_columns("llm_chat_logs")}
    if "timings" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE llm_chat_logs ADD COLUMN timings JSON NULL"))


def init_db():
    import app.models.prompt  # noqa: F401
    import app.models.knowledge  # noqa: F401
    import app.models.chat_log  # noqa: F401
    import app.models.conversation  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _ensure_schema()
