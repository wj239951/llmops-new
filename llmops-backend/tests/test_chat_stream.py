# -*- coding: utf-8 -*-
"""流式聊天测试：SSE 事件顺序、RAG 元数据、错误兜底与日志落库，LLM/RAG 全 mock。"""
import json

from app.core.config import settings
from app.models.chat_log import ChatLog
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService, RetrievedChunk

FAKE_DONE = {
    "done": True,
    "answer": "你好",
    "reasoning": "先思考一下",
    "model_provider": "deepseek",
    "model_name": "deepseek-chat",
    "total_tokens": 9,
}


def _fake_stream(self, **kwargs):
    yield {"reasoning": "先思考"}
    yield {"reasoning": "一下"}
    yield {"content": "你"}
    yield {"content": "好"}
    yield FAKE_DONE


def _failing_stream(self, **kwargs):
    if False:  # pragma: no cover - 让函数成为生成器
        yield {}
    raise RuntimeError("模型调用失败")


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name = "message"
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if data_lines:
            events.append((name, json.loads("\n".join(data_lines))))
    return events


def test_stream_requires_auth(client):
    r = client.post("/api/chat/stream", json={"query": "你好"})
    assert r.status_code == 401


def test_stream_event_order_and_meta(client, auth_headers, monkeypatch):
    # 本用例验证的是「有模型流」时的 SSE 顺序，关掉无命中固定话术走旧路径
    monkeypatch.setattr(settings, "RAG_NO_HIT_REPLY_ENABLED", False)
    monkeypatch.setattr(LLMService, "chat_stream", _fake_stream)
    monkeypatch.setattr(RAGService, "search", lambda self, db, query, use_rerank=None, timing=None: [])

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"query": "今天中午吃什么", "use_rag": True},
        headers=auth_headers,
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())

    events = _parse_sse(text)
    names = [name for name, _ in events]
    assert names == [
        "status", "status", "meta", "status",
        "reasoning", "reasoning", "delta", "delta", "done",
    ]
    assert events[1][1]["phase"] == "retrieving"
    meta = events[2][1]
    assert meta["rag_status"] == "no_hit"
    assert meta["sources"] == []
    assert meta["model_provider"] == "ollama"
    assert meta["model_name"] == settings.OLLAMA_MODEL
    assert meta["score_mode"] == "cosine"
    assert meta["rerank_requested"] is False
    done = events[-1][1]
    assert done["answer"] == "你好"
    assert done["reasoning"] == "先思考一下"
    assert done["total_tokens"] == 9
    assert done["first_token_ms"] is not None


def test_stream_meta_reports_hit_and_sources(db_session, monkeypatch):
    monkeypatch.setattr(LLMService, "chat_stream", _fake_stream)
    monkeypatch.setattr(
        RAGService,
        "search",
        lambda self, db, query, use_rerank=None, timing=None: [
            RetrievedChunk(title="巡检手册", content="润滑脂用 RL-2077", source=None, score=0.8)
        ],
    )
    monkeypatch.setattr(RAGService, "active_score_config",
                            lambda self, use_rerank=None: (0.53, 0.06, "cosine"))
    monkeypatch.setattr(RAGService, "build_context", lambda self, docs: "润滑脂用 RL-2077")

    service = ChatService()
    events = list(service.stream(db_session, ChatRequest(query="润滑脂型号", use_rag=True)))
    meta = [e for e in events if e["event"] == "meta"][0]["data"]
    assert meta["rag_status"] == "hit"
    assert meta["sources"][0]["title"] == "巡检手册"
    assert meta["sources"][0]["score"] == 0.8


def test_stream_saves_success_log(db_session, monkeypatch):
    monkeypatch.setattr(LLMService, "chat_stream", _fake_stream)
    monkeypatch.setattr(RAGService, "search", lambda self, db, query, use_rerank=None, timing=None: [])

    service = ChatService()
    events = list(service.stream(db_session, ChatRequest(query="你好", use_rag=False)))
    assert events[-1]["event"] == "done"
    log = db_session.query(ChatLog).one()
    assert log.status == "success"
    assert log.assistant_output == "你好"
    assert log.total_tokens == 9


def test_stream_error_event_and_failed_log(db_session, monkeypatch):
    monkeypatch.setattr(LLMService, "chat_stream", _failing_stream)
    monkeypatch.setattr(RAGService, "search", lambda self, db, query, use_rerank=None, timing=None: [])

    service = ChatService()
    events = list(service.stream(db_session, ChatRequest(query="你好", use_rag=False)))
    assert events[-1]["event"] == "error"
    assert "模型调用失败" in events[-1]["data"]["message"]
    log = db_session.query(ChatLog).one()
    assert log.status == "failed"
    assert "模型调用失败" in log.error_message
