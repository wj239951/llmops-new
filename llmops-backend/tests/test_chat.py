# -*- coding: utf-8 -*-
"""聊天接口回归测试（鉴权 + RAG 三态），LLM/RAG 全 mock，不碰 Ollama/Chroma/MySQL。"""
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService, RetrievedChunk

FAKE_RESULT = {
    "answer": "测试回答",
    "reasoning": "",
    "model_provider": "ollama",
    "model_name": "test-model",
    "total_tokens": 10,
}


def test_chat_requires_auth(client):
    r = client.post("/api/chat", json={"query": "你好"})
    assert r.status_code == 401


def test_chat_rejects_wrong_auth(client):
    r = client.post(
        "/api/chat",
        json={"query": "你好"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401


def test_chat_no_hit(client, auth_headers, monkeypatch):
    monkeypatch.setattr(LLMService, "chat", lambda self, **kw: FAKE_RESULT)
    monkeypatch.setattr(RAGService, "search", lambda self, db, query: [])

    r = client.post(
        "/api/chat",
        json={"query": "今天中午吃什么", "use_rag": True},
        headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "测试回答"
    assert body["rag_status"] == "no_hit"
    assert body["sources"] == []


def test_chat_hit(client, auth_headers, monkeypatch):
    monkeypatch.setattr(LLMService, "chat", lambda self, **kw: FAKE_RESULT)
    monkeypatch.setattr(
        RAGService,
        "search",
        lambda self, db, query: [RetrievedChunk(title="标题", content="内容", source=None, score=0.7)],
    )

    r = client.post(
        "/api/chat",
        json={"query": "润滑脂型号是什么", "use_rag": True},
        headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["rag_status"] == "hit"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["title"] == "标题"
    assert body["sources"][0]["score"] == 0.7


def test_chat_weak_hit_when_top1_barely_passes(client, auth_headers, monkeypatch):
    """top1 只比阈值高一点点时不能报「命中」，必须降级为 weak_hit。

    这是 P0 那条「三条全错却显示命中」的可见性修复：闸门勉强过线的场景
    在 UI 上要能看出来，而不是和正常命中长得一样。
    """
    from app.core.config import settings

    barely = settings.RAG_SCORE_THRESHOLD + settings.RAG_WEAK_MARGIN / 2
    assert settings.RAG_SCORE_THRESHOLD <= barely < settings.RAG_SCORE_THRESHOLD + settings.RAG_WEAK_MARGIN
    monkeypatch.setattr(LLMService, "chat", lambda self, **kw: FAKE_RESULT)
    monkeypatch.setattr(
        RAGService,
        "search",
        lambda self, db, query: [RetrievedChunk(title="标题", content="内容", source=None, score=barely)],
    )

    r = client.post(
        "/api/chat",
        json={"query": "氧气含量多少才算合格", "use_rag": True},
        headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["rag_status"] == "weak_hit"
    # 弱命中仍然把来源与分数交出来，让人能自己判断
    assert body["sources"][0]["score"] == barely


def test_chat_hit_boundary_is_inclusive(client, auth_headers, monkeypatch):
    """正好等于 阈值+余量 时应判 hit，避免边界抖动。"""
    from app.core.config import settings

    edge = settings.RAG_SCORE_THRESHOLD + settings.RAG_WEAK_MARGIN
    monkeypatch.setattr(LLMService, "chat", lambda self, **kw: FAKE_RESULT)
    monkeypatch.setattr(
        RAGService,
        "search",
        lambda self, db, query: [RetrievedChunk(title="标题", content="内容", source=None, score=edge)],
    )
    r = client.post(
        "/api/chat",
        json={"query": "润滑脂", "use_rag": True},
        headers=auth_headers,
    )
    assert r.json()["rag_status"] == "hit"
