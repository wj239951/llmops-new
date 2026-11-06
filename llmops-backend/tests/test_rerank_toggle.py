# -*- coding: utf-8 -*-
"""精排请求级开关（ChatRequest.use_rerank）测试。"""
from app.core.config import settings
from app.services.rag_service import RAGService


def test_rerank_state_override_off_and_on(monkeypatch):
    monkeypatch.setattr(settings, "RAG_RERANK_ENABLED", True)
    from app.services import rerank_service
    monkeypatch.setattr(rerank_service, "is_available", lambda force=False: True)

    rag = RAGService()
    assert rag.rerank_state(False) == (False, False, "cosine")
    assert rag.rerank_state(True) == (True, True, "rerank")
    assert rag.rerank_state(None) == (True, True, "rerank")
    assert rag.active_score_config(False)[2] == "cosine"
    assert rag.active_score_config(True)[2] == "rerank"


def test_rerank_state_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "RAG_RERANK_ENABLED", False)
    from app.services import rerank_service
    monkeypatch.setattr(rerank_service, "is_available", lambda force=False: False)

    rag = RAGService()
    assert rag.rerank_state(True) == (True, False, "cosine")
    threshold, margin, mode = rag.active_score_config(True)
    assert mode == "cosine"
    assert threshold == settings.RAG_SCORE_THRESHOLD
    assert margin == settings.RAG_WEAK_MARGIN


def test_chat_response_reports_requested_but_unavailable(client, auth_headers, monkeypatch):
    from app.services.llm_service import LLMService
    from app.services.rag_service import RAGService

    monkeypatch.setattr(LLMService, "chat", lambda self, **kwargs: {
        "answer": "ok",
        "reasoning": "",
        "model_provider": "ollama",
        "model_name": settings.OLLAMA_MODEL,
        "total_tokens": 1,
    })
    monkeypatch.setattr(RAGService, "search", lambda self, db, query, use_rerank=None, timing=None: [])
    from app.services import rerank_service
    monkeypatch.setattr(rerank_service, "is_available", lambda force=False: False)

    resp = client.post(
        "/api/chat",
        json={"query": "你好", "use_rag": True, "use_rerank": True},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["rerank_requested"] is True
    assert body["rerank_available"] is False
    assert body["score_mode"] == "cosine"
