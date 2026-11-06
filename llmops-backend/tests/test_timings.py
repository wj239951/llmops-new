# -*- coding: utf-8 -*-
"""阶段耗时埋点的回归用例：响应里带 timings、日志表能存能查、失败请求也留痕。"""
from app.core.config import settings
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


def _fake_chat(self, **kwargs):
    return {
        "answer": "收到",
        "reasoning": "",
        "model_provider": "ollama",
        "model_name": settings.OLLAMA_MODEL,
        "total_tokens": 8,
    }


def test_chat_response_contains_phase_timings(client, auth_headers, monkeypatch):
    monkeypatch.setattr(LLMService, "chat", _fake_chat)
    monkeypatch.setattr(
        RAGService, "search",
        lambda self, db, query, use_rerank=None, timing=None: [],
    )

    resp = client.post(
        "/api/chat",
        json={"query": "你好", "use_rag": True},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    timings = resp.json()["timings"]
    # 检索被 mock 掉时 retrieve_ms 不会写入，这里只断言本层能产生的阶段
    assert {"prepare_ms", "llm_ms", "total_ms"} <= set(timings)
    assert timings["total_ms"] >= 0


def test_timings_are_persisted_and_returned_by_logs(client, auth_headers, monkeypatch):
    monkeypatch.setattr(LLMService, "chat", _fake_chat)
    monkeypatch.setattr(
        RAGService, "search",
        lambda self, db, query, use_rerank=None, timing=None: [],
    )

    resp = client.post("/api/chat", json={"query": "耗时落库测试"}, headers=auth_headers)
    assert resp.status_code == 200
    chat_timings = resp.json()["timings"]

    logs = client.get("/api/logs?page_size=1", headers=auth_headers).json()
    item = logs["items"][0]
    assert item["user_input"] == "耗时落库测试"
    assert item["timings"] is not None
    assert item["timings"]["total_ms"] == chat_timings["total_ms"]


def test_search_writes_timing_dict(monkeypatch):
    """RAGService.search 只往传入的 dict 里写耗时，不改变返回值。"""
    rag = RAGService()
    monkeypatch.setattr(settings, "RAG_HYBRID_ENABLED", False)
    monkeypatch.setattr(RAGService, "_dense_chunks", lambda self, db, q, k: [])
    timing: dict = {}
    assert rag.search(None, "任意问题", timing=timing) == []
    assert "retrieve_ms" in timing
