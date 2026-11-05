# -*- coding: utf-8 -*-
"""针对 2026-11-05 验证报告 6 项修复的回归用例。

覆盖：
1. .env 路径与进程 cwd 无关（配置绝对路径）；
2. 型号/编号「精确命中豁免」（混合检索召回的低分片段不再被 cosine 阈值误杀）；
3. 知识库分页（对象版 + 列表版响应头）；
4. 精排预热不阻塞启动（仅断言开关关闭时不加载，避免 CI 起线程）。
"""
import threading

from app.core.config import BACKEND_DIR, settings
from app.services.rag_service import RAGService, RetrievedChunk, exact_tokens, has_exact_hit


# ── 1. 配置路径 ──────────────────────────────────────────────────────────
def test_env_file_is_absolute_and_under_backend_dir():
    env_file = settings.model_config["env_file"]
    assert str(BACKEND_DIR) in str(env_file)
    assert str(env_file).endswith(".env")


def test_exact_token_extraction():
    assert exact_tokens("ZX-2026 巡检周期") == {"zx-2026"}
    assert exact_tokens("GA-90 压差多少") == {"ga-90"}
    assert exact_tokens("锅水 pH 值范围") == set()          # 无编号，不触发豁免
    assert has_exact_hit("ZX-2026", "注塑机 ZX-2026 巡检手册") is True
    assert has_exact_hit("ZX-2026", "空压机保养规程") is False


# ── 2. 精确命中豁免 ──────────────────────────────────────────────────────
def _chunk(score: float, match: str, title: str = "doc") -> RetrievedChunk:
    return RetrievedChunk(
        title=title, content="ZX-2026 内容", source="t", score=score, match=match,
    )


def test_exact_match_bypasses_cosine_threshold(monkeypatch):
    rag = RAGService()
    monkeypatch.setattr(settings, "RAG_HYBRID_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_HYBRID_EXACT_BYPASS", True)
    from app.services import hybrid_service
    monkeypatch.setattr(hybrid_service, "is_available", lambda : True)
    monkeypatch.setattr(
        RAGService, "_hybrid_candidates",
        lambda self, db, query, k: [_chunk(0.4207, "exact"), _chunk(0.49, "vector")],
    )
    hits = rag.search(None, "ZX-2026")
    assert [h.match for h in hits] == ["exact"]       # 低分但精确命中：放行并排在最前


def test_vector_low_score_still_filtered(monkeypatch):
    rag = RAGService()
    monkeypatch.setattr(settings, "RAG_HYBRID_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_HYBRID_EXACT_BYPASS", True)
    from app.services import hybrid_service
    monkeypatch.setattr(hybrid_service, "is_available", lambda : True)
    monkeypatch.setattr(
        RAGService, "_hybrid_candidates",
        lambda self, db, query, k: [_chunk(0.4207, "vector")],
    )
    assert rag.search(None, "ZX-2026") == []          # 普通低分片段仍被闸门拦住


def test_exact_bypass_can_be_disabled(monkeypatch):
    rag = RAGService()
    monkeypatch.setattr(settings, "RAG_HYBRID_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_HYBRID_EXACT_BYPASS", False)
    from app.services import hybrid_service
    monkeypatch.setattr(hybrid_service, "is_available", lambda : True)
    monkeypatch.setattr(
        RAGService, "_hybrid_candidates",
        lambda self, db, query, k: [_chunk(0.4207, "exact")],
    )
    assert rag.search(None, "ZX-2026") == []


# ── 3. 知识库分页 ────────────────────────────────────────────────────────
def _add_docs(client, headers, n):
    for i in range(n):
        resp = client.post("/api/knowledge", json={
            "title": f"分页测试-{i}", "content": f"内容 {i}", "source": "test",
        }, headers=headers)
        assert resp.status_code == 200


def test_knowledge_paged_object(client, auth_headers):
    _add_docs(client, auth_headers, 3)
    resp = client.get("/api/knowledge/paged?page=1&page_size=2", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["page"] == 1 and body["page_size"] == 2


def test_knowledge_list_header_total(client, auth_headers):
    _add_docs(client, auth_headers, 3)
    resp = client.get("/api/knowledge?page=2&page_size=2", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.headers["X-Total-Count"] == "3"
    assert len(resp.json()) == 1          # 第 2 页只剩 1 条


# ── 4. 启动自检与预热开关 ────────────────────────────────────────────────
def test_warmup_skipped_when_disabled(monkeypatch):
    from app.main import _warmup_rerank
    monkeypatch.setattr(settings, "RAG_RERANK_ENABLED", False)
    before = threading.active_count()
    _warmup_rerank()
    assert threading.active_count() == before     # 未开启精排时不起线程
