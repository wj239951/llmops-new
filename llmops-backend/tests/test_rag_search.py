# -*- coding: utf-8 -*-
"""RAG search 回归测试（mock 向量库 + SQLite 文档，不碰真实 Chroma/Ollama）。"""
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.models.knowledge import KnowledgeDocument
from app.services import rag_service
from app.services.rag_service import RAGService, RetrievedChunk


def _make_hit(doc_id, page_content, distance):
    """伪造一条 (Document, distance)，distance 是 Chroma 的 1-cosine。"""
    doc = MagicMock()
    doc.metadata = {"id": doc_id, "title": "t", "chunk": 0}
    doc.page_content = page_content
    return doc, distance


def test_empty_query_returns_empty(db_session):
    assert RAGService().search(db_session, "   ") == []


def test_search_converts_distance_to_cosine(db_session, monkeypatch):
    d = KnowledgeDocument(title="标题", content="正文")
    db_session.add(d)
    db_session.commit()
    db_session.refresh(d)

    # distance=0.4 → cosine 应为 0.6
    monkeypatch.setattr(
        rag_service.vectorstore,
        "similarity_search_with_score",
        MagicMock(return_value=[_make_hit(d.id, "正文", 0.4)]),
    )

    hits = RAGService().search(db_session, "查询")
    assert len(hits) == 1
    assert hits[0].title == "标题"
    assert hits[0].content == "正文"
    assert hits[0].score == pytest.approx(0.6)


def test_orphan_hit_is_skipped(db_session, monkeypatch):
    # metadata 指向一个 MySQL 里不存在的 id → 应被跳过，返回空（不静默假命中）
    monkeypatch.setattr(
        rag_service.vectorstore,
        "similarity_search_with_score",
        MagicMock(return_value=[_make_hit(99999, "孤儿向量", 0.1)]),
    )
    assert RAGService().search(db_session, "查询") == []


def test_build_context_contains_title_and_content():
    chunks = [RetrievedChunk(title="A", content="正文内容", source=None, score=0.9)]
    ctx = RAGService().build_context(chunks)
    assert "标题：A" in ctx
    assert "正文内容" in ctx


# ---------------------------------------------------------------------------
# P0 回归：先取宽候选再过滤截断，别让排在 topk 之后的正确片段永远进不来

def _seed_docs(db_session, n):
    ids = []
    for i in range(n):
        d = KnowledgeDocument(title=f"标题{i}", content=f"正文{i}")
        db_session.add(d)
        db_session.commit()
        db_session.refresh(d)
        ids.append(d.id)
    return ids


def test_search_fetches_more_candidates_than_it_returns(db_session, monkeypatch):
    """核心回归：向 Chroma 要候选的条数必须大于最终喂给模型的条数。"""
    mock = MagicMock(return_value=[])
    monkeypatch.setattr(rag_service.vectorstore, "similarity_search_with_score", mock)
    RAGService().search(db_session, "查询")
    called_k = mock.call_args.kwargs["k"]
    assert called_k >= settings.RAG_CANDIDATE_K
    assert called_k > settings.RAG_FEED_TOPK


def test_search_keeps_candidate_beyond_feed_topk_when_earlier_ones_are_filtered(db_session, monkeypatch):
    """旧实现 k=3 时，第 4 条之后的候选永远拿不到；这里构造该场景并验证已修复。

    候选按 cosine 降序：0.70(过) / 0.50(不过) / 0.52(不过) / 0.54(过) / 0.40(不过) / 0.56(过)
    旧行为：只取前 3 条候选 → 过滤后剩 1 条（第 4、6 条根本没被取到）
    新行为：取 6 条候选 → 过滤后剩 3 条
    """
    ids = _seed_docs(db_session, 6)
    hits = [
        _make_hit(ids[0], "正文0", 0.30),  # cos 0.70 过
        _make_hit(ids[1], "正文1", 0.50),  # cos 0.50 不过
        _make_hit(ids[2], "正文2", 0.48),  # cos 0.52 不过
        _make_hit(ids[3], "正文3", 0.46),  # cos 0.54 过 ← 旧实现取不到
        _make_hit(ids[4], "正文4", 0.60),  # cos 0.40 不过
        _make_hit(ids[5], "正文5", 0.44),  # cos 0.56 过 ← 旧实现取不到
    ]
    mock = MagicMock(return_value=hits)
    monkeypatch.setattr(rag_service.vectorstore, "similarity_search_with_score", mock)

    out = RAGService().search(db_session, "查询")
    titles = sorted(c.title for c in out)
    assert titles == sorted(["标题0", "标题3", "标题5"])
    assert all(c.score >= settings.RAG_SCORE_THRESHOLD for c in out)


def test_search_caps_result_at_feed_topk(db_session, monkeypatch):
    """过阈值的候选再多，最终也只返回 RAG_FEED_TOPK 条。"""
    ids = _seed_docs(db_session, 6)
    hits = [_make_hit(i, f"正文{n}", 0.10 + n * 0.01) for n, i in enumerate(ids)]
    monkeypatch.setattr(
        rag_service.vectorstore, "similarity_search_with_score", MagicMock(return_value=hits)
    )
    out = RAGService().search(db_session, "查询")
    assert len(out) == settings.RAG_FEED_TOPK
