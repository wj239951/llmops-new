# -*- coding: utf-8 -*-
"""P2 精排两级检索回归测试（不加载真实模型：mock 掉精排器）。"""
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.models.knowledge import KnowledgeDocument
from app.services import rag_service, rerank_service
from app.services.rag_service import RAGService, RetrievedChunk


def _make_hit(doc_id, page_content, distance):
    """伪造一条 (Document, distance)，distance 是 Chroma 的 1-cosine。"""
    doc = MagicMock()
    doc.metadata = {"id": doc_id, "title": "t", "chunk": 0}
    doc.page_content = page_content
    return doc, distance


def _seed_docs(db_session, n):
    ids = []
    for i in range(n):
        d = KnowledgeDocument(title=f"标题{i}", content=f"正文{i}")
        db_session.add(d)
        db_session.commit()
        db_session.refresh(d)
        ids.append(d.id)
    return ids


@pytest.fixture
def rerank_on(monkeypatch):
    """打开精排开关；模型加载/推理由各用例单独 mock。"""
    monkeypatch.setattr(settings, "RAG_RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RAG_RERANK_CANDIDATE_K", 12)
    monkeypatch.setattr(settings, "RAG_RERANK_SCORE_THRESHOLD", 0.5)
    monkeypatch.setattr(settings, "RAG_RERANK_WEAK_MARGIN", 0.06)
    yield


def test_rerank_reorders_candidates_and_filters_by_rerank_score(
    db_session, monkeypatch, rerank_on
):
    """核心回归：精排分数应覆盖 cosine 排序，且用精排阈值过滤。"""
    ids = _seed_docs(db_session, 4)
    hits = [
        _make_hit(ids[0], "正文0", 0.10),  # cosine 0.90 → 精排 0.20（被降权）
        _make_hit(ids[1], "正文1", 0.20),  # cosine 0.80 → 精排 0.95
        _make_hit(ids[2], "正文2", 0.30),  # cosine 0.70 → 精排 0.60
        _make_hit(ids[3], "正文3", 0.40),  # cosine 0.60 → 精排 0.40（低于阈值）
    ]
    mock = MagicMock(return_value=hits)
    monkeypatch.setattr(rag_service.vectorstore, "similarity_search_with_score", mock)

    rerank_scores = {"正文0": 0.20, "正文1": 0.95, "正文2": 0.60, "正文3": 0.40}
    monkeypatch.setattr(rerank_service, "is_available", lambda: True)
    monkeypatch.setattr(
        rerank_service,
        "rerank_chunks",
        lambda q, chunks, top_k=None: sorted(
            [RetrievedChunk(c.title, c.content, c.source, rerank_scores[c.content]) for c in chunks],
            key=lambda c: c.score,
            reverse=True,
        ),
    )

    out = RAGService().search(db_session, "查询")
    # 召回宽度应放宽到 RAG_RERANK_CANDIDATE_K
    assert mock.call_args.kwargs["k"] == settings.RAG_RERANK_CANDIDATE_K
    # 精排顺序：正文1(0.95) > 正文2(0.60)；正文3(0.40) 被阈值过滤
    assert [c.content for c in out] == ["正文1", "正文2"]
    assert all(c.score >= settings.RAG_RERANK_SCORE_THRESHOLD for c in out)


def test_rerank_falls_back_to_cosine_when_unavailable(db_session, monkeypatch, rerank_on):
    """开了精排但模型不可用时，必须退回原有 cosine 过滤，不能让服务不可用。"""
    ids = _seed_docs(db_session, 3)
    hits = [
        _make_hit(ids[0], "正文0", 0.20),  # cosine 0.80 → 过
        _make_hit(ids[1], "正文1", 0.30),  # cosine 0.70 → 过
        _make_hit(ids[2], "正文2", 0.60),  # cosine 0.40 → 不过
    ]
    monkeypatch.setattr(
        rag_service.vectorstore, "similarity_search_with_score", MagicMock(return_value=hits)
    )
    monkeypatch.setattr(rerank_service, "is_available", lambda: False)

    out = RAGService().search(db_session, "查询")
    assert [c.content for c in out] == ["正文0", "正文1"]
    assert all(c.score >= settings.RAG_SCORE_THRESHOLD for c in out)


def test_active_score_config_switches_to_rerank_thresholds(monkeypatch, rerank_on):
    monkeypatch.setattr(rerank_service, "is_available", lambda: True)
    threshold, margin, mode = RAGService().active_score_config()
    assert mode == "rerank"
    assert threshold == settings.RAG_RERANK_SCORE_THRESHOLD
    assert margin == settings.RAG_RERANK_WEAK_MARGIN


def test_active_score_config_defaults_to_cosine():
    threshold, margin, mode = RAGService().active_score_config()
    assert mode == "cosine"
    assert threshold == settings.RAG_SCORE_THRESHOLD
    assert margin == settings.RAG_WEAK_MARGIN
