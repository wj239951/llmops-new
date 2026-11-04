# -*- coding: utf-8 -*-
"""混合检索（jieba/BM25 稀疏召回 + 向量召回 RRF 融合）单元测试。"""
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.models.knowledge import KnowledgeDocument
from app.services import hybrid_service, rag_service
from app.services.hybrid_service import BM25Index, HybridDoc, rrf_score, tokenize
from app.services.rag_service import RAGService, RetrievedChunk


def _hybrid_doc(vec_id, doc_id, content, embedding=None, chunk=0):
    return HybridDoc(
        vec_id=vec_id,
        doc_id=doc_id,
        title=f"标题{doc_id}" if doc_id is not None else "",
        chunk=chunk,
        content=content,
        embedding=embedding,
    )


def _seed_docs(db_session, n):
    ids = []
    for i in range(n):
        d = KnowledgeDocument(title=f"标题{i}", content=f"正文{i}")
        db_session.add(d)
        db_session.commit()
        db_session.refresh(d)
        ids.append(d.id)
    return ids


def _make_hit(doc_id, page_content, distance, chunk=0):
    doc = MagicMock()
    doc.metadata = {"id": doc_id, "title": "t", "chunk": chunk}
    doc.page_content = page_content
    return doc, distance


# ---------------------------------------------------------------------------
# 分词 / BM25

def test_tokenize_keeps_model_codes_and_drops_stopwords():
    tokens = tokenize("RL-2077 润滑脂的更换周期，E-073 传感器")
    assert "rl-2077" in tokens
    assert "e-073" in tokens
    assert "的" not in tokens  # 单字/停用词被过滤


def test_bm25_ranks_exact_model_number_first():
    docs = [
        _hybrid_doc("a", 1, "空压机排气温度范围 75–95 摄氏度。"),
        _hybrid_doc("b", 2, "螺杆传动链条必须使用 RL-2077 润滑脂。"),
        _hybrid_doc("c", 3, "受限空间氧含量 19.5%–21%。"),
    ]
    hits = BM25Index(docs).search("RL-2077 润滑脂", k=3)
    assert hits
    assert hits[0][0].doc_id == 2


def test_rrf_score_formula():
    assert rrf_score(1, 60) == pytest.approx(1.0 / 61.0)
    assert rrf_score(2, 60) < rrf_score(1, 60)


# ---------------------------------------------------------------------------
# RRF 融合

def test_rrf_fuse_prefers_doc_hit_by_both_channels():
    dense = [
        RetrievedChunk(title="t", content="dense-only", source=None, score=0.9, doc_id=1, chunk=0),
        RetrievedChunk(title="t", content="both", source=None, score=0.8, doc_id=2, chunk=0),
    ]
    sparse = [
        (_hybrid_doc("v2", 2, "both"), 3.0),
        (_hybrid_doc("v3", 3, "sparse-only"), 2.0),
    ]
    rows = RAGService()._rrf_fuse(dense, sparse, rrf_k=60)
    assert rows[0]["chunk"] is not None and rows[0]["chunk"].doc_id == 2
    ids = {row["chunk"].doc_id if row["chunk"] else row["hybrid_doc"].doc_id for row in rows}
    assert ids == {1, 2, 3}


# ---------------------------------------------------------------------------
# search() 混合检索路径与回退

def test_hybrid_search_merges_bm25_only_candidate(db_session, monkeypatch):
    ids = _seed_docs(db_session, 2)
    dense = [
        RetrievedChunk(title="标题0", content="正文0", source=None, score=0.9, doc_id=ids[0], chunk=0)
    ]
    monkeypatch.setattr(RAGService, "_dense_chunks", lambda self, db, query, k: list(dense))
    monkeypatch.setattr(settings, "RAG_HYBRID_ENABLED", True)
    monkeypatch.setattr(hybrid_service, "is_available", lambda: True)
    sparse_doc = _hybrid_doc("v1", ids[1], "RL-2077 润滑脂正文", embedding=(3.0, 4.0))
    monkeypatch.setattr(hybrid_service, "bm25_search", lambda query, k: [(sparse_doc, 5.0)])
    monkeypatch.setattr(hybrid_service, "embed_query", lambda query: [1.0, 0.0])

    out = RAGService().search(db_session, "RL-2077 润滑脂")
    assert {c.doc_id for c in out} == {ids[0], ids[1]}
    merged = next(c for c in out if c.doc_id == ids[1])
    # BM25 独有候选没有 cosine，服务端用 Chroma 里存的向量补算：(3,4)·(1,0)/5 = 0.6
    assert merged.score == pytest.approx(0.6)
    assert merged.title == "标题1"


def test_hybrid_falls_back_to_dense_when_index_unavailable(db_session, monkeypatch):
    ids = _seed_docs(db_session, 1)
    monkeypatch.setattr(
        rag_service.vectorstore,
        "similarity_search_with_score",
        MagicMock(return_value=[_make_hit(ids[0], "正文0", 0.4)]),
    )
    monkeypatch.setattr(settings, "RAG_HYBRID_ENABLED", True)
    monkeypatch.setattr(hybrid_service, "is_available", lambda: False)

    out = RAGService().search(db_session, "查询")
    assert len(out) == 1
    assert out[0].score == pytest.approx(0.6)
