# -*- coding: utf-8 -*-
"""查询分解层回归：mock LLM 不发真实请求；search 集成验证合并/去重/闸门。"""
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.services import query_decompose, rerank_service
from app.services.rag_service import RAGService, RetrievedChunk


def _llm(monkeypatch, answer=None, exc=None):
    cls = MagicMock()
    if exc is not None:
        cls.return_value.chat.side_effect = exc
    else:
        cls.return_value.chat.return_value = {"answer": answer}
    monkeypatch.setattr("app.services.llm_service.LLMService", cls)
    return cls


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "RAG_QUERY_DECOMPOSE_ENABLED", True)
    return monkeypatch


def test_disabled_never_calls_llm(monkeypatch):
    monkeypatch.setattr(settings, "RAG_QUERY_DECOMPOSE_ENABLED", False)
    cls = _llm(monkeypatch, exc=AssertionError("开关关闭时不应调用 LLM"))
    assert query_decompose.decompose("复合问题") == ["复合问题"]
    cls.assert_not_called()


def test_valid_json_split(enabled):
    _llm(enabled, '["驾驶公务车辆出差怎么报销", "公务车辆使用有什么制度"]')
    assert query_decompose.decompose("驾驶公务车辆出差怎么报销，有什么制度") == [
        "驾驶公务车辆出差怎么报销", "公务车辆使用有什么制度",
    ]


def test_fenced_json_with_prose(enabled):
    _llm(enabled, '分解结果如下：\n```json\n["甲主题", "乙主题"]\n```\n以上。')
    assert query_decompose.decompose("复合问题") == ["甲主题", "乙主题"]


def test_garbage_falls_back_to_original(enabled):
    _llm(enabled, "抱歉，我无法分解这个问题。")
    assert query_decompose.decompose("复合问题") == ["复合问题"]


def test_llm_exception_falls_back(enabled):
    _llm(enabled, exc=RuntimeError("模型服务不可用"))
    assert query_decompose.decompose("复合问题") == ["复合问题"]


def test_non_list_json_falls_back(enabled):
    _llm(enabled, '{"sub": "甲"}')
    assert query_decompose.decompose("复合问题") == ["复合问题"]


def test_dedupe_and_cap_at_three(enabled):
    _llm(enabled, '["甲", "甲", "乙", "丙", "丁"]')
    assert query_decompose.decompose("复合问题") == ["甲", "乙", "丙"]


def test_empty_and_non_string_items_filtered(enabled):
    _llm(enabled, '["甲", "", 3, "   "]')
    assert query_decompose.decompose("复合问题") == ["甲"]


def test_single_topic_passthrough(enabled):
    _llm(enabled, '["单主题问题"]')
    assert query_decompose.decompose("单主题问题") == ["单主题问题"]


# ── search 集成：分解 → 各子问句召回/精排 → 合并去重过闸 ─────────────────


def _chunk(doc_id, chunk=0):
    return RetrievedChunk(f"标题{doc_id}", f"正文{doc_id}-{chunk}", "src", 0.5,
                          doc_id=doc_id, chunk=chunk)


@pytest.fixture
def search_env(monkeypatch):
    monkeypatch.setattr(settings, "RAG_RERANK_SCORE_THRESHOLD", 0.76)
    monkeypatch.setattr(RAGService, "rerank_state",
                        lambda self, use_rerank=None: (True, True, "rerank"))
    return monkeypatch


def _fake_rerank(scores):
    return lambda query, chunks: [
        replace(c, score=scores.get((query, c.doc_id), 0.1)) for c in chunks
    ]


def test_search_multi_sub_union_keeps_max_score(search_env):
    q = "复合问题"
    pool = {
        q: [_chunk(1)],
        "子问A": [_chunk(1), _chunk(2)],
        "子问B": [_chunk(1), _chunk(3)],
    }
    search_env.setattr(RAGService, "_retrieve_candidates",
                       lambda self, db, query, k: pool.get(query))
    search_env.setattr(query_decompose, "decompose", lambda query: ["子问A", "子问B"])
    search_env.setattr(rerank_service, "rerank_chunks", _fake_rerank({
        ("子问A", 1): 0.90, ("子问A", 2): 0.80,
        ("子问B", 1): 0.95, ("子问B", 3): 0.70,  # 0.70 < 0.76 应被滤掉
    }))
    out = RAGService().search(None, q)
    # doc1 两个子问句都命中，保留较高分 0.95；按分数降序
    assert [(c.doc_id, c.score) for c in out] == [(1, 0.95), (2, 0.80)]


def test_search_single_sub_keeps_original_path(search_env):
    q = "单主题问题"
    calls = []

    def retrieve(self, db, query, k):
        calls.append(query)
        return [_chunk(1), _chunk(2)]

    search_env.setattr(RAGService, "_retrieve_candidates", retrieve)
    search_env.setattr(query_decompose, "decompose", lambda query: [query])
    search_env.setattr(rerank_service, "rerank_chunks", _fake_rerank({(q, 1): 0.9}))
    out = RAGService().search(None, q)
    assert [c.doc_id for c in out] == [1]
    assert calls == [q]  # 不额外召回


def test_search_failed_sub_retrieval_skipped(search_env):
    q = "复合问题"
    pool = {q: [_chunk(9)], "子问A": None, "子问B": [_chunk(2)]}
    search_env.setattr(RAGService, "_retrieve_candidates",
                       lambda self, db, query, k: pool.get(query))
    search_env.setattr(query_decompose, "decompose", lambda query: ["子问A", "子问B"])
    search_env.setattr(rerank_service, "rerank_chunks", _fake_rerank({("子问B", 2): 0.88}))
    out = RAGService().search(None, q)
    assert [(c.doc_id, c.score) for c in out] == [(2, 0.88)]