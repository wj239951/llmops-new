# -*- coding: utf-8 -*-
"""云端精排（RAG_RERANK_PROVIDER=cloud）回归：mock HTTP，不发真实请求。"""
from unittest.mock import MagicMock

import pytest
import requests

from app.core.config import settings
from app.services import rerank_service
from app.services.rag_service import RetrievedChunk
from app.services.rerank_service import ApiReranker, CrossEncoderReranker, RerankerUnavailableError


@pytest.fixture
def cloud(monkeypatch):
    """切到 cloud 模式并重置单例；requests.post 由各用例单独 mock。"""
    monkeypatch.setattr(settings, "RAG_RERANK_PROVIDER", "cloud")
    monkeypatch.setattr(settings, "RAG_RERANK_API_KEY", "sk-test")
    monkeypatch.setattr(rerank_service, "_reranker", None)
    yield monkeypatch


def _mock_post(monkeypatch, results):
    resp = MagicMock()
    resp.json.return_value = {"results": results}
    post = MagicMock(return_value=resp)
    monkeypatch.setattr(rerank_service.requests, "post", post)
    return post


def test_provider_switches_singleton(cloud):
    assert isinstance(rerank_service.get_reranker(), ApiReranker)
    cloud.setattr(settings, "RAG_RERANK_PROVIDER", "local")
    assert isinstance(rerank_service.get_reranker(), CrossEncoderReranker)


def test_score_restores_input_order_from_index(cloud):
    post = _mock_post(cloud, [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.1},
        {"index": 1, "relevance_score": 0.5},
    ])
    scores = rerank_service.get_reranker().score("查询", ["a", "b", "c"])
    assert scores == [0.1, 0.5, 0.9]
    payload = post.call_args.kwargs["json"]
    assert payload["documents"] == ["a", "b", "c"]
    assert payload["top_n"] == 3
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"


def test_rerank_chunks_sorts_by_cloud_score(cloud):
    _mock_post(cloud, [
        {"index": 0, "relevance_score": 0.2},
        {"index": 1, "relevance_score": 0.9},
    ])
    chunks = [
        RetrievedChunk("t0", "正文0", "s", 0.9),
        RetrievedChunk("t1", "正文1", "s", 0.1),
    ]
    out = rerank_service.rerank_chunks("查询", chunks)
    assert [c.content for c in out] == ["正文1", "正文0"]
    assert [c.score for c in out] == [0.9, 0.2]


def test_missing_key_falls_back(cloud):
    cloud.setattr(settings, "RAG_RERANK_API_KEY", "")
    cloud.setattr(settings, "SILICONFLOW_API_KEY", "")
    cloud.setattr(settings, "RAG_RERANK_ENABLED", True)
    assert rerank_service.is_available() is False
    assert "key" in rerank_service.get_reranker().load_error


def test_key_falls_back_to_siliconflow(cloud):
    cloud.setattr(settings, "RAG_RERANK_API_KEY", "")
    cloud.setattr(settings, "SILICONFLOW_API_KEY", "sk-emb")
    post = _mock_post(cloud, [{"index": 0, "relevance_score": 0.8}])
    assert rerank_service.get_reranker().score("q", ["a"]) == [0.8]
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-emb"


def test_http_failure_raises_unavailable_for_cosine_fallback(cloud):
    post = MagicMock(side_effect=requests.exceptions.ConnectionError("boom"))
    cloud.setattr(rerank_service.requests, "post", post)
    with pytest.raises(RerankerUnavailableError):
        rerank_service.get_reranker().score("q", ["a"])