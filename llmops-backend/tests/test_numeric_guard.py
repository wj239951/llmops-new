# -*- coding: utf-8 -*-
"""数字级 grounding 校验测试（docs/19）。"""
from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.services import grounding
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService, RetrievedChunk

CONTEXT = "注塑机 ZX-1200 巡检与保养手册\n锁模力 1200 千牛，液压油滤芯每 400 小时更换一次。"


def test_numbers_present_are_ok():
    r = grounding.check_numbers("ZX-1200 锁模力 1200 千牛，滤芯每 400 小时换一次【依据：知识片段1】", CONTEXT)
    assert r["ok"] is True
    assert r["missing"] == []


def test_fabricated_number_is_flagged():
    r = grounding.check_numbers("锁模力是 1200 千牛，模具重量上限 5000 公斤。", CONTEXT)
    assert r["ok"] is False
    assert r["missing"] == ["5000"]


def test_citation_index_and_single_digits_are_ignored():
    r = grounding.check_numbers("共 3 步：先停机【依据：知识片段2】", CONTEXT)
    assert r["ok"] is True          # 3 与引用编号 2 都不参与校验


def test_thousands_separator_normalized():
    r = grounding.check_numbers("额定容量 1,600 千伏安", "容量 1600 千伏安")
    assert r["ok"] is True


def test_guard_appends_warning_only_when_missing(monkeypatch):
    ctx = {"rag_docs": [RetrievedChunk(title="手册", content="锁模力 1200 千牛", source=None, score=0.9)]}
    ok_answer, ok_meta = ChatService._guard_numbers("锁模力 1200 千牛", ctx)
    assert ok_answer == "锁模力 1200 千牛" and ok_meta["ok"] is True

    bad_answer, bad_meta = ChatService._guard_numbers("锁模力 1200 千牛，重量 5000 公斤", ctx)
    assert bad_meta["ok"] is False
    assert "5000" in bad_answer and "未在知识库片段中找到依据" in bad_answer


def test_guard_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RAG_NUMERIC_GUARD_ENABLED", False)
    ctx = {"rag_docs": [RetrievedChunk(title="手册", content="锁模力 1200 千牛", source=None, score=0.9)]}
    answer, meta = ChatService._guard_numbers("重量 5000 公斤", ctx)
    assert answer == "重量 5000 公斤" and meta is None


def test_stream_appends_grounding_warning(db_session, monkeypatch):
    def _fake_stream(self, **kw):
        yield {"content": "锁模力 1200 千牛，整机重量 5000 公斤"}
        yield {"done": True, "answer": "锁模力 1200 千牛，整机重量 5000 公斤",
               "reasoning": "", "model_provider": "ollama", "model_name": "m", "total_tokens": 1}

    monkeypatch.setattr(LLMService, "chat_stream", _fake_stream)
    monkeypatch.setattr(RAGService, "search",
                        lambda self, db, query, use_rerank=None, timing=None:
                        [RetrievedChunk(title="手册", content="锁模力 1200 千牛", source=None, score=0.9)])
    monkeypatch.setattr(RAGService, "active_score_config",
                        lambda self, use_rerank=None: (0.56, 0.06, "cosine"))

    events = list(ChatService().stream(db_session, ChatRequest(query="锁模力", use_rag=True)))
    done = [e for e in events if e["event"] == "done"][0]["data"]
    assert done["grounding"]["ok"] is False
    assert "5000" in done["answer"]
    assert "未在知识库片段中找到依据" in done["answer"]
