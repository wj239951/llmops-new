# -*- coding: utf-8 -*-
"""拒答策略测试（docs/17）：无命中固定话术 + 命中提示词收紧。"""
import pytest

from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService, RetrievedChunk


def _chunk(score=0.8):
    return RetrievedChunk(title="巡检手册", content="润滑脂型号为 RL-2077", source=None, score=score)


def test_strict_grounding_prompt_forbids_general_knowledge():
    prompt = ChatService._grounded_prompt("【知识片段1】润滑脂用 RL-2077", "用什么润滑脂")
    assert "只依据" in prompt
    assert "知识库未收录该信息" in prompt
    assert "结合通用知识补充" not in prompt
    assert "RL-2077" in prompt and "用什么润滑脂" in prompt


def test_prompt_rollback_uses_old_template(monkeypatch):
    monkeypatch.setattr(settings, "RAG_STRICT_GROUNDING", False)
    prompt = ChatService._grounded_prompt("正文", "问题")
    assert "结合通用知识补充" in prompt


def test_no_hit_stream_returns_canned_without_calling_model(db_session, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("no_hit 不应调用模型")

    monkeypatch.setattr(LLMService, "chat_stream", _boom)
    monkeypatch.setattr(RAGService, "search", lambda self, db, query, use_rerank=None, timing=None: [])

    events = list(ChatService().stream(db_session, ChatRequest(query="年假有几天", use_rag=True)))
    meta = [e for e in events if e["event"] == "meta"][0]["data"]
    done = [e for e in events if e["event"] == "done"][0]["data"]
    deltas = [e["data"]["content"] for e in events if e["event"] == "delta"]

    assert meta["rag_status"] == "no_hit"
    assert meta["sources"] == []
    assert "".join(deltas) == settings.RAG_NO_HIT_REPLY_TEXT
    assert done["answer"] == settings.RAG_NO_HIT_REPLY_TEXT
    assert done["total_tokens"] == 0
    assert done["timings"]["llm_ms"] == 0


def test_hit_stream_uses_grounded_prompt(db_session, monkeypatch):
    captured = {}

    def _fake_chat_stream(self, **kw):
        captured.update(kw)
        yield {"content": "RL-2077"}
        yield {"done": True, "answer": "RL-2077", "reasoning": "",
               "model_provider": "ollama", "model_name": "m", "total_tokens": 1}

    monkeypatch.setattr(LLMService, "chat_stream", _fake_chat_stream)
    monkeypatch.setattr(RAGService, "search",
                        lambda self, db, query, use_rerank=None, timing=None: [_chunk()])
    monkeypatch.setattr(RAGService, "active_score_config",
                        lambda self, use_rerank=None: (0.56, 0.06, "cosine"))

    events = list(ChatService().stream(db_session, ChatRequest(query="润滑脂型号", use_rag=True)))
    assert [e for e in events if e["event"] == "meta"][0]["data"]["rag_status"] == "hit"
    assert "只依据" in captured["query"]
    assert "RL-2077" in captured["query"]
