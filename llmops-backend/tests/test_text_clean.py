# -*- coding: utf-8 -*-
"""输入清洗的回归用例。

覆盖三件事：
1. 规则清洗的行为边界（零宽字符 / 全角 / 重复字 / HTML / 乱码 / URL 保护）；
2. 模型通道失败时按配置回退，且不把异常抛到问答主链路；
3. 接入层语义——**用户原文入库、清洗文本进检索与提问**。
"""
import pytest

from app.core.config import settings
from app.services import text_clean
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.text_clean import InputCleaner, clean_rule


# ── 规则清洗 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw, expected", [
    ("你\u200b好\u200c世界", "你好世界"),                    # 零宽字符
    ("Ｈｅｌｌｏ　１２３", "Hello 123"),                      # 全角字母数字与全角空格
    ("我我我 明天 去去 北京", "我 明天 去去 北京"),            # ≥3 次重复压成 1 次；两字叠词保留
    ("太好了！！！", "太好了！"),                             # 连续感叹号
    ("<p>你好&nbsp;世界</p>", "你好 世界"),                   # HTML 实体与标签
    ("æ˜Žå¤©åŽ»åŒ—äº¬", "明天去北京"),                       # UTF-8 被按 Latin-1 解码
    ("看看 https://example.com/a?b=1&c=2 谢谢", "看看 https://example.com/a?b=1&c=2 谢谢"),  # URL 不动
    ("   ", ""),                                             # 纯空白
])
def test_clean_rule(raw, expected):
    assert clean_rule(raw) == expected


def test_clean_rule_keeps_legal_repeats():
    """两字叠词是合法词，不能被压缩——"刚刚"变"刚"会改变语义。"""
    assert clean_rule("刚刚发货，谢谢") == "刚刚发货，谢谢"


# ── 通道选择与降级 ──────────────────────────────────────────────────────
def test_mode_off_returns_original():
    cleaner = InputCleaner(mode="off")
    text, meta = cleaner.preprocess("我我我")
    assert text == "我我我"
    assert meta["applied"] is False and meta["reason"] == "off"


def test_mode_rule_cleans_and_marks_changed():
    text, meta = InputCleaner(mode="rule").preprocess("我我我 明天 去北京！！！")
    assert text == "我 明天 去北京！"
    assert meta["applied"] is True and meta["changed"] is True and meta["mode"] == "rule"


def test_mode_model_falls_back_to_rule(monkeypatch):
    def _boom(*args, **kwargs):
        raise ConnectionError("clean server 挂了")

    monkeypatch.setattr(text_clean, "clean_model", _boom)
    text, meta = InputCleaner(mode="model", fallback="rule").preprocess("我我我 明天 去北京")
    assert text == "我 明天 去北京"          # 规则兜底
    assert meta["fallback"] == "rule"
    assert meta["reason"] == "error"


def test_mode_model_fallback_off_keeps_original(monkeypatch):
    monkeypatch.setattr(text_clean, "clean_model", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    text, meta = InputCleaner(mode="model", fallback="off").preprocess("我我我 明天")
    assert text == "我我我 明天"
    assert meta["fallback"] == "off"


def test_mode_model_uses_server_result(monkeypatch):
    monkeypatch.setattr(text_clean, "clean_model", lambda text, url, timeout: "清洗后")
    text, meta = InputCleaner(mode="model").preprocess("随便")
    assert text == "清洗后" and meta["applied"] is True


def test_too_long_input_is_skipped():
    long_text = "脏" * (settings.INPUT_CLEAN_MAX_CHARS + 1)
    text, meta = InputCleaner(mode="rule").preprocess(long_text)
    assert text == long_text and meta["reason"] == "too_long"


# ── 接入层：原文入库、清洗文本进模型 ────────────────────────────────────
CAPTURED: dict = {}


def _fake_chat(self, **kwargs):
    CAPTURED["query"] = kwargs.get("query")
    return {
        "answer": "收到",
        "reasoning": "",
        "model_provider": "ollama",
        "model_name": settings.OLLAMA_MODEL,
        "total_tokens": 8,
    }


def test_chat_cleans_query_but_stores_original(client, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODE", "rule")
    monkeypatch.setattr(LLMService, "chat", _fake_chat)
    monkeypatch.setattr(RAGService, "search", lambda self, db, query, use_rerank=None, timing=None: [])

    raw = "我我我 明天  去北京！！！"
    resp = client.post("/api/chat", json={"query": raw}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()

    # 模型看到的是清洗后的文本
    assert CAPTURED["query"] == "我 明天 去北京！"
    # 响应里带清洗元信息
    assert body["clean"]["mode"] == "rule" and body["clean"]["changed"] is True
    assert body["clean"]["clean_query"] == "我 明天 去北京！"

    # 日志里存的是用户原文，不是清洗结果
    item = client.get("/api/logs?page_size=1", headers=auth_headers).json()["items"][0]
    assert item["user_input"] == raw


def test_chat_respects_per_request_switch(client, auth_headers, monkeypatch):
    """服务端开着清洗时，单次请求 use_clean=false 应当绕过。"""
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODE", "rule")
    monkeypatch.setattr(LLMService, "chat", _fake_chat)
    monkeypatch.setattr(RAGService, "search", lambda self, db, query, use_rerank=None, timing=None: [])

    raw = "我我我 明天  去北京"
    client.post("/api/chat", json={"query": raw, "use_clean": False}, headers=auth_headers)
    assert CAPTURED["query"] == raw


def test_options_exposes_clean_config(client, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODE", "rule")
    body = client.get("/api/chat/options", headers=auth_headers).json()
    assert body["clean_mode"] == "rule" and body["clean_enabled"] is True
