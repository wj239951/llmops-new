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
    ("巡检周期  ，  是怎么规定的  ？", "巡检周期，是怎么规定的？"),   # 中文标点两侧空格
    ("工业 LLMOps 的核\n心能力包括哪些？", "工业 LLMOps 的核心能力包括哪些？"),  # 短文本折行要合并
    ("报错误码 Ｅ-\n073 时应先检查", "报错误码 E-073 时应先检查"),   # 型号被折开不能补空格
    ("合模压力低于多少少\n少少吨", "合模压力低于多少吨"),           # 跨行重复字先合并再压缩
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


# ── 请求级通道选择（use_clean_model）─────────────────────────────────────
# 语义：主链路按 INPUT_CLEAN_MODE 走规则清洗；只有前端显式点「精细清洗」才升级到模型通道。
def _stub_downstream(monkeypatch):
    monkeypatch.setattr(LLMService, "chat", _fake_chat)
    monkeypatch.setattr(RAGService, "search",
                        lambda self, db, query, use_rerank=None, timing=None: [])


def test_use_clean_model_upgrades_to_model_channel(client, auth_headers, monkeypatch):
    """默认规则清洗时，请求里点「精细清洗」应当把本轮升级到模型通道。"""
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODE", "rule")
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODEL_ALLOWED", True)
    _stub_downstream(monkeypatch)
    seen = []

    def _fake_model(text, url, timeout):
        seen.append(text)
        return "模型清洗结果"

    monkeypatch.setattr(text_clean, "clean_model", _fake_model)

    raw = "我我我 明天  去北京"
    body = client.post("/api/chat", json={"query": raw, "use_clean_model": True},
                       headers=auth_headers).json()

    # 交给模型的是原文（由模型自己判断要不要改），不是先规则清洗过的文本
    assert seen == [raw]
    assert body["clean"]["mode"] == "model"
    assert body["clean"]["requested_model"] is True
    assert CAPTURED["query"] == "模型清洗结果"


def test_use_clean_model_denied_falls_back_to_rule(client, auth_headers, monkeypatch):
    """服务端关掉模型通道时，请求里的 true 不报错，退回规则清洗并标出原因。"""
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODE", "rule")
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODEL_ALLOWED", False)
    _stub_downstream(monkeypatch)
    monkeypatch.setattr(text_clean, "clean_model",
                        lambda *a, **k: pytest.fail("模型通道不该被调用"))

    body = client.post("/api/chat",
                       json={"query": "我我我 明天  去北京", "use_clean_model": True},
                       headers=auth_headers).json()

    assert body["clean"]["mode"] == "rule"
    assert body["clean"]["fallback"] == "rule"
    assert body["clean"]["fallback_reason"] == "not_allowed"
    assert CAPTURED["query"] == "我 明天 去北京"


def test_use_clean_model_ignored_when_clean_disabled(client, auth_headers, monkeypatch):
    """use_clean=false 是更强的「本次不要清洗」，通道开关不能把它翻回来。"""
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODE", "rule")
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODEL_ALLOWED", True)
    _stub_downstream(monkeypatch)
    monkeypatch.setattr(text_clean, "clean_model",
                        lambda *a, **k: pytest.fail("清洗已关闭，不该调模型"))

    raw = "我我我 明天  去北京"
    body = client.post("/api/chat",
                       json={"query": raw, "use_clean": False, "use_clean_model": True},
                       headers=auth_headers).json()

    assert body["clean"]["mode"] == "off"
    assert body["clean"]["applied"] is False
    assert CAPTURED["query"] == raw


def test_use_clean_model_failure_records_error_reason(client, auth_headers, monkeypatch):
    """模型通道报错时按 INPUT_CLEAN_FALLBACK 回退规则，并把原因标成 error。"""
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODE", "rule")
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODEL_ALLOWED", True)
    _stub_downstream(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("clean server down")

    monkeypatch.setattr(text_clean, "clean_model", _boom)

    body = client.post("/api/chat",
                       json={"query": "我我我 明天  去北京", "use_clean_model": True},
                       headers=auth_headers).json()

    assert body["clean"]["mode"] == "model"          # 请求的是模型通道
    assert body["clean"]["fallback"] == "rule"        # 实际退回规则
    assert body["clean"]["fallback_reason"] == "error"
    assert body["clean"]["reason"] == "error"
    assert CAPTURED["query"] == "我 明天 去北京"


def test_clean_note_mentions_denied_channel():
    """工作流说明要能区分「模型失败」与「服务端不允许」。"""
    denied = {"mode": "rule", "applied": True, "changed": True, "ms": 1,
              "fallback": "rule", "fallback_reason": "not_allowed"}
    failed = {"mode": "model", "applied": True, "changed": True, "ms": 9,
              "fallback": "rule", "fallback_reason": "error"}
    assert "未允许模型通道" in ChatService._clean_note(denied)
    assert "模型通道失败" in ChatService._clean_note(failed)


def test_options_exposes_clean_model_allowed(client, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODE", "rule")
    monkeypatch.setattr(settings, "INPUT_CLEAN_MODEL_ALLOWED", False)
    body = client.get("/api/chat/options", headers=auth_headers).json()
    assert body["clean_model_allowed"] is False


# ── 词内分隔符修复（词典闸门）──

def test_merge_split_word_single_char_left():
    # 单字 + 词，拼起来是词典词 → 去分隔符
    assert clean_rule("注, 塑机, 怎么操, 作") == "注塑机，怎么操作"


def test_merge_split_word_keeps_enumeration():
    # 两侧都是词 / 都是单字 → 不动，避免把列举项拼成一个词
    assert clean_rule("苹果，香蕉") == "苹果，香蕉"
    assert clean_rule("一，二，三") == "一，二，三"
    assert clean_rule("我，你") == "我，你"


def test_merge_split_word_no_dict_hit():
    # 拼不成词 → 保留分隔符
    assert text_clean._merge_split_words("某某，啊啊") == "某某，啊啊"
