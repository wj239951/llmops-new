# -*- coding: utf-8 -*-
"""多轮检索查询改写（app/services/query_rewrite.py）单元测试。"""
from app.core.config import settings
from app.services import query_rewrite


HISTORY = [{"role": "user", "content": "注塑机 ZX-1200 的锁模力是多少？"},
           {"role": "assistant", "content": "ZX-1200 的锁模力是 1200 千牛。"}]


def _mode(monkeypatch, mode):
    monkeypatch.setattr(settings, "RAG_QUERY_REWRITE_MODE", mode)


def test_off_mode_returns_query_unchanged(monkeypatch):
    _mode(monkeypatch, "off")
    q, meta = query_rewrite.rewrite("那它的最大注射量呢？", HISTORY)
    assert q == "那它的最大注射量呢？"
    assert meta["applied"] is False and meta["changed"] is False


def test_no_history_means_no_rewrite(monkeypatch):
    _mode(monkeypatch, "rule")
    q, meta = query_rewrite.rewrite("那它的最大注射量呢？", [])
    assert q == "那它的最大注射量呢？"
    assert meta["applied"] is False


def test_self_contained_query_is_not_rewritten(monkeypatch):
    _mode(monkeypatch, "rule")
    q, meta = query_rewrite.rewrite("ZX-2030 的液压油滤芯多久换一次？", HISTORY)
    assert q == "ZX-2030 的液压油滤芯多久换一次？"
    assert meta["applied"] is False


def test_rule_replaces_pronoun_with_last_model(monkeypatch):
    _mode(monkeypatch, "rule")
    q, meta = query_rewrite.rewrite("那它的最大注射量呢？", HISTORY)
    assert "zx-1200" in q.lower()
    assert "它" not in q
    assert meta["changed"] is True and meta["via"] == "rule"


def test_rule_prepends_subject_for_ellipsis(monkeypatch):
    _mode(monkeypatch, "rule")
    q, meta = query_rewrite.rewrite("液压油滤芯多久一次？", HISTORY)
    assert q.lower().startswith("zx-1200")
    assert "液压油滤芯" in q
    assert meta["changed"] is True


def test_llm_failure_falls_back_to_rule(monkeypatch):
    _mode(monkeypatch, "llm")

    def _boom(query, history):
        raise RuntimeError("model down")

    monkeypatch.setattr(query_rewrite, "_llm_rewrite", _boom)
    q, meta = query_rewrite.rewrite("那它的最大注射量呢？", HISTORY)
    assert meta["fallback"] == "rule"
    assert "zx-1200" in q.lower()
    assert meta["changed"] is True


def test_llm_dropping_exact_token_is_repaired(monkeypatch):
    _mode(monkeypatch, "llm")
    monkeypatch.setattr(query_rewrite, "_llm_rewrite",
                        lambda q, h: "传感器的更换周期是多少")
    q, meta = query_rewrite.rewrite("E-073 的更换周期是多少？它多久换一次",
                                    [{"role": "user", "content": "E-073 是什么故障"}])
    assert "e-073" in q.lower()
    assert meta["changed"] is True


def test_llm_mode_uses_model_output(monkeypatch):
    _mode(monkeypatch, "llm")
    monkeypatch.setattr(query_rewrite, "_llm_rewrite",
                        lambda q, h: "ZX-1200 的最大注射量是多少")
    q, meta = query_rewrite.rewrite("那它的最大注射量呢？", HISTORY)
    assert q == "ZX-1200 的最大注射量是多少"
    assert meta["via"] == "llm" and meta["changed"] is True

def test_long_ellipsis_without_model_is_rewritten(monkeypatch):
    """不带型号的省略句即使超过 12 字也要补主体——长度不是可靠信号。"""
    _mode(monkeypatch, "rule")
    q, meta = query_rewrite.rewrite("合模机构间隙检测多久一次？", HISTORY)
    assert q.lower().startswith("zx-1200")
    assert meta["changed"] is True