# -*- coding: utf-8 -*-
"""评测包自身的单测：指标口径、评测集校验、门禁判定。

不依赖 MySQL / Ollama / Chroma，可在 CI 里直接跑。
"""
import json

import pytest

from eval import golden, metrics, report


def _rec(id_, answerable, rank, hit, returned, score=0.0, latency=10):
    return {
        "id": id_, "type": "exact", "question": "q", "answerable": answerable,
        "rank": rank, "hit": hit, "returned": returned, "n_candidates": 1, "n_final": 1,
        "top1_score": score, "top1_doc_id": 1, "match": "vector",
        "latency_ms": latency, "candidate_ms": latency, "finalize_ms": 0,
    }


def test_first_hit_rank_prefers_first_occurrence():
    cands = [{"doc_id": 9}, {"doc_id": 46}, {"doc_id": 46}]
    assert metrics.first_hit_rank(cands, {46}) == 2
    assert metrics.first_hit_rank(cands, {1}) is None


def test_summarize_mode_mixed_records():
    records = [
        _rec("a", True, 1, True, True, 0.8, 10),
        _rec("b", True, None, False, False, 0.0, 30),
        _rec("c", False, None, False, True, 0.6, 20),
    ]
    s = metrics.summarize_mode(records, "dense")
    assert s["answerable"] == 2 and s["no_answer"] == 1
    assert s["recall_at_candidate_pct"] == 50.0
    assert s["mrr"] == 0.5
    assert s["hit_at_feed_pct"] == 50.0
    assert s["false_accept_pct"] == 100.0
    assert s["false_accept_ids"] == ["c"]
    assert s["missed_ids"] == ["b"]
    assert s["latency_p50_ms"] == 20.0


def test_percentile_interpolates():
    assert metrics.percentile([], 95) == 0.0
    assert metrics.percentile([10], 95) == 10.0
    assert metrics.percentile([10, 20, 30, 40], 50) == 25.0
    assert metrics.percentile([10, 20], 0) == 10.0
    assert metrics.percentile([10, 20], 100) == 20.0


def test_golden_set_loads_and_is_balanced():
    questions = golden.load_set("qa")
    assert len(questions) == 45
    stats = golden.stats(questions)
    assert stats["answerable"] == 37
    assert stats["no_answer"] == 8
    assert stats["by_type"]["exact"] == 21


def test_golden_rejects_bad_records(tmp_path_factory):
    # 不用 pytest 默认 tmp（在系统 TEMP，CI 之外可能不可写），落到包目录下的临时文件
    from pathlib import Path

    bad = Path(__file__).with_name("_tmp_bad.jsonl")
    bad.write_text(json.dumps({"id": "x", "question": "q"}), encoding="utf-8")
    try:
        with pytest.raises(golden.GoldenError, match="缺少字段"):
            golden.load_set("qa", path=bad)
    finally:
        bad.unlink(missing_ok=True)


def test_gate_flags_regression():
    base = {"dense": {"hit_at_feed_pct": 100.0, "false_accept_pct": 10.0}}
    cur = {"dense": {"hit_at_feed_pct": 95.0, "false_accept_pct": 20.0}}
    ok, problems = report.gate(base, cur)
    assert ok is False
    assert len(problems) == 2


def test_gate_passes_within_tolerance():
    base = {"dense": {"hit_at_feed_pct": 100.0, "false_accept_pct": 10.0}}
    cur = {"dense": {"hit_at_feed_pct": 99.0, "false_accept_pct": 12.0}}
    ok, problems = report.gate(base, cur)
    assert ok is True and problems == []


def test_markdown_contains_all_modes():
    summary = {
        "dense": metrics.summarize_mode([_rec("a", True, 1, True, True, 0.8)], "dense"),
        "rerank": metrics.summarize_mode([_rec("a", True, 1, True, True, 0.9)], "rerank"),
    }
    md = report.to_markdown(summary, {"set": "qa", "questions": 1, "answerable": 1, "no_answer": 0})
    assert "| 模式 |" in md
    assert "rerank" in md and "dense" in md
    assert "精排把无答案误放行" in md
