# -*- coding: utf-8 -*-
"""评测公共指标：检索层（Recall@K / MRR / 命中率 / 误放行 / 延迟分位）。

定义统一放在这里，报告、门禁、单测都引用同一份实现，避免"文档一套、脚本一套"。
"""
from __future__ import annotations

import statistics
from typing import Iterable


def recall_at_k(rank: int | None) -> int:
    """正确文档是否出现在前 K 条候选里（rank 为 None 表示没进候选池）。"""
    return 1 if rank is not None else 0


def reciprocal_rank(rank: int | None) -> float:
    """MRR 的单条贡献；同篇文档多个 chunk 只取首次出现的名次。"""
    return 0.0 if rank is None else 1.0 / rank


def first_hit_rank(candidates: Iterable[dict], expected_ids: set[int]) -> int | None:
    """在候选列表里找第一个属于 expected 的片段，返回从 1 开始的名次。"""
    for idx, cand in enumerate(candidates, start=1):
        if cand.get("doc_id") in expected_ids:
            return idx
    return None


def percentile(values: list[float], p: float) -> float:
    """线性插值分位数（p 取 0–100），样本为空时返回 0。"""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    weight = k - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def pct(numerator: float, denominator: float) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def summarize_mode(records: list[dict], mode: str) -> dict:
    """把逐题记录聚合成一个模式的结果块。

    记录字段（runner 产出）：
      type / answerable / rank / hit / returned / latency_ms / top1_score
    """
    answerable = [r for r in records if r["answerable"]]
    no_answer = [r for r in records if not r["answerable"]]

    ranks = [r["rank"] for r in answerable]
    mrr_values = [reciprocal_rank(r) for r in ranks]
    latencies = [r["latency_ms"] for r in records]

    hit_answerable = sum(1 for r in answerable if r["hit"])
    covered = sum(1 for r in answerable if r["returned"])
    false_accept = sum(1 for r in no_answer if r["returned"])

    ans_scores = [r["top1_score"] for r in answerable if r["returned"]]
    noans_scores = [r["top1_score"] for r in no_answer if r["returned"]]

    return {
        "mode": mode,
        "total": len(records),
        "answerable": len(answerable),
        "no_answer": len(no_answer),
        "recall_at_candidate_pct": pct(sum(recall_at_k(r) for r in ranks), len(answerable)),
        "mrr": round(statistics.mean(mrr_values), 4) if mrr_values else 0.0,
        "hit_at_feed_pct": pct(hit_answerable, len(answerable)),
        "coverage_pct": pct(covered, len(answerable)),
        "false_accept_pct": pct(false_accept, len(no_answer)),
        "false_accept_ids": [r["id"] for r in no_answer if r["returned"]],
        "missed_ids": [r["id"] for r in answerable if not r["hit"]],
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "answerable_top1_min": round(min(ans_scores), 4) if ans_scores else 0.0,
        "no_answer_top1_max": round(max(noans_scores), 4) if noans_scores else 0.0,
    }
