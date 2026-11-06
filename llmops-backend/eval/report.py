# -*- coding: utf-8 -*-
"""报告输出与回归门禁：JSON 存档、Markdown 表格、指标下降判定。"""
from __future__ import annotations

import json
from pathlib import Path

COLUMNS = [
    ("mode", "模式"),
    ("recall_at_candidate_pct", "Recall@K(%)"),
    ("mrr", "MRR"),
    ("hit_at_feed_pct", "命中率(%)"),
    ("coverage_pct", "覆盖率(%)"),
    ("false_accept_pct", "误放行(%)"),
    ("latency_p50_ms", "P50(ms)"),
    ("latency_p95_ms", "P95(ms)"),
]


def to_markdown(summaries: dict, meta: dict) -> str:
    lines = [
        "# RAG 检索评测报告",
        "",
        f"- 评测集：{meta.get('set')}（{meta.get('questions')} 题，可答 {meta.get('answerable')} / 无答案 {meta.get('no_answer')}）",
        f"- 生成时间：{meta.get('generated_at')}",
        f"- 配置：candidate_k={meta.get('candidate_k')}，feed_topk={meta.get('feed_topk')}，"
        f"cosine 阈值={meta.get('threshold_cosine')}，精排阈值={meta.get('threshold_rerank')}",
        "",
        "| " + " | ".join(label for _, label in COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for mode, s in summaries.items():
        row = []
        for key, _ in COLUMNS:
            value = s.get(key, "")
            row.append(f"{value}" if not isinstance(value, float) else f"{value:g}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 结论", ""]
    lines.append(_conclusion(summaries))
    lines.append("")
    lines.append("> 指标口径：Recall@K = 正确文档是否进入 K 条候选；MRR = 正确文档首次出现名次的倒数均值；"
                 "命中率 = 正确答案是否出现在最终喂给模型的片段里；误放行 = 无答案问题仍返回片段的占比。")
    return "\n".join(lines)


def _conclusion(summaries: dict) -> str:
    if not summaries:
        return "（无结果）"
    parts = []
    dense = summaries.get("dense")
    rerank = summaries.get("rerank")
    hybrid = summaries.get("hybrid")
    if dense and rerank:
        parts.append(
            f"- 精排把无答案误放行从 {dense['false_accept_pct']}% 压到 {rerank['false_accept_pct']}%，"
            f"代价是命中率 {dense['hit_at_feed_pct']}% → {rerank['hit_at_feed_pct']}%，"
            f"P95 延迟 {dense['latency_p95_ms']}ms → {rerank['latency_p95_ms']}ms。"
        )
    if dense and hybrid:
        delta = hybrid["hit_at_feed_pct"] - dense["hit_at_feed_pct"]
        parts.append(
            f"- 混合检索相对纯向量：命中率 {dense['hit_at_feed_pct']}% → {hybrid['hit_at_feed_pct']}%"
            f"（{delta:+.1f} 个点），Recall@K {dense['recall_at_candidate_pct']}% → {hybrid['recall_at_candidate_pct']}%。"
        )
    return "\n".join(parts) if parts else "（只有单一模式，未做对比结论）"


def write_outputs(out_dir: Path, summary: dict, per_query: dict, meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "results": summary}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    lines = ["mode,id,type,question,answerable,rank,hit,returned,top1_score,latency_ms,candidate_ms,finalize_ms"]
    for mode, records in per_query.items():
        for r in records:
            q = str(r["question"]).replace('"', "'").replace(",", "，")
            lines.append(
                f'{mode},{r["id"]},{r["type"]},"{q}",{int(r["answerable"])},{r["rank"] or ""},'
                f'{int(r["hit"])},{int(r["returned"])},{r["top1_score"]},{r["latency_ms"]},'
                f'{r["candidate_ms"]},{r["finalize_ms"]}'
            )
    (out_dir / "per_query.csv").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "report.md").write_text(to_markdown(summary, meta), encoding="utf-8")
    return {"dir": str(out_dir), "files": ["summary.json", "per_query.csv", "report.md"]}


def gate(baseline: dict, current: dict, max_hit_drop: float = 2.0, max_fa_rise: float = 5.0) -> tuple[bool, list[str]]:
    """回归门禁：命中率跌幅、误放行涨幅超过阈值就判失败。"""
    problems: list[str] = []
    for mode, cur in current.items():
        base = baseline.get(mode)
        if not base:
            continue
        hit_drop = base["hit_at_feed_pct"] - cur["hit_at_feed_pct"]
        fa_rise = cur["false_accept_pct"] - base["false_accept_pct"]
        if hit_drop > max_hit_drop:
            problems.append(
                f"[{mode}] 命中率下降 {hit_drop:.1f} 个点（{base['hit_at_feed_pct']}% → {cur['hit_at_feed_pct']}%，"
                f"阈值 {max_hit_drop}）"
            )
        if fa_rise > max_fa_rise:
            problems.append(
                f"[{mode}] 误放行上升 {fa_rise:.1f} 个点（{base['false_accept_pct']}% → {cur['false_accept_pct']}%，"
                f"阈值 {max_fa_rise}）"
            )
    return (not problems, problems)
