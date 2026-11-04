# -*- coding: utf-8 -*-
"""RAG 检索质量评测与阈值 / 弱命中余量标定脚本。

不启动服务、不调用大模型，只测「检索层」：
    问题 -> bge-m3 向量检索 -> 候选片段（含 cosine 分数）
         -> 阈值过滤 -> 最终喂给模型的 top-k 片段

指标定义（与 app/services/rag_service.py、app/services/chat_service.py 的实现一致）：
- recall@K      ：正确文档是否出现在 K 条原始候选里（对应 RAG_CANDIDATE_K）
- hit@feed_topk ：按阈值过滤后，正确文档是否真的进了最终喂给模型的片段（对应 RAG_FEED_TOPK）
- top1 准确率   ：最终喂给模型的第一条片段是否来自正确文档
- MRR           ：正确文档在原始候选里的排名倒数（同篇多 chunk 只算一次，取首次出现）
- 误接受率      ：无答案问题在阈值过滤后仍然喂了片段给模型的比例（越低越好）

用法（在 llmops-backend 目录执行）：
    .venv\\Scripts\\python.exe tools\\evaluate_rag.py
    .venv\\Scripts\\python.exe tools\\evaluate_rag.py --threshold-scan 0.30,0.80,0.01

输出（默认 experiments/results/）：
    per_query.csv          每条问题的检索明细
    threshold_sweep.csv    阈值扫描
    margin_sweep.csv       弱命中余量扫描（基于当前阈值）
    summary.json           汇总指标与推荐值
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.services.rag_service import vectorstore  # noqa: E402


NO_ANSWER_TYPES = {"no_answer", "no_answer_far", "no_answer_near"}


def display_path(path: Path) -> str:
    """summary.json 里只记录仓库内相对路径，避免提交本机绝对路径。"""
    try:
        return path.resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- 基础工具
def load_questions(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} JSON 解析失败: {exc}") from exc
            if "question" not in item or "expected_doc_ids" not in item:
                raise SystemExit(f"{path}:{line_no} 缺少 question / expected_doc_ids 字段")
            item.setdefault("type", "unknown")
            items.append(item)
    return items


def retrieve(query: str, k: int) -> list[dict]:
    """直接查向量库取原始候选，保留 distance -> cosine 的换算。"""
    results = vectorstore.similarity_search_with_score(query, k=k)
    cands = [
        {
            "doc_id": r.metadata.get("id"),
            "title": r.metadata.get("title", ""),
            "chunk": r.metadata.get("chunk", ""),
            "score": 1.0 - float(dist),
            "content": r.page_content,
        }
        for r, dist in results
    ]
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def first_doc_rank(cands: list[dict], expected: set[int]) -> int | None:
    """正确文档在候选里的排名（doc 级去重，同篇多 chunk 只算首次出现）。"""
    seen: set[int] = set()
    rank = 0
    for cand in cands:
        doc_id = cand["doc_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        if doc_id in expected:
            return rank
    return None


def fed_chunks(cands: list[dict], threshold: float, feed_topk: int) -> list[dict]:
    """复刻 RAGService.search：先按分数过滤，再取前 feed_topk 条 chunk。"""
    return [c for c in cands if c["score"] >= threshold][:feed_topk]


def scan_float_range(spec: str) -> list[float]:
    parts = spec.split(",")
    if len(parts) != 3:
        raise SystemExit(f"范围参数格式应为 start,end,step：{spec}")
    start, end, step = (float(p) for p in parts)
    if step <= 0:
        raise SystemExit("step 必须大于 0")
    values: list[float] = []
    value = start
    while value <= end + 1e-9:
        values.append(round(value, 4))
        value += step
    return values


def pct(n: float, d: float) -> float:
    return (n / d * 100.0) if d else 0.0


def fmt_pct(n: float, d: float) -> str:
    return f"{pct(n, d):5.1f}%"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评测与标定")
    parser.add_argument("--questions", default=str(REPO_DIR / "experiments" / "eval_qa.jsonl"))
    parser.add_argument("--output-dir", default=str(REPO_DIR / "experiments" / "results"))
    parser.add_argument("--candidate-k", type=int, default=settings.RAG_CANDIDATE_K,
                        help="原始候选数量，默认取 RAG_CANDIDATE_K")
    parser.add_argument("--feed-topk", type=int, default=settings.RAG_FEED_TOPK,
                        help="最终喂给模型的片段数，默认取 RAG_FEED_TOPK")
    parser.add_argument("--threshold", type=float, default=settings.RAG_SCORE_THRESHOLD,
                        help="基线阈值，默认取 RAG_SCORE_THRESHOLD")
    parser.add_argument("--margin", type=float, default=settings.RAG_WEAK_MARGIN,
                        help="基线弱命中余量，默认取 RAG_WEAK_MARGIN")
    parser.add_argument("--threshold-scan", default="0.30,0.80,0.01")
    parser.add_argument("--margin-scan", default="0.00,0.15,0.01")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    questions = load_questions(questions_path)
    answerable = [q for q in questions if q["type"] not in NO_ANSWER_TYPES]
    no_answer = [q for q in questions if q["type"] in NO_ANSWER_TYPES]

    print("=" * 78)
    print("RAG 检索评测")
    print(f"  题库      : {display_path(questions_path)}")
    print(f"  问题数    : {len(questions)}（可回答 {len(answerable)} / 无答案 {len(no_answer)}）")
    print(f"  向量集合  : {vectorstore._collection.name}，共 {vectorstore._collection.count()} 条向量")
    print(f"  候选 K    : {args.candidate_k}   喂给模型 top-k: {args.feed_topk}")
    print(f"  当前阈值  : {args.threshold}   当前弱命中余量: {args.margin}")
    print("=" * 78)

    # 可选：从 MySQL 拿 id -> 标题映射，方便报表阅读；数据库不可用不影响指标
    title_by_id: dict[int, str] = {}
    try:
        from app.core.database import SessionLocal
        from app.models.knowledge import KnowledgeDocument

        db = SessionLocal()
        try:
            title_by_id = {d.id: d.title for d in db.query(KnowledgeDocument).all()}
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 - 报表增强项，失败只警告
        print(f"[警告] 读取文档标题失败（不影响指标计算）：{exc}")

    rows: list[dict] = []
    cands_by_q: list[list[dict]] = []
    started = time.time()

    for idx, q in enumerate(questions, 1):
        expected = {int(d) for d in (q.get("expected_doc_ids") or [])}
        cands = retrieve(q["question"], k=args.candidate_k)
        cands_by_q.append(cands)
        chunks = fed_chunks(cands, args.threshold, args.feed_topk)
        top1 = chunks[0] if chunks else None
        is_no_answer = q["type"] in NO_ANSWER_TYPES

        rows.append({
            "id": q.get("id", f"q{idx:03d}"),
            "type": q["type"],
            "question": q["question"],
            "expected_doc_ids": ";".join(str(d) for d in sorted(expected)),
            "candidate_count": len(cands),
            "top1_doc_id": cands[0]["doc_id"] if cands else "",
            "top1_score": round(cands[0]["score"], 4) if cands else "",
            "top1_title": (title_by_id.get(cands[0]["doc_id"], cands[0]["title"]) if cands else ""),
            "expected_rank": first_doc_rank(cands, expected) if expected else "",
            "recall_at_k": int(first_doc_rank(cands, expected) is not None) if expected else "",
            "fed_count": len(chunks),
            "fed_doc_ids": ";".join(str(c["doc_id"]) for c in chunks),
            "hit_at_feed": int(any(c["doc_id"] in expected for c in chunks)) if expected else "",
            "top1_correct": ("" if is_no_answer else int(bool(top1 and top1["doc_id"] in expected))),
            "no_answer": int(is_no_answer),
            "no_answer_accepted": (int(bool(chunks)) if is_no_answer else ""),
        })

        flag = ""
        if is_no_answer:
            flag = "误接受!" if chunks else "正确拒答"
        elif expected:
            rank = rows[-1]["expected_rank"]
            flag = "hit" if rows[-1]["hit_at_feed"] else ("未命中" if rank is not None else "未召回")
        print(f"  [{idx:02d}/{len(questions)}] {str(rows[-1]['id']):<4} {q['type']:<15} "
              f"top1={rows[-1]['top1_score']!s:<8} rank={rows[-1]['expected_rank']!s:<3} "
              f"{flag:<6} {q['question'][:32]}")

    elapsed = time.time() - started

    # ---------- 基线指标（当前阈值） ----------
    rec_hits = 0
    mrr_values: list[float] = []
    fed_ans = 0
    hit_ans = 0
    top1_ans = 0
    for q, row in zip(questions, rows):
        if q["type"] in NO_ANSWER_TYPES:
            continue
        if row["recall_at_k"] == 1:
            rec_hits += 1
        if row["expected_rank"] != "":
            mrr_values.append(1.0 / int(row["expected_rank"]))
        if row["fed_count"] > 0:
            fed_ans += 1
        if row["hit_at_feed"] == 1:
            hit_ans += 1
        if row["top1_correct"] == 1:
            top1_ans += 1

    far = [r for r in rows if r["type"] == "no_answer_far"]
    near = [r for r in rows if r["type"] == "no_answer_near"]
    fa_far = sum(int(r["no_answer_accepted"] or 0) for r in far)
    fa_near = sum(int(r["no_answer_accepted"] or 0) for r in near)
    fa_all = fa_far + fa_near
    no_ans_count = len(no_answer)

    # 正确命中的 top1 分数下界 vs 无答案问题的 top1 分数上界：阈值可分离性
    answerable_rows = [(q, cands, row) for q, cands, row in zip(questions, cands_by_q, rows)
                       if q["type"] not in NO_ANSWER_TYPES]
    correct_top1 = [float(cands[0]["score"]) for _, cands, row in answerable_rows
                    if cands and row["top1_correct"] == 1]
    ans_top1 = [float(cands[0]["score"]) for _, cands, _ in answerable_rows if cands]
    noans_top1 = [float(cands[0]["score"]) for q, cands in zip(questions, cands_by_q)
                  if q["type"] in NO_ANSWER_TYPES and cands]

    # ---------- 阈值扫描 ----------
    thresholds = scan_float_range(args.threshold_scan)
    sweep: list[dict] = []
    for t in thresholds:
        t_fed = t_hit = t_top1 = 0
        for q, cands in zip(answerable, _slice_cands(cands_by_q, questions, NO_ANSWER_TYPES)):
            chunks = fed_chunks(cands, t, args.feed_topk)
            expected = {int(d) for d in (q.get("expected_doc_ids") or [])}
            if chunks:
                t_fed += 1
                if chunks[0]["doc_id"] in expected:
                    t_top1 += 1
                if any(c["doc_id"] in expected for c in chunks):
                    t_hit += 1
        t_fa = t_fa_far = t_fa_near = 0
        for q, cands in zip(questions, cands_by_q):
            if q["type"] not in NO_ANSWER_TYPES:
                continue
            if fed_chunks(cands, t, args.feed_topk):
                t_fa += 1
                if q["type"] == "no_answer_far":
                    t_fa_far += 1
                else:
                    t_fa_near += 1
        balanced = 0.5 * (t_hit / len(answerable) if answerable else 0.0) + \
                   0.5 * (1.0 - t_fa / no_ans_count if no_ans_count else 0.0)
        sweep.append({
            "threshold": t,
            "coverage_pct": round(pct(t_fed, len(answerable)), 2),
            "hit_at_feed_pct": round(pct(t_hit, len(answerable)), 2),
            "top1_accuracy_pct": round(pct(t_top1, len(answerable)), 2),
            "false_accept_pct": round(pct(t_fa, no_ans_count), 2),
            "balanced_accuracy_pct": round(balanced * 100, 2),
        })
        sweep[-1]["false_accept_far_pct"] = round(pct(t_fa_far, len(far)), 2)
        sweep[-1]["false_accept_near_pct"] = round(pct(t_fa_near, len(near)), 2)

    best = max(sweep, key=lambda r: (r["balanced_accuracy_pct"], -r["threshold"]))
    stable = [r["threshold"] for r in sweep
              if r["hit_at_feed_pct"] >= 85.0 and r["false_accept_pct"] <= 20.0]
    max_full_recall = max((r["threshold"] for r in sweep if r["coverage_pct"] >= 100.0), default=None)

    # ---------- 弱命中余量扫描（基于当前阈值） ----------
    margins = scan_float_range(args.margin_scan)
    fed_records: list[dict] = []
    for q, cands in zip(answerable, _slice_cands(cands_by_q, questions, NO_ANSWER_TYPES)):
        chunks = fed_chunks(cands, args.threshold, args.feed_topk)
        if not chunks:
            continue
        expected = {int(d) for d in (q.get("expected_doc_ids") or [])}
        fed_records.append({
            "id": q.get("id"),
            "score": float(chunks[0]["score"]),
            "correct": chunks[0]["doc_id"] in expected,
        })
    wrong_scores = [r["score"] for r in fed_records if not r["correct"]]
    correct_scores = [r["score"] for r in fed_records if r["correct"]]
    margin_rows: list[dict] = []
    for m in margins:
        weak = [r for r in fed_records if r["score"] < args.threshold + m]
        hit_band = [r for r in fed_records if r["score"] >= args.threshold + m]
        weak_err = sum(1 for r in weak if not r["correct"])
        hit_err = sum(1 for r in hit_band if not r["correct"])
        wrong_total = len(wrong_scores)
        correct_total = len(correct_scores)
        capture = (sum(1 for r in weak if not r["correct"]) / wrong_total) if wrong_total else 0.0
        false_weak = (sum(1 for r in weak if r["correct"]) / correct_total) if correct_total else 0.0
        margin_rows.append({
            "margin": m,
            "hit_band_pct": round(pct(len(hit_band), len(fed_records)), 2),
            "weak_band_pct": round(pct(len(weak), len(fed_records)), 2),
            "err_in_hit_band_pct": round(pct(hit_err, len(hit_band)), 2),
            "err_in_weak_band_pct": round(pct(weak_err, len(weak)), 2),
            "error_capture_pct": round(capture * 100, 2),
            "correct_downgraded_pct": round(false_weak * 100, 2),
            "youden_j_pct": round((capture - false_weak) * 100, 2),
        })
    best_margin = max(margin_rows, key=lambda r: (r["youden_j_pct"], -r["margin"]))
    margin_all_wrong = (max(wrong_scores) - args.threshold + 0.01) if wrong_scores else None
    current_margin_row = min(margin_rows, key=lambda r: abs(r["margin"] - args.margin))

    # ---------- 导出 ----------
    per_query_fields = list(rows[0].keys())
    write_csv(output_dir / "per_query.csv", rows, per_query_fields)
    write_csv(output_dir / "threshold_sweep.csv", sweep, list(sweep[0].keys()))
    write_csv(output_dir / "margin_sweep.csv", margin_rows, list(margin_rows[0].keys()))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 2),
        "questions": {
            "path": display_path(questions_path),
            "total": len(questions),
            "answerable": len(answerable),
            "no_answer": no_ans_count,
            "by_type": dict(Counter(q["type"] for q in questions)),
        },
        "collection": {
            "name": vectorstore._collection.name,
            "vectors": vectorstore._collection.count(),
        },
        "config": {
            "threshold": args.threshold,
            "margin": args.margin,
            "candidate_k": args.candidate_k,
            "feed_topk": args.feed_topk,
        },
        "baseline": {
            "recall_at_candidate_k_pct": round(pct(rec_hits, len(answerable)), 2),
            "mrr": round(mean(mrr_values), 4) if mrr_values else 0.0,
            "coverage_pct": round(pct(fed_ans, len(answerable)), 2),
            "hit_at_feed_pct": round(pct(hit_ans, len(answerable)), 2),
            "top1_accuracy_pct": round(pct(top1_ans, len(answerable)), 2),
            "top1_accuracy_among_fed_pct": round(pct(top1_ans, fed_ans), 2),
            "false_accept_far_pct": round(pct(fa_far, len(far)), 2),
            "false_accept_near_pct": round(pct(fa_near, len(near)), 2),
            "false_accept_all_pct": round(pct(fa_all, no_ans_count), 2),
            "missed_answerable_ids": [r["id"] for r in rows
                                      if not r["no_answer"] and r["hit_at_feed"] == 0],
            "false_accepted_ids": [r["id"] for r in rows
                                   if r["no_answer"] and r["no_answer_accepted"] == 1],
        },
        "score_separation": {
            "answerable_top1_min": round(min(ans_top1), 4) if ans_top1 else None,
            "answerable_top1_median": round(median(ans_top1), 4) if ans_top1 else None,
            "answerable_top1_mean": round(mean(ans_top1), 4) if ans_top1 else None,
            "correct_hit_top1_min": round(min(correct_top1), 4) if correct_top1 else None,
            "no_answer_top1_max": round(max(noans_top1), 4) if noans_top1 else None,
            "no_answer_top1_mean": round(mean(noans_top1), 4) if noans_top1 else None,
        },
        "recommendations": {
            "threshold": best["threshold"],
            "threshold_balanced_accuracy_pct": best["balanced_accuracy_pct"],
            "stable_threshold_range": [stable[0], stable[-1]] if stable else None,
            "max_threshold_keep_full_recall": max_full_recall,
            "margin": best_margin["margin"] if wrong_scores else None,
            "margin_youden_j_pct": best_margin["youden_j_pct"] if wrong_scores else None,
            "margin_to_capture_all_wrong": round(margin_all_wrong, 3) if wrong_scores else None,
            "margin_data_available": bool(wrong_scores),
            "correct_downgraded_at_current_margin_pct": current_margin_row["correct_downgraded_pct"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 控制台摘要 ----------
    print("\n" + "=" * 78)
    print("基线指标（当前阈值 %.2f）" % args.threshold)
    print(f"  recall@{args.candidate_k}          : {fmt_pct(rec_hits, len(answerable))}  "
          f"({rec_hits}/{len(answerable)})")
    print(f"  MRR                    : {summary['baseline']['mrr']:.4f}")
    print(f"  覆盖（有片段可喂）     : {fmt_pct(fed_ans, len(answerable))}  "
          f"({fed_ans}/{len(answerable)})")
    print(f"  hit@{args.feed_topk}（正确片段进上下文）: {fmt_pct(hit_ans, len(answerable))}  "
          f"({hit_ans}/{len(answerable)})")
    print(f"  top1 准确率            : {fmt_pct(top1_ans, len(answerable))}  "
          f"（已喂片段中 {fmt_pct(top1_ans, fed_ans)}）")
    print(f"  无答案误接受（far/near/all）: {fmt_pct(fa_far, len(far))} / "
          f"{fmt_pct(fa_near, len(near))} / {fmt_pct(fa_all, no_ans_count)}")
    print("\n分数可分离性")
    print(f"  正确命中 top1 最低分   : {summary['score_separation']['correct_hit_top1_min']}")
    print(f"  可回答 top1 中位数     : {summary['score_separation']['answerable_top1_median']}")
    print(f"  无答案 top1 最高分     : {summary['score_separation']['no_answer_top1_max']}")
    print("\n推荐值")
    print(f"  阈值   : {best['threshold']}（balanced accuracy {best['balanced_accuracy_pct']}%）")
    print(f"  稳定区间: {summary['recommendations']['stable_threshold_range']}")
    print(f"  不丢正确片段的最大阈值: {max_full_recall}")
    if wrong_scores:
        print(f"  弱命中余量: {best_margin['margin']}（Youden J {best_margin['youden_j_pct']}%）")
        print(f"  覆盖全部错误的最小余量: {round(margin_all_wrong, 3)}")
    else:
        print("  弱命中余量: 题库中无错误 top1 样本，余量无法标定；"
              f"当前 {args.margin} 会把 {current_margin_row['correct_downgraded_pct']}% "
              "的正确命中降级为 weak_hit")
    print(f"\n结果文件: {output_dir}")
    print(f"用时 {elapsed:.1f}s")
    print("=" * 78)
    return 0


def _slice_cands(cands_by_q: list[list[dict]], questions: list[dict], exclude_types: set[str] | None):
    """按类型过滤候选列表，保持与 questions 同序。"""
    if exclude_types is None:
        return [c for q, c in zip(questions, cands_by_q) if q["type"] in NO_ANSWER_TYPES]
    return [c for q, c in zip(questions, cands_by_q) if q["type"] not in exclude_types]


if __name__ == "__main__":
    sys.exit(main())
