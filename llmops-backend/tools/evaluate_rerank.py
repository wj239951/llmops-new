# -*- coding: utf-8 -*-
"""RAG 第二阶段精排评测：bge-reranker-v2-m3 对 bge-m3 召回候选重排的效果与阈值标定。

对照关系（同一题库、同一批向量，可与 docs/04 的 cosine 基线直接比）：
- cosine 基线：bge-m3 召回 6 条 → cosine 阈值过滤 → top-3
- 精排两级：bge-m3 召回 K=12 条 → bge-reranker-v2-m3 成对重打分
           → 精排阈值过滤 → top-3

指标定义：
- recall@candidate_k：正确文档是否在精排前的原始候选里（召回上限）
- hit@feed_topk     ：精排阈值过滤后，正确文档是否进了最终 top-3
- top1 准确率       ：最终 top-1 片段是否来自正确文档
- MRR               ：精排后正确文档排名倒数（doc 级去重）
- 误接受率          ：无答案问题在精排阈值过滤后仍喂了片段的比例（越低越好）
- rerank_ms         ：单条查询精排耗时（不含向量检索与模型加载）

用法（在 llmops-backend 目录执行）：
    .venv\\Scripts\\python.exe tools\\evaluate_rerank.py --model-path D:\\models\\bge-reranker-v2-m3
    .venv\\Scripts\\python.exe tools\\evaluate_rerank.py --model-path ... --device cpu

输出（默认 experiments/results/）：
    rerank_per_query.csv
    rerank_threshold_sweep.csv
    rerank_margin_sweep.csv
    rerank_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.services.rerank_service import CrossEncoderReranker  # noqa: E402
from tools.evaluate_rag import (  # noqa: E402
    NO_ANSWER_TYPES,
    display_path,
    first_doc_rank,
    fmt_pct,
    load_questions,
    pct,
    retrieve,
    scan_float_range,
    write_csv,
)


def fed_chunks(cands: list[dict], threshold: float, feed_topk: int) -> list[dict]:
    """复刻 RAGService.search 的精排分支：按精排分数过滤，再取前 feed_topk 条。"""
    return [c for c in cands if c["rerank_score"] >= threshold][:feed_topk]


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 精排评测与阈值标定")
    parser.add_argument("--questions", default=str(REPO_DIR / "experiments" / "eval_qa.jsonl"))
    parser.add_argument("--output-dir", default=str(REPO_DIR / "experiments" / "results"))
    parser.add_argument("--model-path", default=settings.RAG_RERANK_MODEL_PATH or "",
                        help="本地权重目录；留空则按 RAG_RERANK_MODEL 从 HF 缓存/网络加载")
    parser.add_argument("--device", default="", help="cpu / cuda / auto（默认 auto）")
    parser.add_argument("--candidate-k", type=int, default=settings.RAG_RERANK_CANDIDATE_K,
                        help="精排前召回候选数，默认取 RAG_RERANK_CANDIDATE_K")
    parser.add_argument("--feed-topk", type=int, default=settings.RAG_FEED_TOPK,
                        help="最终喂给模型的片段数，默认取 RAG_FEED_TOPK")
    parser.add_argument("--threshold", type=float, default=settings.RAG_RERANK_SCORE_THRESHOLD,
                        help="精排分数阈值，默认取 RAG_RERANK_SCORE_THRESHOLD")
    parser.add_argument("--margin", type=float, default=settings.RAG_RERANK_WEAK_MARGIN,
                        help="精排弱命中余量，默认取 RAG_RERANK_WEAK_MARGIN")
    parser.add_argument("--threshold-scan", default="0.05,0.95,0.01")
    parser.add_argument("--margin-scan", default="0.00,0.30,0.01")
    parser.add_argument("--baseline-summary",
                        default=str(REPO_DIR / "experiments" / "results" / "summary.json"),
                        help="docs/04 的 cosine 基线 summary.json，用于对照展示")
    args = parser.parse_args()

    if args.device:
        settings.RAG_RERANK_DEVICE = args.device

    questions_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    questions = load_questions(questions_path)
    answerable = [q for q in questions if q["type"] not in NO_ANSWER_TYPES]
    no_answer = [q for q in questions if q["type"] in NO_ANSWER_TYPES]

    # ---------- 加载精排模型 ----------
    reranker = CrossEncoderReranker(args.model_path or None)
    load_started = time.time()
    loaded = reranker.load()
    load_seconds = time.time() - load_started
    if not loaded:
        print(f"[错误] 精排模型加载失败：{reranker.load_error}")
        return 1

    print("=" * 78)
    print("RAG 精排评测（bge-reranker 两级检索）")
    print(f"  题库      : {display_path(questions_path)}")
    print(f"  问题数    : {len(questions)}（可回答 {len(answerable)} / 无答案 {len(no_answer)}）")
    print(f"  精排模型  : {reranker._model_ref()}（device={reranker.device}，"
          f"加载 {load_seconds:.1f}s）")
    print(f"  召回 K    : {args.candidate_k}   最终 top-k: {args.feed_topk}")
    print(f"  精排阈值  : {args.threshold}   弱命中余量: {args.margin}")
    print("=" * 78)

    rows: list[dict] = []
    reranked_by_q: list[list[dict]] = []
    rerank_ms_list: list[float] = []

    for idx, q in enumerate(questions, 1):
        expected = {int(d) for d in (q.get("expected_doc_ids") or [])}
        cands = retrieve(q["question"], k=args.candidate_k)

        started = time.time()
        scores = reranker.score(q["question"], [c["content"] for c in cands])
        rerank_ms = (time.time() - started) * 1000
        rerank_ms_list.append(rerank_ms)

        reranked = [
            {**c, "rerank_score": float(s)} for c, s in zip(cands, scores)
        ]
        reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        reranked_by_q.append(reranked)

        chunks = fed_chunks(reranked, args.threshold, args.feed_topk)
        top1 = chunks[0] if chunks else None
        is_no_answer = q["type"] in NO_ANSWER_TYPES
        rank_after = first_doc_rank(reranked, expected) if expected else None

        rows.append({
            "id": q.get("id", f"q{idx:03d}"),
            "type": q["type"],
            "question": q["question"],
            "expected_doc_ids": ";".join(str(d) for d in sorted(expected)),
            "candidate_count": len(cands),
            "cosine_top1_doc_id": cands[0]["doc_id"] if cands else "",
            "cosine_top1_score": round(cands[0]["score"], 4) if cands else "",
            "rerank_top1_doc_id": reranked[0]["doc_id"] if reranked else "",
            "rerank_top1_score": round(reranked[0]["rerank_score"], 4) if reranked else "",
            "rerank_top1_title": (reranked[0]["title"] if reranked else ""),
            "expected_rank_after_rerank": rank_after if rank_after is not None else "",
            "candidate_recall": (int(rank_after is not None) if expected else ""),
            "fed_count": len(chunks),
            "fed_doc_ids": ";".join(str(c["doc_id"]) for c in chunks),
            "hit_at_feed": int(any(c["doc_id"] in expected for c in chunks)) if expected else "",
            "top1_correct": ("" if is_no_answer else int(bool(top1 and top1["doc_id"] in expected))),
            "no_answer": int(is_no_answer),
            "no_answer_accepted": (int(bool(chunks)) if is_no_answer else ""),
            "rerank_ms": round(rerank_ms, 1),
        })

        flag = ""
        if is_no_answer:
            flag = "误接受!" if chunks else "正确拒答"
        elif expected:
            flag = "hit" if rows[-1]["hit_at_feed"] else "未命中"
        print(f"  [{idx:02d}/{len(questions)}] {str(rows[-1]['id']):<4} {q['type']:<15} "
              f"cos={rows[-1]['cosine_top1_score']!s:<8} "
              f"rerank={rows[-1]['rerank_top1_score']!s:<8} "
              f"{flag:<6} {q['question'][:30]}")

    elapsed = time.time() - load_started

    # ---------- 当前阈值下的指标 ----------
    rec_hits = 0
    mrr_values: list[float] = []
    fed_ans = hit_ans = top1_ans = 0
    for q, row, reranked in zip(questions, rows, reranked_by_q):
        if q["type"] in NO_ANSWER_TYPES:
            continue
        if row["candidate_recall"] == 1:
            rec_hits += 1
        if row["expected_rank_after_rerank"] != "":
            mrr_values.append(1.0 / int(row["expected_rank_after_rerank"]))
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

    # 分数可分离性：正确 top1 的下界 vs 无答案 top1 的上界（精排分数尺度）
    ans_reranked = [c for q, c in zip(questions, reranked_by_q) if q["type"] not in NO_ANSWER_TYPES]
    correct_top1 = [float(c[0]["rerank_score"]) for q, c in zip(questions, reranked_by_q)
                    if q["type"] not in NO_ANSWER_TYPES and c and
                    c[0]["doc_id"] in {int(d) for d in (q.get("expected_doc_ids") or [])}]
    ans_top1 = [float(c[0]["rerank_score"]) for c in ans_reranked if c]
    noans_top1 = [float(c[0]["rerank_score"]) for q, c in zip(questions, reranked_by_q)
                  if q["type"] in NO_ANSWER_TYPES and c]

    # ---------- 精排阈值扫描 ----------
    thresholds = scan_float_range(args.threshold_scan)
    sweep: list[dict] = []
    for t in thresholds:
        t_fed = t_hit = t_top1 = 0
        for q, reranked in zip(answerable, [c for q, c in zip(questions, reranked_by_q)
                                            if q["type"] not in NO_ANSWER_TYPES]):
            chunks = fed_chunks(reranked, t, args.feed_topk)
            expected = {int(d) for d in (q.get("expected_doc_ids") or [])}
            if chunks:
                t_fed += 1
                if chunks[0]["doc_id"] in expected:
                    t_top1 += 1
                if any(c["doc_id"] in expected for c in chunks):
                    t_hit += 1
        t_fa = t_fa_far = t_fa_near = 0
        for q, reranked in zip(questions, reranked_by_q):
            if q["type"] not in NO_ANSWER_TYPES:
                continue
            if fed_chunks(reranked, t, args.feed_topk):
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
            "false_accept_far_pct": round(pct(t_fa_far, len(far)), 2),
            "false_accept_near_pct": round(pct(t_fa_near, len(near)), 2),
        })

    best = max(sweep, key=lambda r: (r["balanced_accuracy_pct"], -r["threshold"]))
    stable = [r["threshold"] for r in sweep
              if r["hit_at_feed_pct"] >= 85.0 and r["false_accept_pct"] <= 20.0]
    max_full_recall = max((r["threshold"] for r in sweep if r["coverage_pct"] >= 100.0), default=None)

    # ---------- 精排余量扫描（基于当前阈值） ----------
    margins = scan_float_range(args.margin_scan)
    fed_records: list[dict] = []
    for q, reranked in zip(answerable, [c for q, c in zip(questions, reranked_by_q)
                                        if q["type"] not in NO_ANSWER_TYPES]):
        chunks = fed_chunks(reranked, args.threshold, args.feed_topk)
        if not chunks:
            continue
        expected = {int(d) for d in (q.get("expected_doc_ids") or [])}
        fed_records.append({
            "id": q.get("id"),
            "score": float(chunks[0]["rerank_score"]),
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
        capture = (weak_err / wrong_total) if wrong_total else 0.0
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

    # ---------- cosine 基线（docs/04 结果，若存在） ----------
    baseline_cosine = None
    baseline_path = Path(args.baseline_summary)
    if baseline_path.exists():
        try:
            raw = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_cosine = {
                "source": display_path(baseline_path),
                "config": raw.get("config"),
                "baseline": raw.get("baseline"),
                "score_separation": raw.get("score_separation"),
            }
        except Exception as exc:  # noqa: BLE001 - 对照展示项，失败不影响指标
            print(f"[警告] 读取 cosine 基线失败：{exc}")

    # ---------- 导出 ----------
    write_csv(output_dir / "rerank_per_query.csv", rows, list(rows[0].keys()))
    write_csv(output_dir / "rerank_threshold_sweep.csv", sweep, list(sweep[0].keys()))
    write_csv(output_dir / "rerank_margin_sweep.csv", margin_rows, list(margin_rows[0].keys()))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 2),
        "model": {
            "ref": reranker._model_ref(),
            "device": reranker.device,
            "load_seconds": round(load_seconds, 2),
            "rerank_ms_per_query_mean": round(mean(rerank_ms_list), 1) if rerank_ms_list else None,
            "rerank_ms_per_query_median": round(median(rerank_ms_list), 1) if rerank_ms_list else None,
            "rerank_ms_per_query_max": round(max(rerank_ms_list), 1) if rerank_ms_list else None,
            "candidate_k": args.candidate_k,
        },
        "questions": {
            "path": display_path(questions_path),
            "total": len(questions),
            "answerable": len(answerable),
            "no_answer": no_ans_count,
            "by_type": {t: sum(1 for q in questions if q["type"] == t)
                        for t in sorted({q["type"] for q in questions})},
        },
        "config": {
            "threshold": args.threshold,
            "margin": args.margin,
            "candidate_k": args.candidate_k,
            "feed_topk": args.feed_topk,
        },
        "rerank_metrics": {
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
        "cosine_baseline": baseline_cosine,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rerank_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 控制台摘要 ----------
    print("\n" + "=" * 78)
    print("精排指标（阈值 %.2f）" % args.threshold)
    print(f"  recall@{args.candidate_k}（召回上限）: {fmt_pct(rec_hits, len(answerable))}  "
          f"({rec_hits}/{len(answerable)})")
    print(f"  MRR                    : {summary['rerank_metrics']['mrr']:.4f}")
    print(f"  覆盖（有片段可喂）     : {fmt_pct(fed_ans, len(answerable))}  "
          f"({fed_ans}/{len(answerable)})")
    print(f"  hit@{args.feed_topk}（正确片段进上下文）: {fmt_pct(hit_ans, len(answerable))}  "
          f"({hit_ans}/{len(answerable)})")
    print(f"  top1 准确率            : {fmt_pct(top1_ans, len(answerable))}  "
          f"（已喂片段中 {fmt_pct(top1_ans, fed_ans)}）")
    print(f"  无答案误接受（far/near/all）: {fmt_pct(fa_far, len(far))} / "
          f"{fmt_pct(fa_near, len(near))} / {fmt_pct(fa_all, no_ans_count)}")
    print("\n精排分数可分离性")
    print(f"  正确命中 top1 最低分   : {summary['score_separation']['correct_hit_top1_min']}")
    print(f"  可回答 top1 中位数     : {summary['score_separation']['answerable_top1_median']}")
    print(f"  无答案 top1 最高分     : {summary['score_separation']['no_answer_top1_max']}")
    print(f"  单条精排耗时           : 均值 {summary['model']['rerank_ms_per_query_mean']} ms / "
          f"最大 {summary['model']['rerank_ms_per_query_max']} ms")
    print("\n推荐值")
    print(f"  精排阈值 : {best['threshold']}（balanced accuracy {best['balanced_accuracy_pct']}%）")
    print(f"  稳定区间 : {summary['recommendations']['stable_threshold_range']}")
    print(f"  不丢正确片段的最大阈值: {max_full_recall}")
    if wrong_scores:
        print(f"  弱命中余量: {best_margin['margin']}（Youden J {best_margin['youden_j_pct']}%）")
        print(f"  覆盖全部错误的最小余量: {round(margin_all_wrong, 3)}")
    else:
        print("  弱命中余量: 题库中无错误 top1 样本，余量无法标定；"
              f"当前 {args.margin} 会把 {current_margin_row['correct_downgraded_pct']}% "
              "的正确命中降级为 weak_hit")
    if baseline_cosine:
        base = baseline_cosine.get("baseline") or {}
        print("\n对照 docs/04 cosine 基线（同题库）")
        print(f"  hit@3 {base.get('hit_at_feed_pct')}% / top1 {base.get('top1_accuracy_pct')}% / "
              f"误接受 {base.get('false_accept_all_pct')}% / MRR {base.get('mrr')}")
    print(f"\n结果文件: {output_dir}")
    print(f"用时 {elapsed:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
