# -*- coding: utf-8 -*-
"""混合检索离线评测：dense / hybrid / dense+rerank / hybrid+rerank 四模式对照。

复用 tools/evaluate_rag.py 的题库加载与指标工具，不经过 FastAPI，直接调用检索层：
  dense          : bge-m3 向量召回 → cosine 阈值 → topN
  hybrid         : bge-m3 向量召回 + jieba/BM25 关键词召回 → RRF 融合 → cosine 阈值 → topN
  dense_rerank   : 向量候选 → bge-reranker 精排 → 精排阈值 → topN
  hybrid_rerank  : 混合候选 → bge-reranker 精排 → 精排阈值 → topN

用法（在 llmops-backend 目录下跑）：
    python tools/evaluate_hybrid.py
    python tools/evaluate_hybrid.py --candidate-k 6 --feed-topk 3
输出：
    experiments/results/hybrid_compare.csv
    experiments/results/hybrid_per_query.csv
    experiments/results/hybrid_summary.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.core.database import SessionLocal  # noqa: E402
from app.services.rag_service import RAGService, RetrievedChunk  # noqa: E402
from app.services.rerank_service import CrossEncoderReranker  # noqa: E402
from tools.evaluate_rag import (  # noqa: E402
    NO_ANSWER_TYPES,
    display_path,
    fed_chunks,
    first_doc_rank,
    load_questions,
    pct,
    write_csv,
)

MODES = ("dense", "hybrid", "dense_rerank", "hybrid_rerank")
DEFAULT_QUESTIONS = (
    REPO_DIR / "experiments" / "eval_qa.jsonl",
    REPO_DIR / "experiments" / "eval_qa_hard.jsonl",
    REPO_DIR / "experiments" / "eval_qa_token.jsonl",
)


def _fmt_rate(value: float) -> str:
    """把 _summarize 里的百分数（单位 %）格式化成字符串。"""
    return f"{value:5.1f}%"


def _chunk_to_dict(chunk: RetrievedChunk) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "chunk": chunk.chunk,
        "score": chunk.score,
        "content": chunk.content,
    }


def _pool(service: RAGService, db, mode: str, query: str, candidate_k: int):
    """返回 (候选列表, rerank 耗时 ms)。"""
    if mode.startswith("hybrid"):
        pool = service._hybrid_candidates(db, query, candidate_k)
    else:
        pool = service._dense_chunks(db, query, candidate_k)
    return pool


def _apply_rerank(pool: list[RetrievedChunk], query: str, reranker: CrossEncoderReranker):
    started = time.perf_counter()
    scores = reranker.score(query, [c.content for c in pool])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    reranked = [
        dataclasses.replace(c, score=float(s)) for c, s in zip(pool, scores)
    ]
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked, elapsed_ms


def _evaluate(dataset_name, questions, mode, service, db, reranker, candidate_k,
              feed_topk, cosine_threshold, rerank_threshold):
    rows = []
    latencies = []
    rerank_times = []
    for q in questions:
        started = time.perf_counter()
        if mode.endswith("_rerank"):
            pool = _pool(service, db, mode, q["question"], candidate_k)
            if pool:
                pool, rerank_ms = _apply_rerank(pool, q["question"], reranker)
                rerank_times.append(rerank_ms)
            threshold = rerank_threshold
        else:
            pool = _pool(service, db, mode, q["question"], candidate_k)
            threshold = cosine_threshold
        cands = [_chunk_to_dict(c) for c in pool]
        if not mode.endswith("_rerank"):
            # 与 RAGService.search 的纯 cosine 路径一致：混合候选按 RRF 排序，最终仍需按 cosine 重排
            cands.sort(key=lambda c: c["score"], reverse=True)
        fed = fed_chunks(cands, threshold, feed_topk)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        rows.append(
            {
                "dataset": dataset_name,
                "mode": mode,
                "qid": q["id"],
                "type": q.get("type", ""),
                "question": q["question"],
                "expected_doc_ids": "|".join(str(x) for x in q["expected_doc_ids"]),
                "rank": first_doc_rank(cands, set(q["expected_doc_ids"])),
                "fed_doc_ids": "|".join(str(c["doc_id"]) for c in fed),
                "top1_doc_id": fed[0]["doc_id"] if fed else "",
                "latency_ms": round(latency_ms, 2),
                "rerank_ms": round(rerank_times[-1], 2) if rerank_times and mode.endswith("_rerank") else "",
            }
        )
    return rows, latencies, rerank_times


def _summarize(rows, latencies, rerank_times, candidate_k):
    answerable = [r for r in rows if r["type"] not in NO_ANSWER_TYPES]
    no_answer = [r for r in rows if r["type"] in NO_ANSWER_TYPES]
    hit_rows = [r for r in answerable if r["rank"] is not None and r["rank"] <= candidate_k]
    covered = [r for r in answerable if r["fed_doc_ids"]]
    fed_hit = [
        r
        for r in answerable
        if set(str(r["fed_doc_ids"]).split("|")) & {str(x) for x in r["expected_doc_ids"].split("|")}
    ]
    top1 = [
        r for r in answerable if r["top1_doc_id"] != "" and str(r["top1_doc_id"]) in str(r["expected_doc_ids"]).split("|")
    ]
    fa = [r for r in no_answer if r["fed_doc_ids"]]
    fa_far = [r for r in no_answer if r["type"] == "no_answer_far" and r["fed_doc_ids"]]
    fa_near = [r for r in no_answer if r["type"] == "no_answer_near" and r["fed_doc_ids"]]
    far_total = [r for r in no_answer if r["type"] == "no_answer_far"]
    near_total = [r for r in no_answer if r["type"] == "no_answer_near"]
    mrr = (
        sum(1.0 / r["rank"] for r in answerable if r["rank"]) / len(answerable)
        if answerable
        else 0.0
    )
    lat_sorted = sorted(latencies)
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))] if lat_sorted else 0.0
    return {
        "questions": len(rows),
        "answerable": len(answerable),
        "no_answer": len(no_answer),
        "recall_at_k": pct(len(hit_rows), len(answerable)),
        "mrr": round(mrr, 4),
        "coverage": pct(len(covered), len(answerable)),
        "hit_at_feed": pct(len(fed_hit), len(answerable)),
        "top1_accuracy": pct(len(top1), len(answerable)),
        "false_accept_far": pct(len(fa_far), len(far_total)),
        "false_accept_near": pct(len(fa_near), len(near_total)),
        "false_accept_all": pct(len(fa), len(no_answer)),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(p95, 1),
        "mean_rerank_ms": round(sum(rerank_times) / len(rerank_times), 1) if rerank_times else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="混合检索四模式离线评测")
    parser.add_argument("--questions", nargs="+", default=[str(p) for p in DEFAULT_QUESTIONS])
    parser.add_argument("--output-dir", default=str(REPO_DIR / "experiments" / "results"))
    parser.add_argument("--candidate-k", type=int, default=6)
    parser.add_argument("--feed-topk", type=int, default=settings.RAG_FEED_TOPK)
    parser.add_argument("--cosine-threshold", type=float, default=settings.RAG_SCORE_THRESHOLD)
    parser.add_argument("--rerank-threshold", type=float, default=settings.RAG_RERANK_SCORE_THRESHOLD)
    parser.add_argument("--model-path", default=settings.RAG_RERANK_MODEL_PATH or None)
    parser.add_argument("--device", default=settings.RAG_RERANK_DEVICE)
    parser.add_argument("--skip-rerank", action="store_true", help="只跑 dense/hybrid 两种纯召回模式")
    args = parser.parse_args(argv)

    datasets = []
    for path_text in args.questions:
        path = Path(path_text)
        if not path.exists():
            print(f"[skip] 题库不存在：{path}")
            continue
        datasets.append((path.stem.replace("eval_qa_", "") or "main", path, load_questions(path)))
    if not datasets:
        print("没有可用题库")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = [m for m in MODES if not (args.skip_rerank and m.endswith("_rerank"))]
    if any(m.startswith("hybrid") for m in modes):
        # 离线评测需要显式打开混合检索；线上由 .env 的 RAG_HYBRID_ENABLED 控制。
        settings.RAG_HYBRID_ENABLED = True
        from app.services import hybrid_service

        if hybrid_service.is_available():
            print(f"[info] 混合检索索引已加载：docs={len(hybrid_service.get_index().docs)}")
        else:
            print("[warn] 混合检索索引不可用，hybrid 模式将退化为 dense")
    reranker = None
    skipped_modes = []
    if any(m.endswith("_rerank") for m in modes):
        reranker = CrossEncoderReranker(model_ref=args.model_path)
        if not reranker.load():
            print(f"[warn] 精排模型不可用，跳过精排模式：{reranker.load_error}")
            modes = [m for m in modes if not m.endswith("_rerank")]
            skipped_modes = [m for m in MODES if m.endswith("_rerank")]
        else:
            print(f"[info] 精排模型已加载：device={reranker.device}")

    service = RAGService()
    db = SessionLocal()
    per_query_rows = []
    compare_rows = []
    results = []
    try:
        for dataset_name, path, questions in datasets:
            for mode in modes:
                rows, latencies, rerank_times = _evaluate(
                    dataset_name, questions, mode, service, db, reranker,
                    args.candidate_k, args.feed_topk, args.cosine_threshold,
                    args.rerank_threshold,
                )
                per_query_rows.extend(rows)
                summary = _summarize(rows, latencies, rerank_times, args.candidate_k)
                summary.update(
                    {
                        "dataset": dataset_name,
                        "dataset_path": display_path(path),
                        "mode": mode,
                        "candidate_k": args.candidate_k,
                        "feed_topk": args.feed_topk,
                        "threshold": args.rerank_threshold if mode.endswith("_rerank") else args.cosine_threshold,
                    }
                )
                results.append(summary)
                compare_rows.append(
                    {
                        "dataset": dataset_name,
                        "mode": mode,
                        "questions": summary["questions"],
                        "answerable": summary["answerable"],
                        "no_answer": summary["no_answer"],
                        "recall_at_k": _fmt_rate(summary["recall_at_k"]),
                        "mrr": summary["mrr"],
                        "coverage": _fmt_rate(summary["coverage"]),
                        "hit_at_feed": _fmt_rate(summary["hit_at_feed"]),
                        "top1_accuracy": _fmt_rate(summary["top1_accuracy"]),
                        "false_accept_far": _fmt_rate(summary["false_accept_far"]),
                        "false_accept_near": _fmt_rate(summary["false_accept_near"]),
                        "false_accept_all": _fmt_rate(summary["false_accept_all"]),
                        "mean_latency_ms": summary["mean_latency_ms"],
                        "mean_rerank_ms": summary["mean_rerank_ms"] if summary["mean_rerank_ms"] is not None else "",
                    }
                )
                print(
                    f"[{dataset_name:>5}] {mode:<14} recall@{args.candidate_k}={_fmt_rate(summary['recall_at_k'])} "
                    f"MRR={summary['mrr']:.4f} hit@{args.feed_topk}={_fmt_rate(summary['hit_at_feed'])} "
                    f"top1={_fmt_rate(summary['top1_accuracy'])} 误接受={_fmt_rate(summary['false_accept_all'])} "
                    f"延迟={summary['mean_latency_ms']}ms"
                )
    finally:
        db.close()

    per_query_fields = [
        "dataset", "mode", "qid", "type", "question", "expected_doc_ids",
        "rank", "fed_doc_ids", "top1_doc_id", "latency_ms", "rerank_ms",
    ]
    compare_fields = [
        "dataset", "mode", "questions", "answerable", "no_answer", "recall_at_k",
        "mrr", "coverage", "hit_at_feed", "top1_accuracy", "false_accept_far",
        "false_accept_near", "false_accept_all", "mean_latency_ms", "mean_rerank_ms",
    ]
    write_csv(output_dir / "hybrid_per_query.csv", per_query_rows, per_query_fields)
    write_csv(output_dir / "hybrid_compare.csv", compare_rows, compare_fields)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "candidate_k": args.candidate_k,
            "feed_topk": args.feed_topk,
            "cosine_threshold": args.cosine_threshold,
            "rerank_threshold": args.rerank_threshold,
            "rerank_model": args.model_path or settings.RAG_RERANK_MODEL,
            "hybrid_dense_k": settings.RAG_HYBRID_DENSE_K,
            "hybrid_sparse_k": settings.RAG_HYBRID_SPARSE_K,
            "hybrid_rrf_k": settings.RAG_HYBRID_RRF_K,
        },
        "datasets": [
            {"name": name, "path": display_path(path), "questions": len(qs)}
            for name, path, qs in datasets
        ],
        "modes": modes,
        "skipped_modes": skipped_modes,
        "results": results,
        "notes": [
            "dense/hybrid 模式使用 cosine 阈值，rerank 模式使用精排阈值；两者分数尺度不同。",
            "hybrid 模式即为「jieba/BM25 关键词召回 + 向量召回 RRF 融合」，与线上 RAG_HYBRID_ENABLED 开关配套。",
        ],
    }
    with open(output_dir / "hybrid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[ok] 评测完成：{display_path(output_dir)}")
    for name in ("hybrid_compare.csv", "hybrid_per_query.csv", "hybrid_summary.json"):
        print(f"      - {output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
