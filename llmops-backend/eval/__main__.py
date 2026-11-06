# -*- coding: utf-8 -*-
"""RAG 评测入口：一条命令跑完「检索三模式 + 可选 LLM 判分 + 回归门禁」。

用法（在 llmops-backend 目录下）：
    python -m eval run --modes dense,hybrid,rerank
    python -m eval run --set token --modes dense,hybrid
    python -m eval run --judge --judge-limit 10
    python -m eval gate --baseline eval_results/baseline/summary.json --current eval_results/latest/summary.json
    python -m eval check          # 只校验评测集格式，不连数据库
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from eval import golden, report
from eval.runner import MODES

DEFAULT_OUT = Path("eval_results")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def cmd_check(args) -> int:
    for name in golden.SETS:
        questions = golden.load_set(name)
        s = golden.stats(questions)
        print(f"[OK] {name}: {s['total']} 题（可答 {s['answerable']} / 无答案 {s['no_answer']}）{s['by_type']}")
    return 0


def cmd_run(args) -> int:
    from app.core.config import settings
    from app.core.database import SessionLocal
    from eval import judge as judge_mod
    from eval.runner import run_mode

    questions = golden.load_set(args.set, Path(args.file) if args.file else None)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        print(f"未知模式：{unknown}，可选 {list(MODES)}", file=sys.stderr)
        return 2

    db = SessionLocal()
    summary: dict = {}
    per_query: dict = {}
    try:
        for mode in modes:
            print(f"→ 跑 {mode}（{MODES[mode]['desc']}）…", flush=True)
            t0 = time.time()
            records, s = run_mode(mode, questions, db)
            s["elapsed_s"] = round(time.time() - t0, 1)
            summary[mode] = s
            per_query[mode] = records
            print(
                f"   recall@K={s['recall_at_candidate_pct']}%  MRR={s['mrr']}  "
                f"命中={s['hit_at_feed_pct']}%  误放行={s['false_accept_pct']}%  "
                f"P95={s['latency_p95_ms']}ms  ({s['elapsed_s']}s)"
            )

        meta = {
            "set": args.set,
            "generated_at": _now(),
            "questions": len(questions),
            "answerable": sum(1 for q in questions if q["answerable"]),
            "no_answer": sum(1 for q in questions if not q["answerable"]),
            "candidate_k": settings.RAG_CANDIDATE_K,
            "feed_topk": settings.RAG_FEED_TOPK,
            "threshold_cosine": settings.RAG_SCORE_THRESHOLD,
            "threshold_rerank": settings.RAG_RERANK_SCORE_THRESHOLD,
            "git_commit": _git_commit(),
        }

        judge_result = None
        if args.judge:
            print(f"→ LLM 判分（最多 {args.judge_limit or '全部'} 题）…", flush=True)
            judge_result = judge_mod.run_judge(
                questions, db, use_rerank=args.judge_rerank, limit=args.judge_limit,
            )
            meta["judge"] = {k: v for k, v in judge_result.items() if k != "rows"}
            print(f"   均分={judge_result['mean_score']}  4分以上占比={judge_result['score_ge4_pct']}%")

        out_dir = Path(args.out) if args.out else DEFAULT_OUT / time.strftime("%Y%m%d-%H%M%S")
        files = report.write_outputs(out_dir, summary, per_query, meta)
        if judge_result:
            (out_dir / "judge.json").write_text(
                json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            files["files"].append("judge.json")

        print(f"\n产物：{files['dir']}")
        print((out_dir / "report.md").read_text(encoding="utf-8"))

        if args.gate:
            ok, problems = report.gate(
                json.loads(Path(args.gate).read_text(encoding="utf-8"))["results"],
                summary,
                max_hit_drop=args.max_hit_drop,
                max_fa_rise=args.max_fa_rise,
            )
            if ok:
                print(f"[GATE] 通过（基线 {args.gate}）")
                return 0
            print("[GATE] 失败：", file=sys.stderr)
            for p in problems:
                print("  - " + p, file=sys.stderr)
            return 1
        return 0
    finally:
        db.close()


def cmd_gate(args) -> int:
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))["results"]
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["results"]
    ok, problems = report.gate(
        baseline, current, max_hit_drop=args.max_hit_drop, max_fa_rise=args.max_fa_rise,
    )
    if ok:
        print(f"[GATE] 通过：{args.current} 相对 {args.baseline} 无显著退化")
        return 0
    print("[GATE] 失败：", file=sys.stderr)
    for p in problems:
        print("  - " + p, file=sys.stderr)
    return 1


def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval", description="RAG 检索评测")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="校验评测集格式（不连数据库）")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("run", help="跑评测并生成报告")
    r.add_argument("--set", default="qa", choices=list(golden.SETS))
    r.add_argument("--file", default=None, help="自定义 jsonl 路径（覆盖 --set）")
    r.add_argument("--modes", default="dense,hybrid,rerank")
    r.add_argument("--out", default=None, help="产物目录，默认 eval_results/<时间戳>")
    r.add_argument("--judge", action="store_true", help="额外跑 LLM-as-judge 答案评分")
    r.add_argument("--judge-limit", type=int, default=10, help="判分题数上限（默认 10）")
    r.add_argument("--judge-rerank", action="store_true", help="判分时使用精排检索")
    r.add_argument("--gate", default=None, help="给定期望基线 summary.json，退化则退出码 1")
    r.add_argument("--max-hit-drop", type=float, default=2.0)
    r.add_argument("--max-fa-rise", type=float, default=5.0)
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("gate", help="对比两份 summary.json")
    g.add_argument("--baseline", required=True)
    g.add_argument("--current", required=True)
    g.add_argument("--max-hit-drop", type=float, default=2.0)
    g.add_argument("--max-fa-rise", type=float, default=5.0)
    g.set_defaults(func=cmd_gate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except golden.GoldenError as exc:
        print(f"评测集错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
