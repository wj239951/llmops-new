# -*- coding: utf-8 -*-
"""多轮检索改写评测：20 段会话，比较 off / rule / llm 下第 2 轮的检索命中。

第 1 轮按单轮检索建"历史"，第 2 轮才是被测对象（指代 / 省略）。
指标与 docs/13 一致：hit@3（正确片段是否进上下文）、top1 是否正确、recall@6。
"""
from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
REPO = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from app.core.config import settings  # noqa: E402
from app.core.database import SessionLocal  # noqa: E402
from app.services import query_rewrite  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402

SEGMENTS = os.path.join(REPO, "experiments", "eval_multiturn.jsonl")
OUT_DIR = os.path.join(REPO, "experiments", "results", "rewrite_20261117")


def run_mode(svc, db, segs, mode: str):
    settings.RAG_QUERY_REWRITE_MODE = mode
    rows = []
    for seg in segs:
        turns = seg["turns"]
        first = svc.search(db, turns[0]["question"], use_rerank=False)
        history = [{"role": "user", "content": turns[0]["question"]}]
        second = turns[1]
        search_q, meta = query_rewrite.rewrite(second["question"], history)
        docs = svc.search(db, search_q, use_rerank=False)
        got = [d.doc_id for d in docs]
        expected = set(second["expected_doc_ids"])
        rows.append({
            "id": seg["id"],
            "mode": mode,
            "turn1_hit": int(bool(expected & {d.doc_id for d in first})),
            "query": second["question"],
            "rewritten": search_q,
            "changed": meta["changed"],
            "hit3": int(bool(expected & set(got))),
            "top1": int(bool(got) and got[0] in expected),
            "scores": [round(d.score, 4) for d in docs],
        })
    return rows


def summarize(rows):
    n = len(rows)
    return {
        "n": n,
        "hit3_pct": round(100 * sum(r["hit3"] for r in rows) / n, 1),
        "top1_pct": round(100 * sum(r["top1"] for r in rows) / n, 1),
        "changed": sum(r["changed"] for r in rows),
    }


def main():
    modes = sys.argv[1:] or ["off", "rule"]
    segs = [json.loads(line) for line in io.open(SEGMENTS, encoding="utf-8") if line.strip()]
    svc = RAGService()
    db = SessionLocal()
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    print(f"多轮改写评测：{len(segs)} 段，阈值 cosine={settings.RAG_SCORE_THRESHOLD}")
    print(f"{'模式':<6}{'hit@3':>8}{'top1':>8}{'改写数':>8}")
    summary = {}
    try:
        for mode in modes:
            rows = run_mode(svc, db, segs, mode)
            s = summarize(rows)
            summary[mode] = s
            all_rows.extend(rows)
            print(f"{mode:<6}{s['hit3_pct']:>7.1f}%{s['top1_pct']:>7.1f}%{s['changed']:>8}")
    finally:
        db.close()

    tag = "_".join(modes)
    with io.open(os.path.join(OUT_DIR, f"per_segment_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with io.open(os.path.join(OUT_DIR, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果：{OUT_DIR}")


if __name__ == "__main__":
    main()
