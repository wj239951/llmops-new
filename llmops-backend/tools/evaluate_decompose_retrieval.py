# -*- coding: utf-8 -*-
"""查询分解层在「检索链路内」的评测底座。

背景：60 条耦合问句是复合问句（A 两问拼接 / B 含型号 / C 三问 / D 含无答案主题），
一次 dense 检索只能拿到其中一部分文档。分解之后每个子句各查一次、并集合并，
才能把期望文档全部捞回。本脚本量出这个差距，作为蒸馏的靶子。

三种条件：
- off      ：整句直接检索（线上未开分解层的现状）
- teacher  ：用 decompose_cache.jsonl 里老师的子句（上限参考）
- student  ：用某个学生模型产出的子句（--student 文件，格式同 teacher）

指标沿用 tests 里"full 率"的口径：
- full      ：期望文档是否**全部**出现在喂给模型的片段里（这是分解层的终极判据）
- coverage  ：期望文档命中比例（部分命中给部分分）
- recall@K  ：逐文档看，K 条候选里命中率
- first_doc ：第一条候选是否来自任一期望文档

用法（在 llmops-backend 目录执行）：
    .venv\\Scripts\\python.exe tools/evaluate_decompose_retrieval.py
    .venv\\Scripts\\python.exe tools/evaluate_decompose_retrieval.py --feed-topk 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "tools"))

from evaluate_clean_retrieval import (  # noqa: E402
    OllamaEmbedder, build_chunks, normalize, parse_docs, SQL_PATH, top_k,
)

DATA_DIR = Path(r"E:\数据集收集\耦合问句60题")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def retrieve_docs(qvec, doc_vecs, chunks, k: int) -> list[int]:
    """取 K 条候选，返回按分数排序的 doc_id（去重）。"""
    cands = top_k(qvec, doc_vecs, chunks, k)
    seen: list[int] = []
    for c in cands:
        if c["doc_id"] not in seen:
            seen.append(c["doc_id"])
    return seen


def evaluate(rows: list[dict], sub_map: dict[str, list[str]], doc_vecs, chunks,
             embedder, candidate_k: int, feed_topk: int, tag: str = "on") -> dict:
    """逐题跑「整句」与「分解后」两种检索，返回两份逐题结果。"""
    queries = [r["query"] for r in rows]
    qvecs = normalize(embedder.encode(queries, "整句"))

    all_subs = [s for r in rows for s in sub_map.get(r["id"], [])]
    svecs = normalize(embedder.encode(all_subs, tag)) if all_subs else []
    cursor = 0
    sub_vecs: dict[str, list[list[float]]] = {}
    for r in rows:
        n = len(sub_map.get(r["id"], []))
        sub_vecs[r["id"]] = svecs[cursor: cursor + n]
        cursor += n

    off_rows, on_rows = [], []
    for r, qvec in zip(rows, qvecs):
        expected = set(r["expected_doc_ids"])

        # ── off：整句一条路
        docs = retrieve_docs(qvec, doc_vecs, chunks, candidate_k)
        fed = set(docs[:feed_topk])
        off_rows.append({
            "id": r["id"], "group": r.get("group"),
            "full": 1.0 if expected <= fed else 0.0,
            "coverage": len(expected & fed) / len(expected),
            "first_doc": 1.0 if docs and docs[0] in expected else 0.0,
        })

        # ── on：每个子句各取 K 条，按分数并集后截断到 feed_topk
        if not sub_vecs.get(r["id"]):
            on_rows.append(off_rows[-1] | {"id": r["id"]})
            continue
        merged: list[tuple[float, int]] = []
        for svec in sub_vecs[r["id"]]:
            cands = top_k(svec, doc_vecs, chunks, candidate_k)
            merged.extend((c["score"], c["doc_id"]) for c in cands)
        merged.sort(reverse=True)
        docs_on: list[int] = []
        for _, doc_id in merged:
            if doc_id not in docs_on:
                docs_on.append(doc_id)
        fed_on = set(docs_on[:feed_topk])
        on_rows.append({
            "id": r["id"], "group": r.get("group"),
            "full": 1.0 if expected <= fed_on else 0.0,
            "coverage": len(expected & fed_on) / len(expected),
            "first_doc": 1.0 if docs_on and docs_on[0] in expected else 0.0,
        })
    return {"off": off_rows, "on": on_rows}


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "full": mean(r["full"] for r in rows),
        "coverage": mean(r["coverage"] for r in rows),
        "first_doc": mean(r["first_doc"] for r in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-k", type=int, default=6)
    ap.add_argument("--feed-topk", type=int, default=3)
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--subs-file", default="", help="学生产出的子句文件（id/query/subs），给了就加一列")
    ap.add_argument("--out", default=str(REPO_DIR / "experiments" / "results"))
    args = ap.parse_args()

    docs = parse_docs(SQL_PATH)
    chunks = build_chunks(docs)
    rows = load_jsonl(DATA_DIR / "questions.jsonl")
    cache = {r["id"]: r.get("subs") or [] for r in load_jsonl(DATA_DIR / "decompose_cache.jsonl")}
    print(f"语料 {len(docs)} 篇 -> {len(chunks)} 片段；验收题 {len(rows)} 条；"
          f"老师子句 {sum(len(v) for v in cache.values())} 条")

    embedder = OllamaEmbedder(args.embed_model, args.ollama)
    doc_vecs = normalize(embedder.encode([c["content"] for c in chunks], "docs"))

    res_teacher = evaluate(rows, cache, doc_vecs, chunks, embedder, args.candidate_k,
                           args.feed_topk, tag="老师子句")
    conditions: dict[str, dict] = {
        "off 整句": summarize(res_teacher["off"]),
        "on 老师分解": summarize(res_teacher["on"]),
    }
    res_student = None
    if args.subs_file:
        student_rows = load_jsonl(Path(args.subs_file))
        student_map = {r["id"]: r.get("subs") or [] for r in student_rows}
        res_student = evaluate(rows, student_map, doc_vecs, chunks, embedder,
                               args.candidate_k, args.feed_topk, tag="学生子句")
        conditions["on 学生分解"] = summarize(res_student["on"])

    print(f"\n{'条件':<14}{'full':>8}{'coverage':>10}{'first_doc':>11}")
    for name, m in conditions.items():
        print(f"{name:<14}{m['full']:>7.1%}{m['coverage']:>10.1%}{m['first_doc']:>11.1%}")

    print("\n按形态（A 两问 / B 含型号 / C 三问 / D 含无答案）：")
    head = f"{'形态':<6}{'n':>4}{'off':>9}{'老师':>9}" + (f"{'学生':>9}" if res_student else "")
    print(head)
    for g in ("A", "B", "C", "D"):
        o = [r for r in res_teacher["off"] if r["group"] == g]
        if not o:
            continue
        t = [r for r in res_teacher["on"] if r["group"] == g]
        line = f"{g:<6}{len(o):>4}{summarize(o)['full']:>9.1%}{summarize(t)['full']:>9.1%}"
        if res_student:
            s = [r for r in res_student["on"] if r["group"] == g]
            line += f"{summarize(s)['full']:>9.1%}"
        print(line)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_questions": len(rows), "n_docs": len(docs), "n_chunks": len(chunks),
        "candidate_k": args.candidate_k, "feed_topk": args.feed_topk,
        "conditions": conditions,
        "by_group": {
            g: {name: summarize([r for r in res_teacher[key] if r["group"] == g])
                for name, key in (("off", "off"), ("teacher", "on"))}
            for g in ("A", "B", "C", "D")
        },
    }
    if res_student:
        summary["by_group"] = {
            g: {**summary["by_group"][g],
                "student": summarize([r for r in res_student["on"] if r["group"] == g])}
            for g in ("A", "B", "C", "D")
        }
        summary["subs_file"] = str(args.subs_file)
    path = out / f"decompose_retrieval_baseline_k{args.feed_topk}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
