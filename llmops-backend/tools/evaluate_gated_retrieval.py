# -*- coding: utf-8 -*-
"""带闸门的检索评测：把「线上到底能不能捞全 / 能不能拒答」量出来。

之前两个评测脚本（evaluate_clean_retrieval / evaluate_decompose_retrieval）都是
**无阈值**口径——量的是"top-K 里有没有正确文档"，而线上是"过闸后还剩几条"。
两者差一个数量级（历史 0.76 闸门下 full 率 23.3%，无闸门是 90%），不能混着用。

本脚本把三种打分方式 × 两种查询组织方式一次跑完：

    打分方式：none（无闸门上界） / cosine 0.53（默认链路） / rerank 0.76（精排链路）
    组织方式：off（整句一条路） / on（复合句拆子句，各自检索后并集）

指标：
    full          期望文档是否全部进入"过闸后的 top-K"
    coverage      期望文档命中比例
    first_doc     第一条是否来自期望文档
    false_accept  无答案题是否仍被喂了片段（越低越好；只有 no_answer 题参与统计）
    延迟          retrieve / rerank 各多少 ms

用法（在 llmops-backend 目录执行）：
    .venv\\Scripts\\python.exe tools\\evaluate_gated_retrieval.py
    .venv\\Scripts\\python.exe tools\\evaluate_gated_retrieval.py --student-subs <jsonl>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "tools"))

from evaluate_clean_retrieval import (  # noqa: E402
    OllamaEmbedder, build_chunks, normalize, parse_docs, SQL_PATH, top_k,
)

COUPLED_DIR = Path(r"E:\数据集收集\耦合问句60题")
GOLDEN_DIR = BACKEND_DIR / "eval" / "golden"
NO_ANSWER_TYPES = {"no_answer", "no_answer_far", "no_answer_near"}


@dataclass
class Chunk:
    """与 rag_service.RetrievedChunk 同构的轻量替身，供 rerank_chunks 的 replace() 使用。"""

    content: str
    doc_id: int | None
    title: str = ""
    chunk: int | None = None
    score: float = 0.0
    source: str | None = None
    match: str = "vector"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def dedupe_docs(chunks: list[dict], limit: int) -> list[int]:
    seen: list[int] = []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.append(c["doc_id"])
        if len(seen) >= limit:
            break
    return seen


def fed_sets(passed: list[dict], feed_topk: int) -> tuple[list[int], list[int]]:
    """返回两种口径的"被喂进去的文档集合"。

    - chunk 级：生产实现是 `passed[:feed_topk]` 截断**片段**，同一篇文档可能占满 3 条，
      所以这是线上真实行为；
    - doc 级：先按文档去重再取 top-K，对应历史评测（0.76 闸门 23.3%）的口径。
    两个都给出来，避免拿宽松口径当线上表现。
    """
    fed = passed[:feed_topk]
    doc_level = dedupe_docs(passed, feed_topk)
    return list(dict.fromkeys(c["doc_id"] for c in fed)), doc_level


def run_mode(name: str, rows: list[dict], sub_map: dict[str, list[str]],
             doc_vecs, chunks, embedder, *, candidate_k: int, feed_topk: int,
             cosine_thr: float, rerank_thr: float, reranker):
    """在指定打分方式下跑 off / on 两种组织方式，返回逐题明细。"""
    use_rerank = name == "rerank"
    thr = {"none": 0.0, "cosine": cosine_thr, "rerank": rerank_thr}[name]

    # 一次把整句和所有子句向量化
    qvecs = normalize(embedder.encode([r["query"] for r in rows], f"{name}/整句"))
    subs_flat = [s for r in rows for s in sub_map.get(r["id"], [])]
    svecs = normalize(embedder.encode(subs_flat, f"{name}/子句")) if subs_flat else []
    cursor, sub_vecs = 0, {}
    for r in rows:
        n = len(sub_map.get(r["id"], []))
        sub_vecs[r["id"]] = svecs[cursor: cursor + n]
        cursor += n

    out = {"off": [], "on": []}
    retrieve_ms, rerank_ms = [], []
    content_map = {(c["doc_id"], c["chunk"]): c["content"] for c in chunks}

    def score_and_filter(query: str, cands: list[dict]) -> list[dict]:
        """按当前打分方式重排并过闸。"""
        nonlocal rerank_ms
        if use_rerank and cands:
            from app.services.rerank_service import rerank_chunks

            t0 = time.perf_counter()
            scored = rerank_chunks(query, [
                Chunk(content=content_map.get((c["doc_id"], c["chunk"]), ""),
                      doc_id=c["doc_id"], title=c.get("title", ""),
                      chunk=c.get("chunk"), score=c["score"]) for c in cands
            ])
            rerank_ms.append((time.perf_counter() - t0) * 1000)
            items = [{"doc_id": c.doc_id, "chunk": c.chunk,
                      "title": c.title, "score": float(c.score)} for c in scored]
        else:
            items = cands
        return [c for c in items if c["score"] >= thr]

    for r, qvec in zip(rows, qvecs):
        expected = set(r["expected_doc_ids"])

        t0 = time.perf_counter()
        cands_off = top_k(qvec, doc_vecs, chunks, candidate_k)
        retrieve_ms.append((time.perf_counter() - t0) * 1000)
        passed_off = score_and_filter(r["query"], cands_off)
        fed_off, fed_off_doc = fed_sets(passed_off, feed_topk)
        out["off"].append({
            "id": r["id"], "group": r.get("group") or r.get("type"),
            "expected": sorted(expected), "fed": fed_off, "fed_doc": fed_off_doc,
            "full": 1.0 if expected and expected <= set(fed_off) else (1.0 if not expected and not fed_off else 0.0),
            "full_doc": 1.0 if expected and expected <= set(fed_off_doc) else (1.0 if not expected and not fed_off_doc else 0.0),
            "coverage": (len(expected & set(fed_off)) / len(expected)) if expected else 0.0,
            "first_doc": 1.0 if fed_off and fed_off[0] in expected else 0.0,
            "false_accept": 1.0 if (not expected and fed_off) else 0.0,
            "n_fed": len(fed_off),
        })

        # on：每个子句各自检索、各自过闸，再按分数并集
        sub_list = sub_map.get(r["id"], [])
        if not sub_list:
            out["on"].append(out["off"][-1])
            continue
        merged: dict[tuple, dict] = {}
        for svec, sub in zip(sub_vecs[r["id"]], sub_list):
            t0 = time.perf_counter()
            cands = top_k(svec, doc_vecs, chunks, candidate_k)
            retrieve_ms.append((time.perf_counter() - t0) * 1000)
            for c in score_and_filter(sub, cands):
                key = (c["doc_id"], c["chunk"])
                if key not in merged or c["score"] > merged[key]["score"]:
                    merged[key] = c
        ordered = sorted(merged.values(), key=lambda c: c["score"], reverse=True)
        fed_on, fed_on_doc = fed_sets(ordered, feed_topk)
        out["on"].append({
            "id": r["id"], "group": r.get("group") or r.get("type"),
            "expected": sorted(expected), "fed": fed_on, "fed_doc": fed_on_doc,
            "full": 1.0 if expected and expected <= set(fed_on) else (1.0 if not expected and not fed_on else 0.0),
            "full_doc": 1.0 if expected and expected <= set(fed_on_doc) else (1.0 if not expected and not fed_on_doc else 0.0),
            "coverage": (len(expected & set(fed_on)) / len(expected)) if expected else 0.0,
            "first_doc": 1.0 if fed_on and fed_on[0] in expected else 0.0,
            "false_accept": 1.0 if (not expected and fed_on) else 0.0,
            "n_fed": len(fed_on),
        })
    return out, {"retrieve_ms": mean(retrieve_ms) if retrieve_ms else 0,
                 "rerank_ms": mean(rerank_ms) if rerank_ms else 0}


def summarize(rows: list[dict]) -> dict:
    """分开统计：有标准答案的题看 full/coverage，无答案题看 false_accept。"""
    answerable = [r for r in rows if r["expected"]]
    no_answer = [r for r in rows if not r["expected"]]
    m = {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_no_answer": len(no_answer),
        "full": mean(r["full"] for r in rows) if rows else 0.0,
        "full_doc": mean(r["full_doc"] for r in rows) if rows else 0.0,
        "coverage": mean(r["coverage"] for r in rows) if rows else 0.0,
        "first_doc": mean(r["first_doc"] for r in rows) if rows else 0.0,
        "false_accept": mean(r["false_accept"] for r in no_answer) if no_answer else None,
    }
    if answerable:
        m["full_answerable"] = mean(r["full"] for r in answerable)
        m["coverage_answerable"] = mean(r["coverage"] for r in answerable)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-k", type=int, default=6)
    ap.add_argument("--feed-topk", type=int, default=3)
    ap.add_argument("--cosine-threshold", type=float, default=0.53)
    ap.add_argument("--rerank-threshold", type=float, default=0.76)
    ap.add_argument("--student-subs", default="", help="学生分解结果（id/query/subs），给了就多跑一列")
    ap.add_argument("--out", default=str(REPO_DIR / "experiments" / "results"))
    args = ap.parse_args()

    # 精排：本地权重 + GPU。注意 settings 在 import 时已构建，必须改对象而不是环境变量。
    from app.core.config import settings
    settings.RAG_RERANK_ENABLED = True
    settings.RAG_RERANK_MODEL_PATH = r"D:\models\bge-reranker-v2-m3"
    settings.RAG_RERANK_DEVICE = "cuda"
    settings.RAG_RERANK_WARMUP = False
    from app.services import rerank_service

    if not rerank_service.is_available(force=True):
        print("精排不可用，rerank 档将退化为 cosine 分数（结果不可信）", file=sys.stderr)

    docs = parse_docs(SQL_PATH)
    chunks = build_chunks(docs)
    coupled = load_jsonl(COUPLED_DIR / "questions.jsonl")
    teacher_map = {r["id"]: r.get("subs") or [] for r in load_jsonl(COUPLED_DIR / "decompose_cache.jsonl")}
    no_answer_raw = [r for r in load_jsonl(GOLDEN_DIR / "qa.jsonl") if r.get("type") in NO_ANSWER_TYPES]
    # golden 的字段名是 question / expected_doc_ids，统一成评测内部用的 query
    no_answer = [{
        "id": r["id"],
        "query": r.get("question") or r.get("query"),
        "expected_doc_ids": r.get("expected_doc_ids") or [],
        "group": r.get("type"),
    } for r in no_answer_raw]
    print(f"语料 {len(docs)} 篇 / {len(chunks)} 片段；耦合题 {len(coupled)}；无答案题 {len(no_answer)}")

    embedder = OllamaEmbedder()
    doc_vecs = normalize(embedder.encode([c["content"] for c in chunks], "docs"))

    maps = {"老师": teacher_map}
    if args.student_subs:
        maps["学生"] = {r["id"]: r.get("subs") or [] for r in load_jsonl(Path(args.student_subs))}

    result: dict = {"meta": {"n_coupled": len(coupled), "n_no_answer": len(no_answer),
                             "n_docs": len(docs), "n_chunks": len(chunks),
                             "candidate_k": args.candidate_k, "feed_topk": args.feed_topk,
                             "cosine_threshold": args.cosine_threshold,
                             "rerank_threshold": args.rerank_threshold},
                    "coupled": {}, "no_answer": {}, "latency": {}}

    for mode in ("none", "cosine", "rerank"):
        for who, smap in maps.items():
            res, lat = run_mode(mode, coupled, smap, doc_vecs, chunks, embedder,
                                candidate_k=args.candidate_k, feed_topk=args.feed_topk,
                                cosine_thr=args.cosine_threshold, rerank_thr=args.rerank_threshold,
                                reranker=None)
            key = f"{mode}/off" if who == "老师" else f"{mode}/on-{who}"
            # off 与 who 无关，只存一份
            if who == "老师":
                result["coupled"][f"{mode}/off"] = summarize(res["off"])
            result["coupled"][f"{mode}/on-{who}"] = summarize(res["on"])
            result["latency"][key] = lat

            # 无答案题：只关心"有没有被喂片段"
            res_na, lat_na = run_mode(mode, no_answer, {r["id"]: [r["query"]] for r in no_answer},
                                      doc_vecs, chunks, embedder,
                                      candidate_k=args.candidate_k, feed_topk=args.feed_topk,
                                      cosine_thr=args.cosine_threshold, rerank_thr=args.rerank_threshold,
                                      reranker=None)
            result["no_answer"][f"{mode}/off"] = summarize(res_na["off"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gated_retrieval.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'条件':<18}{'full(片段)':>12}{'full(文档)':>12}{'coverage':>10}{'拒答失败':>10}")
    for key, m in result["coupled"].items():
        fa = m.get("false_accept")
        print(f"{key:<18}{m['full']:>11.1%}{m['full_doc']:>12.1%}{m['coverage']:>10.1%}"
              f"{(f'{fa:.0%}' if fa is not None else '-'):>10}")
    print("\n无答案题（false_accept 越低越好）：")
    for key, m in result["no_answer"].items():
        fa = m.get("false_accept")
        print(f"  {key:<16}{(f'{fa:.1%}' if fa is not None else '-'):>8}")
    print(f"\n汇总 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
