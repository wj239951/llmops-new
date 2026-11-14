# -*- coding: utf-8 -*-
"""验证「分解在检索入口生效」这条生产代码路径，而不是只在评测脚本里模拟。

做法：把 RAGService 的检索层替换成内存向量检索（与评测脚本同一份语料/切块/embedding），
其余全部走真实代码——search() 入口、query_decompose.decompose()、阈值过滤、
_multi_query_cosine() 的并集合并。

期望：off 与 on 的 full 率，与 evaluate_gated_retrieval.py 在 cosine 档下算出的
81.7% / 90.0% 一致。一致即说明评测口径与生产实现没有分叉。

用法：
    .venv\\Scripts\\python.exe tools/verify_decompose_entry.py
"""
from __future__ import annotations

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

COUPLED_DIR = Path(r"E:\数据集收集\耦合问句60题")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> int:
    from app.core.config import settings
    from app.services import query_decompose
    from app.services.rag_service import RAGService, RetrievedChunk

    settings.RAG_RERANK_ENABLED = False
    settings.RAG_SCORE_THRESHOLD = 0.53
    settings.RAG_HYBRID_ENABLED = False
    settings.RAG_QUERY_DECOMPOSE_ENABLED = True

    docs = parse_docs(SQL_PATH)
    chunks = build_chunks(docs)
    questions = load_jsonl(COUPLED_DIR / "questions.jsonl")
    subs_map = {r["id"]: r.get("subs") or [r["query"]]
                for r in load_jsonl(COUPLED_DIR / "decompose_cache.jsonl")}

    embedder = OllamaEmbedder()
    doc_vecs = normalize(embedder.encode([c["content"] for c in chunks], "docs"))
    qvecs = normalize(embedder.encode([q["query"] for q in questions], "queries"))
    sub_flat = [s for q in questions for s in subs_map[q["id"]]]
    svecs_flat = normalize(embedder.encode(sub_flat, "subs"))
    cursor, sub_vecs = 0, {}
    for q in questions:
        n = len(subs_map[q["id"]])
        sub_vecs[q["id"]] = svecs_flat[cursor: cursor + n]
        cursor += n

    svc = RAGService()
    svc.rerank_state = lambda use_rerank=None: (False, False, "cosine")  # type: ignore[assignment]

    # 检索层替身：按 query 字符串在内存向量里取 top-k
    def retrieve(db, query, k):  # noqa: ANN001
        if query in sub_flat:
            i = sub_flat.index(query)
            vec = svecs_flat[i]
        else:
            vec = qvecs[[q["query"] for q in questions].index(query)]
        out = []
        for c in top_k(vec, doc_vecs, chunks, k):
            out.append(RetrievedChunk(
                title=c["title"], content="", source=None,
                score=float(c["score"]), doc_id=c["doc_id"], chunk=c["chunk"],
            ))
        return out

    svc._retrieve_candidates = retrieve  # type: ignore[assignment]

    def run(decompose_on: bool) -> dict:
        rows = []
        for q, vec in zip(questions, qvecs):
            if decompose_on:
                query_decompose.decompose = lambda text, _q=q: subs_map[_q["id"]]  # type: ignore
                fed = [c.doc_id for c in svc.search(None, q["query"])]
            else:
                query_decompose.decompose = lambda text: [text]  # type: ignore
                fed = [c.doc_id for c in svc.search(None, q["query"])]
            expected = set(q["expected_doc_ids"])
            rows.append({
                "full": 1.0 if expected <= set(fed) else 0.0,
                "coverage": len(expected & set(fed)) / len(expected),
            })
        return {"n": len(rows), "full": mean(r["full"] for r in rows),
                "coverage": mean(r["coverage"] for r in rows)}

    off, on = run(False), run(True)
    out = {
        "note": "走真实 RAGService.search()；检索层替换为内存向量，其余未改",
        "cosine_threshold": settings.RAG_SCORE_THRESHOLD,
        "off": off, "on_teacher": on,
    }
    path = REPO_DIR / "experiments" / "results" / "code_path_check.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"off（整句）   full={off['full']:.1%}  coverage={off['coverage']:.1%}")
    print(f"on（入口分解）full={on['full']:.1%}  coverage={on['coverage']:.1%}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
