# -*- coding: utf-8 -*-
"""跑一个检索模式：候选生成 → （可选）精排 → 阈值截断，逐题记录指标。

刻意复用生产链路的构件（RAGService._dense_chunks / _hybrid_candidates /
rerank_service.rerank_chunks / 同一套阈值），而不是另写一套检索，
保证「评测跑的就是线上跑的那条路」。
"""
from __future__ import annotations

import time

from app.core.config import settings
from app.services.rag_service import RAGService
from eval import metrics

MODES = {
    "dense": {"hybrid": False, "rerank": False, "desc": "纯向量召回 + cosine 阈值"},
    "hybrid": {"hybrid": True, "rerank": False, "desc": "BM25 混合召回 + RRF + cosine 阈值"},
    "rerank": {"hybrid": True, "rerank": True, "desc": "混合召回 + bge-reranker 精排 + 精排阈值"},
}


def _candidates(rag: RAGService, db, query: str, cfg: dict) -> list:
    if cfg["hybrid"]:
        k = settings.RAG_RERANK_CANDIDATE_K if cfg["rerank"] else settings.RAG_CANDIDATE_K
        return rag._hybrid_candidates(db, query, k)
    return rag._dense_chunks(db, query, settings.RAG_CANDIDATE_K)


def _finalize(cands: list, cfg: dict, query: str) -> list:
    """按当前模式的阈值与截断规则，得到最终喂给模型的片段。"""
    feed_topk = settings.RAG_FEED_TOPK
    if cfg["rerank"]:
        from app.services import rerank_service
        rescored = rerank_service.rerank_chunks(query, cands)
        return [c for c in rescored if c.score >= settings.RAG_RERANK_SCORE_THRESHOLD][:feed_topk]
    passed = [
        c for c in cands
        if c.score >= settings.RAG_SCORE_THRESHOLD
        or (settings.RAG_HYBRID_EXACT_BYPASS and getattr(c, "match", "vector") == "exact")
    ]
    passed.sort(key=lambda c: (getattr(c, "match", "vector") == "exact", c.score), reverse=True)
    return passed[:feed_topk]


def run_mode(mode: str, questions: list[dict], db) -> tuple[list[dict], dict]:
    if mode not in MODES:
        raise ValueError(f"未知模式 {mode}，可选：{', '.join(MODES)}")
    cfg = MODES[mode]

    settings.RAG_HYBRID_ENABLED = cfg["hybrid"]
    if cfg["rerank"]:
        from app.services import rerank_service
        if not rerank_service.is_available(force=True):
            raise RuntimeError(f"模式 {mode} 需要精排，但模型不可用：{rerank_service.get_reranker().load_error}")

    rag = RAGService()
    records: list[dict] = []

    for q in questions:
        expected = set(q.get("expected_doc_ids") or [])
        t0 = time.time()
        cands = _candidates(rag, db, q["question"], cfg)
        cand_ms = int((time.time() - t0) * 1000)

        t1 = time.time()
        final = _finalize(cands, cfg, q["question"])
        final_ms = int((time.time() - t1) * 1000)
        total_ms = int((time.time() - t0) * 1000)

        rank = metrics.first_hit_rank(
            [{"doc_id": c.doc_id} for c in cands], expected
        ) if expected else None
        returned = bool(final)
        hit = any(c.doc_id in expected for c in final) if expected else False
        top1 = final[0].score if final else 0.0

        records.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answerable": bool(expected),
            "rank": rank,
            "hit": hit,
            "returned": returned,
            "n_candidates": len(cands),
            "n_final": len(final),
            "top1_score": round(float(top1), 4),
            "top1_doc_id": final[0].doc_id if final else None,
            "match": getattr(final[0], "match", "vector") if final else None,
            "latency_ms": total_ms,
            "candidate_ms": cand_ms,
            "finalize_ms": final_ms,
        })

    return records, metrics.summarize_mode(records, mode)


def check_against_production(rag: RAGService, db, question: str, mode: str) -> bool:
    """一致性自检：评测得到的 final 与生产 search() 的返回是否一致（doc_id 序列）。"""
    cfg = MODES[mode]
    cands = _candidates(rag, db, question, cfg)
    mine = [c.doc_id for c in _finalize(cands, cfg, question)]
    prod = [c.doc_id for c in rag.search(db, question, use_rerank=cfg["rerank"])]
    return mine == prod
