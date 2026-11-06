# -*- coding: utf-8 -*-
"""LLM-as-judge：用模型给「回答是否答到点上」打 1–5 分。

为什么需要：检索指标只看「片段有没有召回」，不看「回答对不对」。
判官提示词固定、输出只解析数字，保证多次跑分可比。
"""
from __future__ import annotations

import re

from app.services.rag_service import RAGService

JUDGE_PROMPT = """你是严格的答案评审员。请根据【参考要点】给【模型回答】打分。

评分标准：
5 = 完全答对，关键信息齐全
4 = 基本答对，缺少次要细节
3 = 部分正确，但遗漏关键信息或有轻微错误
2 = 只沾边，实质内容不对
1 = 完全错误或答非所问

只输出一行：分数 + 一句话理由，格式如 `4 缺少更换周期`。

【问题】{question}
【参考要点】{reference}
【模型回答】{answer}
"""

_SCORE_RE = re.compile(r"^\s*([1-5])\b")


def parse_score(text: str) -> int | None:
    m = _SCORE_RE.match(text or "")
    return int(m.group(1)) if m else None


def generate_answer(rag: RAGService, llm, db, question: str, use_rerank: bool) -> dict:
    """走生产检索链路生成回答（不写 ChatLog，避免污染线上日志表）。"""
    docs = rag.search(db, question, use_rerank=use_rerank)
    context = rag.build_context(docs)
    query = question
    if context:
        query = (
            "请基于以下知识库内容回答问题。\n"
            "如果知识库内容不足，请说明不足，再结合通用知识补充。\n\n"
            f"{context}\n\n用户问题：{question}"
        )
    result = llm.chat(
        query=query,
        system_prompt="你是严谨的工业设备助手，回答准确、简洁，给出关键数值。",
        provider=None,
        model_name=None,
        temperature=0.1,
        top_p=0.9,
    )
    return {
        "answer": result.get("answer", ""),
        "total_tokens": int(result.get("total_tokens") or 0),
        "rag_status": "hit" if docs else "no_hit",
        "sources": [{"title": d.title, "score": round(d.score, 4)} for d in docs],
    }


def judge_answer(llm, question: str, reference: str, answer: str) -> tuple[int | None, str, int]:
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference or "（无参考要点，按常识判断）", answer=answer[:1500],
    )
    result = llm.chat(
        query=prompt,
        system_prompt="你是答案评审员，只输出评分和简短理由。",
        temperature=0.0,
        top_p=1.0,
    )
    text = (result.get("answer") or "").strip()
    return parse_score(text), text[:200], int(result.get("total_tokens") or 0)


def run_judge(questions: list[dict], db, use_rerank: bool = False, limit: int | None = None) -> dict:
    from app.services.llm_service import LLMService

    rag = RAGService()
    llm = LLMService()
    targets = [q for q in questions if q.get("answerable")]
    if limit:
        targets = targets[:limit]

    rows = []
    for q in targets:
        gen = generate_answer(rag, llm, db, q["question"], use_rerank)
        score, reason, judge_tokens = judge_answer(
            llm, q["question"], q.get("reference_answer", ""), gen["answer"],
        )
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "score": score,
            "reason": reason,
            "answer": gen["answer"][:300],
            "rag_status": gen["rag_status"],
            "gen_tokens": gen["total_tokens"],
            "judge_tokens": judge_tokens,
        })

    scored = [r for r in rows if r["score"] is not None]
    return {
        "evaluated": len(rows),
        "scored": len(scored),
        "mean_score": round(sum(r["score"] for r in scored) / len(scored), 2) if scored else 0.0,
        "score_ge4_pct": round(100.0 * sum(1 for r in scored if r["score"] >= 4) / len(scored), 1) if scored else 0.0,
        "unparsed_ids": [r["id"] for r in rows if r["score"] is None],
        "rows": rows,
    }
