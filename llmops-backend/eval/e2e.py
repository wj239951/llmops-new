# -*- coding: utf-8 -*-
"""端到端拒答评测（生成链路）：从 eval/golden/e2e.jsonl 取题，走真实 ChatService.stream()。

判定口径：
- 可回答题：回答里是否含预期关键值（按数字比，抹平「乘 / ×」写法差异）；被拒答记为「误拒」；
  引用编号是否指向真正含答案的片段。
- 应拒答题（近域 / 远域 / 常识诱导）：是否拒答；没拒答记为「未拒答」。

入口：python -m eval e2e [--limit N]
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app.core.database import SessionLocal
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from eval import golden

DEFAULT_OUT = Path("eval_results") / "e2e"
REFUSAL_MARKERS = ("未收录", "没有检索到", "未检索到", "未找到", "没有找到", "无法回答")
CITE_RE = re.compile(r"【依据：知识片段\s*(\d+)\s*】")


def norm(s: str) -> str:
    return re.sub(r"[\s，。、；：（）()]+", "", s or "").lower()


def key_match(key: str, text: str) -> bool:
    """关键值比对：优先比数字，抹平「乘 / × / *」这类写法差异；没有数字时退回子串比。"""
    nums = re.findall(r"\d+(?:\.\d+)?", key or "")
    if nums:
        return all(n in (text or "") for n in nums)
    return bool(key) and norm(key) in norm(text)


def judge(case: dict, meta: dict, done: dict) -> dict:
    answer = done.get("answer") or ""
    rag_status = meta.get("rag_status")
    sources = meta.get("sources") or []
    refused = rag_status == "no_hit" or any(m in answer for m in REFUSAL_MARKERS)

    cited_title, cited_ok = None, None
    m = CITE_RE.search(answer)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(sources):
            cited_title = sources[idx - 1].get("title")
            key = case.get("expected_key")
            if case["type"] == "answerable" and key:
                cited_ok = key_match(key, sources[idx - 1].get("snippet") or "")

    if case["type"] == "answerable":
        key = case.get("expected_key") or ""
        if refused:
            verdict = "误拒"
        elif key and key_match(key, answer):
            verdict = "答对"
        else:
            verdict = "答错"
    else:
        verdict = "拒答" if refused else "未拒答"

    return {
        "verdict": verdict,
        "passed": verdict in ("答对", "拒答"),
        "refused": refused,
        "rag_status": rag_status,
        "sources": len(sources),
        "cited_title": cited_title,
        "cited_ok": cited_ok,
        "has_citation": bool(m),
        "answer": answer[:200],
    }


def run(limit: int = 0) -> tuple[list[dict], dict]:
    cases = golden.load_set("e2e")
    if limit:
        cases = cases[:limit]
    svc = ChatService()
    db = SessionLocal()
    rows: list[dict] = []
    try:
        for case in cases:
            t0 = time.time()
            events = list(svc.stream(db, ChatRequest(
                query=case["question"], use_rag=True, temperature=0.0,
            )))
            elapsed = round(time.time() - t0, 1)
            meta = next((e["data"] for e in events if e["event"] == "meta"), {})
            done = next((e["data"] for e in events if e["event"] == "done"), {})
            r = judge(case, meta, done)
            r.update({
                "id": case["id"], "type": case["type"], "question": case["question"],
                "expected_key": case.get("expected_key", ""), "elapsed_s": elapsed,
                "grounding": done.get("grounding"),
            })
            rows.append(r)
            mark = "✓" if r["passed"] else "✗"
            print(f"{mark} {r['id']:4s} {r['type']:16s} {r['verdict']:4s} "
                  f"{r['rag_status']:8s} src={r['sources']} {elapsed:5.1f}s", flush=True)
    finally:
        db.close()

    ans = [r for r in rows if r["type"] == "answerable"]
    ref = [r for r in rows if r["type"] != "answerable"]
    summary = {
        "n": len(rows),
        "answerable": {
            "n": len(ans),
            "correct": sum(r["verdict"] == "答对" for r in ans),
            "wrong": sum(r["verdict"] == "答错" for r in ans),
            "false_refusal": sum(r["verdict"] == "误拒" for r in ans),
            "with_citation": sum(r["has_citation"] for r in ans),
            "citation_correct": sum(1 for r in ans if r["cited_ok"] is True),
        },
        "refusal": {
            "n": len(ref),
            "refused": sum(r["refused"] for r in ref),
            "not_refused": sum(not r["refused"] for r in ref),
            "by_type": {
                t: {"n": sum(1 for r in ref if r["type"] == t),
                    "refused": sum(1 for r in ref if r["type"] == t and r["refused"])}
                for t in ("no_answer_near", "no_answer_far", "lure")
            },
        },
        "grounding_flagged": sum(1 for r in rows if (r.get("grounding") or {}).get("ok") is False),
        "overall_pass_pct": round(100 * sum(r["passed"] for r in rows) / len(rows), 1),
        "mean_elapsed_s": round(sum(r["elapsed_s"] for r in rows) / len(rows), 1),
    }
    return rows, summary


def main(limit: int = 0, out: str | None = None) -> int:
    rows, summary = run(limit=limit)
    out_dir = Path(out) if out else DEFAULT_OUT / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    a, rr = summary["answerable"], summary["refusal"]
    print("\n" + "=" * 60)
    print(f"可回答 {a['n']} 题：答对 {a['correct']} / 答错 {a['wrong']} / 误拒 {a['false_refusal']}"
          f"；带引用 {a['with_citation']}，引用正确 {a['citation_correct']}")
    print(f"应拒答 {rr['n']} 题：拒答 {rr['refused']} / 未拒答 {rr['not_refused']}")
    for t, d in rr["by_type"].items():
        print(f"    {t:16s} {d['refused']}/{d['n']}")
    print(f"数字校验触发 {summary['grounding_flagged']} 题；"
          f"整体通过率 {summary['overall_pass_pct']}%，平均 {summary['mean_elapsed_s']}s")
    print(f"产物：{out_dir}")
    return 0
