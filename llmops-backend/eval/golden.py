# -*- coding: utf-8 -*-
"""黄金评测集：加载、校验、按类型切分。

评测集放在 eval/golden/ 下随仓库走（不再依赖 experiments/ 的历史文件），
保证 CI 与本地跑的是同一份题。
"""
from __future__ import annotations

import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

SETS = {
    "qa": "qa.jsonl",        # 45 题主集：exact / paraphrase / no_answer_far / no_answer_near
    "hard": "hard.jsonl",    # 硬样本：精确型号/编号，向量召回易漏
    "token": "token.jsonl",  # 搜索框式短查询：只输型号/编号
    "bank": "bank.jsonl",    # 183 题：196 篇 / 338 片段语料上的主集（docs/13）
}

# 需要真实生成才能评测的集子：只参与格式校验，不参与 run 的检索模式。
# e2e 走 `python -m eval e2e`（生成 + 拒答/引用判定，docs/18）。
EXTRA_SETS = {
    "e2e": "e2e.jsonl",      # 30 题：可回答 10 / 近域 10 / 远域 5 / 常识诱导 5
}

REQUIRED_FIELDS = {"id", "question", "expected_doc_ids"}
ANSWERABLE_TYPES = {"exact", "paraphrase", "exact_token", "token_query", "answerable"}


class GoldenError(ValueError):
    """评测集格式错误：宁可启动即失败，也不要跑出一份口径不明的报告。"""


def load_set(name: str = "qa", path: Path | None = None) -> list[dict]:
    if path is None:
        if name in SETS:
            path = GOLDEN_DIR / SETS[name]
        elif name in EXTRA_SETS:
            path = GOLDEN_DIR / EXTRA_SETS[name]
        else:
            raise GoldenError(
                f"未知评测集 {name}，可选：{', '.join(list(SETS) + list(EXTRA_SETS))}"
            )
    if not path.exists():
        raise GoldenError(f"评测集不存在：{path}")

    questions: list[dict] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldenError(f"{path.name}:{lineno} 不是合法 JSON：{exc}") from exc

            missing = REQUIRED_FIELDS - set(item)
            if missing:
                raise GoldenError(f"{path.name}:{lineno} 缺少字段：{sorted(missing)}")
            if item["id"] in seen:
                raise GoldenError(f"{path.name}:{lineno} id 重复：{item['id']}")
            if not isinstance(item["expected_doc_ids"], list):
                raise GoldenError(f"{path.name}:{lineno} expected_doc_ids 必须是列表")

            seen.add(item["id"])
            item.setdefault("type", "exact")
            item["answerable"] = bool(item["expected_doc_ids"])
            item["source_set"] = path.stem
            questions.append(item)

    if not questions:
        raise GoldenError(f"{path} 里没有任何题目")
    return questions


def stats(questions: list[dict]) -> dict:
    by_type: dict[str, int] = {}
    for q in questions:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    return {
        "total": len(questions),
        "answerable": sum(1 for q in questions if q["answerable"]),
        "no_answer": sum(1 for q in questions if not q["answerable"]),
        "by_type": by_type,
    }
