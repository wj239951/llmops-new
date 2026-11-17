# -*- coding: utf-8 -*-
"""数字级 grounding 校验：回答里的每个数值，都应当在喂进去的片段原文里出现过。

为什么加这一层（docs/18 的实测）：
- 严格提示词能管住"没依据就拒答"，但管不住"模型自己补一个数值"；
- 生成后过一遍数字，能把"编出来的数字"挡在返回之前，成本是一次正则，不掉延迟。

边界（必须说清楚）：
- 它查的是**数字来源**，不是**语义归属**。docs/18 的 E13（把"额定载重 50 千克每米"
  答成"模具重量上限 50 千克"）里，50 确实在片段里，本校验不会报——那属于属性错配，
  需要另一套校验（属性词是否出现在片段中），不在本模块范围。
- 只校验两位及以上、或带小数点的数字：个位数（"第 3 步"、列表序号）噪音太大。
"""
from __future__ import annotations

import re

# 引用标注里的编号不是事实数值，先剔掉
_CITE_RE = re.compile(r"【[^】]*】")
_NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def strip_citations(text: str) -> str:
    return _CITE_RE.sub(" ", text or "")


def extract_numbers(text: str) -> list[str]:
    """抽出值得校验的数字（两位以上或带小数点），去掉千分位逗号。"""
    nums: list[str] = []
    for m in _NUM_RE.finditer(strip_citations(text)):
        raw = m.group(0).replace(",", "")
        if len(raw.replace(".", "")) >= 2 or "." in raw:
            nums.append(raw)
    return nums


def check_numbers(answer: str, context: str) -> dict:
    """返回 {ok, numbers, missing}。missing 是回答里有、片段里找不到的数字。"""
    numbers = extract_numbers(answer)
    if not numbers:
        return {"ok": True, "numbers": [], "missing": []}
    haystack = (context or "").replace(",", "")
    missing: list[str] = []
    for n in numbers:
        if n not in haystack:
            missing.append(n)
    # 去重但保序
    missing = list(dict.fromkeys(missing))
    return {"ok": not missing, "numbers": numbers, "missing": missing}
