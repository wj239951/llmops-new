# -*- coding: utf-8 -*-
"""成本核算：把 token 换算成钱，回答「这个功能值不值」。

单价随供应商调整，集中在这里改；Ollama 本地推理按 0 计（电费不计入）。
"""
from __future__ import annotations

# 单位：美元 / 百万 token（2025-11 官方价，deepseek-chat）
PRICING = {
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "ollama": {"input": 0.0, "output": 0.0},
}

USD_TO_CNY = 7.2


def estimate_usd(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    price = PRICING.get(model, PRICING["ollama"])
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def estimate_cny(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    return estimate_usd(prompt_tokens, completion_tokens, model) * USD_TO_CNY


def per_1k_queries_cny(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """按单次调用成本外推 1000 次问答的花费，便于写进 README 的成本表。"""
    return estimate_cny(prompt_tokens, completion_tokens, model) * 1000
