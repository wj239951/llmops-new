# -*- coding: utf-8 -*-
"""评测包的公共入口。

设计目标：一条命令跑出「可写进 README 的量化表」，并能在指标退化时让 CI 失败。
"""
from eval.metrics import pct, percentile, reciprocal_rank, recall_at_k, first_hit_rank, summarize_mode

__all__ = [
    "pct",
    "percentile",
    "reciprocal_rank",
    "recall_at_k",
    "first_hit_rank",
    "summarize_mode",
]
