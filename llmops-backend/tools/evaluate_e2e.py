# -*- coding: utf-8 -*-
"""端到端拒答评测（兼容入口）。

实现已并入 `eval.e2e`（题库在 eval/golden/e2e.jsonl，可用 `python -m eval e2e` 跑）。
这个脚本保留是为了不破坏既有文档与习惯：参数与原来一致。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from eval.e2e import main  # noqa: E402


def cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="端到端拒答评测")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    return main(limit=args.limit, out=args.out)


if __name__ == "__main__":
    sys.exit(cli())
