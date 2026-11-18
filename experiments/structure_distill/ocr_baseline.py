# -*- coding: utf-8 -*-
"""OCR 两段式基线：RapidOCR 读字 + 简单版面排序 -> Markdown。

目的不是刷分，而是给出「OCR 引擎 + 规则排版」这条工程路线的真实下限，用来判断
端到端微调（TEDS 0.23 / TextEdit 0.36）到底有没有竞争力。
"""
import argparse
import glob
import os

import sys
sys.path.append(r"D:\anaconda\Lib\site-packages")
from PIL import Image

from rapidocr_onnxruntime import RapidOCR

eng = RapidOCR()


def group_lines(boxes, tol=0.6):
    """把 OCR 结果按行聚合：同一行 = 框中心 y 相差 < 行高*tol。"""
    items = []
    for box, text, score in boxes:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
                      "cy": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys), "t": text, "s": score})
    items.sort(key=lambda d: (d["cy"], d["x0"]))
    lines, cur, cur_cy = [], [], None
    for it in items:
        if cur_cy is None or abs(it["cy"] - cur_cy) <= max(it["h"], 8) * tol:
            cur.append(it)
            cur_cy = sum(x["cy"] for x in cur) / len(cur)
        else:
            lines.append(sorted(cur, key=lambda d: d["x0"]))
            cur, cur_cy = [it], it["cy"]
    if cur:
        lines.append(sorted(cur, key=lambda d: d["x0"]))
    return lines


def to_markdown(lines):
    """行 -> 段落：行高变化或缩进变化时断段；最长的首行当标题。"""
    if not lines:
        return ""
    rows = [("".join(x["t"] for x in ln), min(x["x0"] for x in ln), max(x["y1"] for x in ln) - min(x["y0"] for x in ln)) for ln in lines]
    heights = sorted(r[2] for r in rows)
    med = heights[len(heights) // 2] if heights else 10
    title_idx = max(range(len(rows)), key=lambda i: rows[i][2]) if rows else 0
    out, buf = [], []
    for i, (txt, x0, h) in enumerate(rows):
        if i == title_idx and h > med * 1.25 and not out:
            out.append("# " + txt.strip())
            continue
        if buf and (abs(h - med) > med * 0.35 or x0 > min(r[1] for r in rows) + med * 3):
            out.append("".join(buf).strip())
            buf = []
        buf.append(txt)
    if buf:
        out.append("".join(buf).strip())
    return "\n\n".join(x for x in out if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=r"D:\bench\omnidocbench\images")
    ap.add_argument("--out", default=r"D:\文本需求拆分\bench_eval\pred_ocr")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.images, "*")))
    print("pages:", len(files), flush=True)
    for i, f in enumerate(files, 1):
        try:
            im = Image.open(f).convert("RGB")
            if max(im.size) > 2000:
                im.thumbnail((2000, 2000), Image.LANCZOS)
            import numpy as np
            res, _ = eng(np.array(im))
            boxes = res or []
        except Exception as e:
            print(f"[{i}/{len(files)}] FAIL {e}", flush=True)
            boxes = []
        md = to_markdown(group_lines(boxes)) if boxes else ""
        stem = os.path.splitext(os.path.basename(f))[0]
        open(os.path.join(args.out, stem + ".md"), "w", encoding="utf-8").write(md)
        print(f"[{i}/{len(files)}] {stem[:36]} {len(boxes)} boxes -> {len(md)} chars", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
