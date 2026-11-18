# -*- coding: utf-8 -*-
"""MarathonDocBench -> 训练用 JSONL（真实文档，中文页）。

标注结构：list[100] 每项 {page_info:{image_path, images_list}, layout_dets:[{page_id, category_type, text, order, ignore}]}
表格的 HTML 存在 text 字段里。

输出：{image, target, src, cjk}
"""
import argparse
import collections
import json
import os
import re

SKIP = {"header", "footer", "page_number", "figure", "chart_mask", "abandon",
        "equation_isolated", "equation_semantic", "equation_caption",
        "formula_number", "code_algorithm", "code_algorithm_caption"}


def cjk_ratio(s):
    if not s:
        return 0.0
    c = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return c / max(1, len(s))


def md_of(blocks):
    blocks = [b for b in blocks if b.get("category_type") not in SKIP and not b.get("ignore")]
    blocks.sort(key=lambda b: (b.get("order") is None, b.get("order") or 0))
    out = []
    for b in blocks:
        c, t = b.get("category_type"), (b.get("text") or "").strip()
        if not t:
            continue
        if c == "title":
            out.append("# " + t)
        elif c == "table":
            out.append(t if t.lstrip().startswith("<table") else t)
        else:
            out.append(t)
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=r"D:\文本需求拆分\.tmp\MarathonDocBench.json")
    ap.add_argument("--images-root", default=r"D:\文本需求拆分\.tmp\marathon")
    ap.add_argument("--out", default=r"D:\文本需求拆分\bench_eval\marathon_train.json")
    ap.add_argument("--min-cjk", type=float, default=0.5)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-image", action="store_true")
    args = ap.parse_args()

    docs = json.load(open(args.json, encoding="utf-8"))
    rows = []
    for doc in docs:
        info = doc.get("page_info") or {}
        imgs = info.get("images_list") or []
        by_page = collections.defaultdict(list)
        for b in doc.get("layout_dets", []):
            by_page[b.get("page_id")].append(b)
        for pid, blocks in by_page.items():
            txt = "".join((b.get("text") or "") for b in blocks)
            if len(txt) < args.min_chars or cjk_ratio(txt) < args.min_cjk:
                continue
            md = md_of(blocks)
            if len(md) < args.min_chars:
                continue
            # page_id 从 1 开始，images_list 同序
            img = imgs[pid - 1] if 0 < pid <= len(imgs) else None
            if not img:
                continue
            local = os.path.join(args.images_root, img.replace("/", os.sep))
            if args.require_image and not os.path.exists(local):
                continue
            rows.append({"image": img, "local": local, "target": md,
                         "src": "marathon", "cjk": round(cjk_ratio(txt), 3)})
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    with_tbl = sum(1 for r in rows if "<table" in r["target"])
    print(f"pages: {len(rows)} | 含表格: {with_tbl} | 平均长度: {sum(len(r['target']) for r in rows)//max(1,len(rows))}")
    print("sample:", rows[0]["target"][:180].replace("\n", " / ") if rows else "-")


if __name__ == "__main__":
    main()
