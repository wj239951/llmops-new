# -*- coding: utf-8 -*-
"""合成文档页数据（PyMuPDF 版）：语料 -> PDF 页面 -> PNG + 严格对齐的 Markdown 真值。

为什么不用浏览器：本机沙箱禁止子进程建命名管道，Edge/Chrome headless 起不来；
PyMuPDF 的 Story 引擎可离线排版 HTML 子集，且表格由我们手绘（保证有框线）。

用法：
    python synth_pages_v2.py --n 200 --out D:\文本需求拆分\synth
"""
import argparse
import json
import os
import random
import re

import pymupdf

HEADER_TEXTS = ["设备管理部 · 内部资料", "新乡工厂 · 技术文件", "受控文件 · 请勿外传", "版本 V2.1"]
FOOTER_TEXTS = ["第 {n} 页 共 {t} 页", "审阅：设备科", "2026 年 11 月", "文件编号：SB-{code}"]
HEAD_WORDS = ["项目", "参数", "数值", "单位", "检查项", "标准", "周期", "备注", "状态", "方法"]


NUM = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9\-]{2,10})\s*[为是]?\s*(\d+(?:\.\d+)?)\s*"
                 r"(千牛|毫米|千瓦|克|摄氏度|伏|安|分钟|秒|小时|吨|千克|米|台|个|次|%|兆帕)")
BAD_NAME = {"低于", "不高于", "高于", "超过", "大于", "小于", "为", "是", "达到", "约", "不超过",
            "偏差超过正负", "与设定值偏差超过正负"}


def split_units(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    titles, paras, items, phrases, tuples, md_rows = [], [], [], [], [], []
    for l in lines:
        # 语料里的 Markdown 表格行不能当正文（否则渲出裸的 | 管道符）
        if l.startswith("|") or l.count("|") >= 3:
            cells = [c.strip() for c in l.strip("|").split("|")]
            cells = [c for c in cells if c and not re.fullmatch(r":?-{2,}:?", c)]
            for c in cells:
                if 2 <= len(c) <= 14:
                    phrases.append(c)
            if len(cells) >= 3:
                md_rows.append(cells)
            continue
        if re.match(r"^[一二三四五六七八九十]+、", l) or (re.match(r"^\d+\.\s*\S", l) and len(l) < 20):
            titles.append(l)
        elif re.match(r"^\d+[\.、]", l) or l.startswith("- "):
            items.append(re.sub(r"^(\d+[\.、]|-)\s*", "", l))
        else:
            paras.append(l)
        for seg in re.split(r"[，。、；：\s]+", l):
            seg = seg.strip("（）()《》\"'“”·-— .")
            if (2 <= len(seg) <= 14 and re.search(r"[\u4e00-\u9fa5]", seg)
                    and not re.fullmatch(r"[\d\.\-]+", seg) and not seg.endswith(" ")):
                phrases.append(seg)
        for m in NUM.finditer(l):
            name = m.group(1).lstrip("为是的")
            if name in BAD_NAME or len(name) < 2:
                continue
            tuples.append((name, m.group(2), m.group(3)))
    return titles, paras, items, phrases, tuples, md_rows


def build_page(rng, corpus, idx, total):
    docs = rng.sample(corpus, k=min(len(corpus), rng.randint(3, 4)))
    T, P, I, F, TU, MR = [], [], [], [], [], []
    for d in docs:
        t, p, i, f, tu, mr = split_units(d["content"])
        T += t; P += p; I += i; F += f; TU += tu; MR += mr
    if not P:
        return None

    columns = 1  # 双栏待修复后恢复
    margin = rng.choice([42, 50, 57, 64])
    font_size = rng.choice([10.0, 10.5, 11.0, 12.0])
    line_h = rng.choice([1.5, 1.65, 1.8])
    title = rng.choice(T) if T and rng.random() < 0.75 else re.sub(r"^\d+_|\.txt$", "", rng.choice(docs)["title"])

    # 正文块（按阅读顺序）
    blocks = []
    budget = rng.randint(1000, 1700)
    used, total = set(), 0
    guard = 0
    while total < budget and guard < 40:
        guard += 1
        cand = P[rng.randrange(len(P))]
        if cand in used:
            continue
        used.add(cand)
        blocks.append(("p", cand))
        total += len(cand)
        if T and rng.random() < 0.3:
            blocks.append(("h2", rng.choice(T)))
    if I and rng.random() < 0.55:
        n = rng.randint(2, min(5, len(I)))
        blocks.append(("ul", rng.sample(I, k=n)))

    # 表格（0-2 个），记录位置：top / bottom
    tables = []
    for _ in range(rng.choice([0, 1, 1, 2])):
        style = rng.random()
        if len(MR) >= 2 and style < 0.35:                     # 直接复用语料里的真实表格行
            ncol = min(len(MR[0]), 4)
            k = rng.randint(2, min(5, len(MR)))
            rows = [r[:ncol] for r in rng.sample(MR, k)]
            head = ["项目", "参数", "单位", "备注"][:ncol]
            nrow = len(rows)
        elif len(TU) >= 3 and style < 0.62:                      # 参数表：项目 / 数值 / 单位
            k = rng.randint(3, min(6, len(TU)))
            rows = [[a, b, c] for a, b, c in rng.sample(TU, k)]
            head, nrow = ["项目", "数值", "单位"], len(rows)
        elif I and style < 0.85:                     # 检查表：检查项 / 标准 / 周期
            k = rng.randint(2, min(5, len(I)))
            rows = []
            for it in rng.sample(I, k):
                std = rng.choice(TU)[1] + rng.choice(TU)[2] if TU else "符合"
                rows.append([it[:16], std, rng.choice(["每班", "每日", "每周", "每月", "每季"])])
            head, nrow = ["检查项", "标准", "周期"], len(rows)
        else:                                        # 杂项表
            ncol = rng.randint(2, 4)
            nrow = rng.randint(2, 5)
            head = rng.sample(HEAD_WORDS, k=ncol)
            rows = [[(rng.choice(F) if F else "—") for _ in range(ncol)] for _ in range(nrow)]
        tables.append({"head": head, "rows": rows,
                       "pos": rng.choice(["top", "bottom"]), "nrow": nrow})

    return {
        "idx": idx, "total": total, "columns": columns, "margin": margin,
        "font_size": font_size, "line_h": line_h, "title": title,
        "blocks": blocks, "tables": tables,
        "header": rng.random() < 0.5, "footer": rng.random() < 0.6,
    }


def esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def blocks_html(blocks, fs, lh):
    out = []
    for kind, val in blocks:
        if kind == "p":
            out.append(f'<p style="margin:0 0 6pt 0;text-align:justify;font-size:{fs}pt;line-height:{lh}">{esc(val)}</p>')
        elif kind == "h2":
            out.append(f'<p style="margin:9pt 0 3pt 0;font-size:{fs+1.5}pt;font-weight:bold">{esc(val)}</p>')
        elif kind == "ul":
            for it in val:
                out.append(f'<p style="margin:0 0 2pt 0;font-size:{fs}pt;line-height:{lh}">• {esc(it)}</p>')
    return "".join(out)


def draw_table(page, rect, tbl, fs):
    ncol = len(tbl["head"])
    nrow = tbl["nrow"] + 1
    row_h = 18.0
    widths = []
    for c in range(ncol):
        lens = [len(tbl["head"][c])] + [len(row[c]) for row in tbl["rows"]]
        widths.append(max(lens) + 3)
    tot = sum(widths)
    widths = [rect.width * w / tot for w in widths]
    page.draw_rect(rect, color=(0.6, 0.6, 0.6), width=0.7)
    page.draw_rect(pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + row_h),
                   color=None, fill=(0.92, 0.92, 0.92))
    x = rect.x0
    for c, w in enumerate(widths):
        if c:
            page.draw_line(pymupdf.Point(x, rect.y0), pymupdf.Point(x, rect.y0 + row_h * nrow),
                           color=(0.6, 0.6, 0.6), width=0.7)
        x += w
    for r in range(nrow):
        y = rect.y0 + row_h * (r + 1)
        if r < nrow - 1:
            page.draw_line(pymupdf.Point(rect.x0, y), pymupdf.Point(rect.x1, y),
                           color=(0.6, 0.6, 0.6), width=0.7)

    def put(cx, cy, text, w, bold=False, size=fs - 1.5):
        max_chars = max(2, int((w - 6) / (size * 0.62)))
        try:
            page.insert_text(pymupdf.Point(cx + 3, cy + 12), str(text)[:max_chars],
                             fontsize=size, fontname="china-s",
                             color=(0, 0, 0))
        except Exception:
            pass
    x = rect.x0
    for c, w in enumerate(widths):
        put(x, rect.y0, tbl["head"][c], w, bold=True)
        for r, row in enumerate(tbl["rows"]):
            put(x, rect.y0 + row_h * (r + 1), row[c], w)
        x += w


def render(page_spec, out_dir, dpi):
    p = page_spec
    W, H = 595, 842
    m = p["margin"]
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    fs, lh = p["font_size"], p["line_h"]

    cur_y = m
    if p["header"]:
        page.insert_text(pymupdf.Point(m, m - 6), random.choice(HEADER_TEXTS),
                         fontsize=fs - 2, fontname="china-s", color=(0.4, 0.4, 0.4))
        page.draw_line(pymupdf.Point(m, m + 2), pymupdf.Point(W - m, m + 2), color=(0.75, 0.75, 0.75), width=0.5)
        cur_y = m + 12

    # 标题
    title_rect = pymupdf.Rect(m, cur_y, W - m, cur_y + 34)
    spare, _ = page.insert_htmlbox(
        title_rect, f'<div style="text-align:center;font-size:{fs+5}pt;font-weight:bold">{esc(p["title"])}</div>')
    cur_y += 40

    top_tables = [t for t in p["tables"] if t["pos"] == "top"]
    bot_tables = [t for t in p["tables"] if t["pos"] == "bottom"]

    for t in top_tables:
        h = 18.0 * (t["nrow"] + 1)
        draw_table(page, pymupdf.Rect(m, cur_y, W - m, cur_y + h), t, fs)
        cur_y += h + 10

    bottom_reserve = sum(18.0 * (t["nrow"] + 1) + 10 for t in bot_tables)
    body_bottom = H - m - bottom_reserve - (18 if p["footer"] else 0)
    spare = 0.0
    if p["columns"] == 2:
        gap = 8
        col_w = (W - 2 * m - gap) / 2
        half = (len(p["blocks"]) + 1) // 2
        left, right = p["blocks"][:half], p["blocks"][half:]
        spares = []
        for i, part in enumerate((left, right)):
            r = pymupdf.Rect(m + i * (col_w + gap), cur_y, m + i * (col_w + gap) + col_w, body_bottom)
            sp, _ = page.insert_htmlbox(r, f'<div>{blocks_html(part, fs, lh)}</div>')
            spares.append(sp)
        spare = max(spares)
    else:
        spare, _ = page.insert_htmlbox(pymupdf.Rect(m, cur_y, W - m, body_bottom),
                                       f'<div>{blocks_html(p["blocks"], fs, lh)}</div>')

    y = cur_y + max(0.0, (body_bottom - cur_y) - spare) + 10
    for t in bot_tables:
        h = 18.0 * (t["nrow"] + 1)
        draw_table(page, pymupdf.Rect(m, y, W - m, y + h), t, fs)
        y += h + 10

    if p["footer"]:
        txt = random.choice(FOOTER_TEXTS).format(n=p["idx"] + 1, t=p["total"], code=f"{p['idx']+1:03d}")
        page.draw_line(pymupdf.Point(m, H - m + 6), pymupdf.Point(W - m, H - m + 6), color=(0.75, 0.75, 0.75), width=0.5)
        page.insert_text(pymupdf.Point(m, H - m + 20), txt, fontsize=fs - 2, fontname="china-s", color=(0.4, 0.4, 0.4))

    name = f"synth_{p['idx']:04d}.png"
    page.get_pixmap(dpi=dpi).save(os.path.join(out_dir, name))
    doc.close()
    return name


def markdown_of(p):
    out = ["# " + p["title"]]
    for kind, val in p["blocks"]:
        if kind == "p":
            out.append(val)
        elif kind == "h2":
            out.append("## " + val)
        elif kind == "ul":
            out.append("\n".join("- " + x for x in val))
    for t in p["tables"]:
        th = "".join(f"<td>{esc(h)}</td>" for h in t["head"])
        trs = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in t["rows"])
        out.append(f"<table><tr>{th}</tr>{trs}</table>")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=r"D:\文本需求拆分\synth")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    corpus = json.load(open(r"D:\文本需求拆分\.tmp\corpus.json", encoding="utf-8"))
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in range(args.n):
        spec = build_page(rng, corpus, i, args.n)
        if not spec:
            continue
        name = render(spec, img_dir, args.dpi)
        rows.append({"image": name, "target": markdown_of(spec), "columns": spec["columns"],
                     "tables": len(spec["tables"]), "font_size": spec["font_size"]})
    json.dump(rows, open(os.path.join(args.out, "train.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"生成 {len(rows)} 页 -> {img_dir}")
    print("平均目标长度:", sum(len(r["target"]) for r in rows) // max(len(rows), 1))
    print("含表格页数:", sum(1 for r in rows if r["tables"] > 0))


if __name__ == "__main__":
    main()
