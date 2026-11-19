# -*- coding: utf-8 -*-
"""一次提问的时序图：从点击发送到流式结束，标注实测耗时。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

W, H = 176, 136
fig, ax = plt.subplots(figsize=(22, 16), dpi=110)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")

PARTS = [
    ("浏览器\nChat Studio", 12, "#4A78B5"),
    ("FastAPI\n:8000", 34, "#3F8A55"),
    ("ChatService\n编排", 56, "#C4832B"),
    ("清洗 / OCR", 78, "#C4832B"),
    ("改写 + 术语归一", 99, "#C4832B"),
    ("检索\nChroma+BM25", 119, "#7A54B0"),
    ("精排\nSiliconFlow", 139, "#B4514F"),
    ("LLM\nDeepSeek", 158, "#B4514F"),
]
TOP, BOTTOM = 30, 116
for name, x, color in PARTS:
    ax.add_patch(FancyBboxPatch((x - 8, 20), 16, 8, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=color, edgecolor=color))
    ax.text(x, 24, name, ha="center", va="center", fontsize=9.4, color="white", fontweight="bold")
    ax.plot([x, x], [TOP - 2, BOTTOM], color="#bbb", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)

ax.text(W / 2, 6, "一次提问的完整时序（DeepSeek 云端 + 精排，实测样例）", ha="center", fontsize=17,
        fontweight="bold", color="#141414")
ax.text(W / 2, 12.5, "数字为实测耗时：检索 206 ms · 精排 494 ms · LLM 13.0 s · 总计 13.7 s",
        ha="center", fontsize=10, color="#666")


def msg(y, x0, x1, text, ls="-", color="#444", note="", note_x=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3, color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=3))
    mid = (x0 + x1) / 2
    ax.text(mid, y - 2.2, text, ha="center", va="bottom", fontsize=8.6, color="#333")
    if note:
        ax.text(note_x if note_x else max(x0, x1) + 1, y + 0.4, note, ha="left", va="center",
                fontsize=8.2, color="#8a8a8a")


def step(y, n, label):
    ax.text(2.4, y, f"{n}", ha="center", va="center", fontsize=9.5, color="#fff",
            bbox=dict(boxstyle="circle,pad=0.22", facecolor="#555", edgecolor="none"))
    ax.text(5.2, y, label, ha="left", va="center", fontsize=8.4, color="#8a8a8a")


rows = [
    (34, 1, "发送", 12, 34, "POST /api/chat/stream  {query, image_base64?}"),
    (41, 2, "鉴权", 34, 56, "Bearer 口令校验 → ChatService"),
    (48, 3, "清洗", 56, 78, "规则/模型清洗（clean_ms ≈ 0–40 ms）", "-"),
    (55, 4, "OCR", 56, 78, "图片可选：RapidOCR ≈ 1.6 s（无图跳过）", "-"),
    (62, 5, "改写", 56, 99, "术语归一 + 多轮改写（≈ 20 ms）", "-"),
    (69, 6, "检索", 56, 119, "dense + BM25 并行召回 → RRF 融合", "-"),
    (76, 7, "精排", 119, 139, "候选 6 条 → bge-reranker-v2-m3", "206 ms → 494 ms"),
    (83, 8, "门控", 139, 56, "分数 + 阈值（0.92）→ hit / weak_hit / no_hit", "-"),
    (90, 9, "生成", 56, 158, "grounded prompt → DeepSeek 流式", "≈ 13.0 s"),
    (97, 10, "推送", 158, 12, "SSE delta 增量 → 首字 ≈ 3.5 s", "-"),
    (103, 11, "落库", 56, 34, "写 chat_logs / 会话消息（异步）", "-"),
    (111, 12, "结束", 34, 12, "done 事件：answer + sources + timings", "-"),
]
for row in rows:
    y, n, label, x0, x1, text, note = (list(row) + [""])[:7]
    step(y, n, label)
    msg(y, x0, x1, text, note=note)

# 右侧耗时分布
bx, bw = 148, 26
ax.text(bx + bw / 2, 32, "耗时分布", ha="center", fontsize=11, fontweight="bold", color="#333")
bars = [("检索", 206, "#7A54B0"), ("精排", 494, "#B4514F"), ("LLM", 13024, "#3F8A55")]
total = sum(v for _, v, _ in bars)
y0 = 40
for i, (name, val, color) in enumerate(bars):
    w = bw * val / total
    ax.add_patch(Rectangle((bx, y0 + i * 9), max(w, 0.6), 5.5, facecolor=color, edgecolor="none"))
    ax.text(bx - 1, y0 + i * 9 + 2.8, name, ha="right", va="center", fontsize=9, color="#444")
    ax.text(bx + w + 1.2, y0 + i * 9 + 2.8, f"{val} ms" if val < 1000 else f"{val/1000:.1f} s",
            ha="left", va="center", fontsize=8.6, color="#666")
ax.text(bx + bw / 2, y0 + 3 * 9 + 1, f"总计 {total/1000:.1f} s", ha="center", fontsize=9.5,
        fontweight="bold", color="#333")

# 关键说明
notes = [
    ("LLM 占 95% 的时间", "13.0 s / 13.7 s；换成本地 7B 会更快但质量下降"),
    ("检索 + 精排 ≈ 0.7 s", "可接受；embedding 冷启动时会到 4 s"),
    ("无命中走短路", "不调 LLM，直接返回固定话术，总耗时 ≈ 3.5 s"),
    ("首字延迟 ≈ 3.5 s", "DeepSeek 首包；之后按 token 流式输出"),
]
ax.text(3, 122, "关键观察", fontsize=11, fontweight="bold", color="#333")
for i, (t1, t2) in enumerate(notes):
    ax.text(3, 127 + i * 4.6, f"· {t1}：{t2}", fontsize=8.8, color="#555")

out = r"D:\文本需求拆分\docs\请求时序图.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print("saved:", out)
