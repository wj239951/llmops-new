# -*- coding: utf-8 -*-
"""语料分布 + 召回漏斗。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(16, 8.4), dpi=120)
fig.suptitle("语料分布与召回漏斗：196 篇文档 → 1098 候选 → 143 次命中", fontsize=17,
             fontweight="bold", y=0.97)

# ── 面板 1：文档长度分布 ──
docs = json.load(open(r"D:\文本需求拆分\docs\doc_lengths.json", encoding="utf-8"))
lengths = np.array(docs["lengths"])
ax = axes[0]
ax.hist(lengths, bins=24, color="#4A78B5", edgecolor="white")
med = float(np.median(lengths))
ax.axvline(med, color="#C62828", ls="--", lw=1.8, label=f"中位数 {med:.0f} 字")
ax.axvspan(400, 700, color="#C4832B", alpha=0.12)
ax.text(700, ax.get_ylim()[1] * 0.82, "主要切分区间", fontsize=9, color="#9A6A20")
ax.set_xlabel("单篇文档长度（字符）", fontsize=11)
ax.set_ylabel("文档数", fontsize=11)
ax.set_title(f"① 语料分布：{docs['n']} 篇，平均 {lengths.mean():.0f} 字", fontsize=13,
             fontweight="bold", pad=12)
ax.legend(fontsize=9.5)
ax.grid(axis="y", alpha=0.25)

# ── 面板 2：检索漏斗 ──
ax = axes[1]
stages = [
    ("候选片段", 1098, "183 题 × 候选 6 条"),
    ("精排后过阈值", 144, "0.92 阈值，通过率 13.1%"),
    ("命中正确文档", 143, "占过阈值 99.3%"),
]
colors = ["#7A54B0", "#B4514F", "#2E7D32"]
ax.set_xlim(0, 10)
ax.set_ylim(-0.2, 4.4)
ax.axis("off")
ax.set_title("② 检索漏斗：每个问题走一遍", fontsize=13, fontweight="bold", pad=12)
vmax = max(v for _, v, _ in stages)
for i, (label, val, sub) in enumerate(stages):
    y = 3.4 - i * 1.15
    w = 8.4 * (0.20 + 0.80 * (val / vmax) ** 0.55)
    x0 = 5 - w / 2
    ax.add_patch(Polygon([[x0, y + 0.44], [x0 + w, y + 0.44], [x0 + w * 0.93, y - 0.44],
                          [x0 + w * 0.07, y - 0.44]], closed=True,
                         facecolor=colors[i], edgecolor="white", linewidth=2))
    ax.text(5, y + 0.12, f"{label}  {val:,}", ha="center", va="center", fontsize=13,
            fontweight="bold", color="white")
    ax.text(5, y - 0.24, sub, ha="center", va="center", fontsize=9.5, color="white")
    if i < len(stages) - 1:
        ax.annotate("", xy=(5, y - 0.70), xytext=(5, y - 0.48),
                    arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.6))

ax.text(0.3, 0.35, "不可答题 35 个：仅 1 个被误接受（2.9%）\n"
                   "正确文档排名：rank1 116 题 · rank2 29 · rank3 2 · rank4 1\n"
                   "每题最终进入 prompt 的片段 ≤3 段（共 549 段容量）",
        fontsize=9.2, color="#555", linespacing=1.7)

plt.tight_layout(rect=(0, 0, 1, 0.93))
out = r"D:\文本需求拆分\docs\语料与召回漏斗.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120, facecolor="white")
print("saved:", out)
