# -*- coding: utf-8 -*-
"""端到端评测可视化（30 题）：结果分布、分题型、引用与延迟。"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = r"D:\文本需求拆分\.tmp\e2e_run"
summary = json.load(open(os.path.join(OUT_DIR, "summary.json"), encoding="utf-8"))
cases = [json.loads(l) for l in open(os.path.join(OUT_DIR, "per_case.jsonl"), encoding="utf-8") if l.strip()]

fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=120)
fig.suptitle(f"端到端评测：{summary['n']} 题真实生成（DeepSeek + 精排 + 严格 grounding）",
             fontsize=16.5, fontweight="bold", y=0.98)

# ① 结果构成
ax = axes[0][0]
a = summary["answerable"]
r = summary["refusal"]
parts = [
    ("答对", a["correct"], "#2E7D32"),
    ("答错", a["wrong"], "#C62828"),
    ("误拒（能答却拒）", a["false_refusal"], "#EF6C00"),
    ("正确拒答", r["refused"], "#1565C0"),
    ("该拒未拒", r["not_refused"], "#B71C1C"),
]
parts = [p for p in parts if p[1] > 0]
vals = [p[1] for p in parts]
labels = [f"{p[0]}\n{p[1]}" for p in parts]
colors = [p[2] for p in parts]
ax.bar(range(len(vals)), vals, color=colors)
for i, v in enumerate(vals):
    ax.text(i, v + 0.25, str(v), ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("题数", fontsize=11)
ax.set_ylim(0, max(vals) + 3)
ax.set_title(f"① 结果构成：整体通过率 {summary['overall_pass_pct']:.1f}%", fontsize=12.5, fontweight="bold")
ax.grid(axis="y", alpha=0.25)

# ② 分题型
ax = axes[0][1]
bt = r["by_type"]
groups = [
    ("可答题", a["n"], a["correct"]),
    ("近似无答案", bt.get("no_answer_near", {}).get("n", 0), bt.get("no_answer_near", {}).get("refused", 0)),
    ("无关问题", bt.get("no_answer_far", {}).get("n", 0), bt.get("no_answer_far", {}).get("refused", 0)),
    ("诱导性", bt.get("lure", {}).get("n", 0), bt.get("lure", {}).get("refused", 0)),
]
groups = [g for g in groups if g[1] > 0]
names = [g[0] for g in groups]
tot = [g[1] for g in groups]
ok = [g[2] for g in groups]
rates = [o / t * 100 for o, t in zip(ok, tot)]
bars = ax.bar(range(len(groups)), rates, color=["#2E7D32" if n == "可答题" else "#1565C0" for n in names])
for b, o, t in zip(bars, ok, tot):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{o}/{t}", ha="center", fontsize=10)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(names, fontsize=10.5)
ax.set_ylim(0, 115)
ax.set_ylabel("通过率 (%)", fontsize=11)
ax.set_title("② 分题型：可答 / 该拒 / 诱导", fontsize=12.5, fontweight="bold")
ax.grid(axis="y", alpha=0.25)

# ③ 引用与 grounding
ax = axes[1][0]
cats = ["有引用", "引用正确", "grounding 告警"]
vals = [a["with_citation"], a["citation_correct"], summary.get("grounding_flagged", 0)]
colors = ["#4A78B5", "#2E7D32", "#C62828"]
bars = ax.bar(range(len(cats)), vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(v), ha="center", fontsize=11)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, fontsize=10.5)
ax.set_ylim(0, max(a["n"], max(vals)) + 3)
ax.set_ylabel("题数", fontsize=11)
ax.set_title(f"③ 溯源：{a['with_citation']}/{a['n']} 带引用，{a['citation_correct']}/{a['n']} 引用正确",
             fontsize=12.5, fontweight="bold")
ax.grid(axis="y", alpha=0.25)

# ④ 延迟分布
ax = axes[1][1]
el = [c["elapsed_s"] for c in cases]
idx = list(range(1, len(el) + 1))
ax.bar(idx, el, color="#7A54B0")
ax.axhline(st.mean(el), color="#C62828", ls="--", lw=1.6, label=f"均值 {st.mean(el):.1f}s")
ax.set_xlabel("题号", fontsize=11)
ax.set_ylabel("耗时 (s)", fontsize=11)
ax.set_title(f"④ 每题耗时：均值 {st.mean(el):.1f}s，最大 {max(el):.1f}s", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout(rect=(0, 0, 1, 0.95))
out = r"D:\文本需求拆分\docs\端到端评测.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120, facecolor="white")
print("saved:", out)
