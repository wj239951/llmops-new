# -*- coding: utf-8 -*-
"""检索评测可视化：阈值标定 + 三模式消融 + 分题型 + 延迟。"""
import csv
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"C:\Users\lenovo\Desktop\llmops-new\llmops-backend\eval_results\baseline_bank"
rows = list(csv.DictReader(open(os.path.join(BASE, "per_query.csv"), encoding="utf-8")))
summary = json.load(open(os.path.join(BASE, "summary.json"), encoding="utf-8"))

fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=120)
fig.suptitle("RAG 检索评测：183 题（148 可答 / 35 不可答）", fontsize=17, fontweight="bold", y=0.98)

# ── 面板 1：阈值标定曲线（rerank）──
ax = axes[0][0]
rr = [r for r in rows if r["mode"] == "rerank"]
ans = [r for r in rr if r["answerable"] == "1"]
no = [r for r in rr if r["answerable"] == "0"]
ths = [i / 1000 for i in range(800, 1001)]
pass_rate, fa_rate = [], []
for t in ths:
    ok = sum(1 for r in ans if float(r["top1_score"]) >= t and r["hit"] == "1")
    fa = sum(1 for r in no if float(r["top1_score"]) >= t)
    pass_rate.append(ok / len(ans) * 100)
    fa_rate.append(fa / len(no) * 100)
ax.plot(ths, pass_rate, color="#2E7D32", lw=2.2, label="可答题通过率（答对且过阈值）")
ax.plot(ths, fa_rate, color="#C62828", lw=2.2, label="不可答题误接受率")
ax.axvline(0.92, color="#1565C0", ls="--", lw=1.6)
ax.annotate("当前阈值 0.92\n通过 96.6% / 误接受 2.9%", xy=(0.92, 50), xytext=(0.845, 62),
            fontsize=9.5, color="#1565C0",
            arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2))
ax.set_xlabel("精排分数阈值", fontsize=11)
ax.set_ylabel("百分比 (%)", fontsize=11)
ax.set_title("① 阈值标定：为什么取 0.92", fontsize=12.5, fontweight="bold")
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc="center left")
ax.set_ylim(-2, 105)

# ── 面板 2：三模式消融 ──
ax = axes[0][1]
modes = ["dense", "hybrid", "rerank"]
labels = ["纯向量", "混合检索", "混合+精排"]
hit = [summary["results"][m]["hit_at_feed_pct"] for m in modes]
fa = [summary["results"][m]["false_accept_pct"] for m in modes]
x = range(len(modes))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], hit, w, color="#2E7D32", label="hit@3（可答题）")
b2 = ax.bar([i + w / 2 for i in x], fa, w, color="#C62828", label="误接受率（不可答题）")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
            ha="center", fontsize=9.5)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 110)
ax.set_ylabel("百分比 (%)", fontsize=11)
ax.set_title("② 消融：精排用 2.7% 的 hit 换掉 68% 的误接受", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)

# ── 面板 3：分题型（rerank vs dense）──
ax = axes[1][0]
types = ["exact", "paraphrase"]
type_names = ["字面命中（120）", "改述提问（28）"]
vals = {m: [] for m in ("dense", "rerank")}
for tp in types:
    for m in ("dense", "rerank"):
        sub = [r for r in rows if r["mode"] == m and r["type"] == tp and r["answerable"] == "1"]
        ok = sum(1 for r in sub if r["returned"] == "1" and r["hit"] == "1")
        vals[m].append(ok / max(1, len(sub)) * 100)
x = range(len(types))
b1 = ax.bar([i - w / 2 for i in x], vals["dense"], w, color="#90A4AE", label="纯向量")
b2 = ax.bar([i + w / 2 for i in x], vals["rerank"], w, color="#6A1B9A", label="混合+精排")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
            ha="center", fontsize=9.5)
ax.set_xticks(list(x))
ax.set_xticklabels(type_names, fontsize=11)
ax.set_ylim(0, 110)
ax.set_ylabel("hit@3 (%)", fontsize=11)
ax.set_title("③ 题型差异：改述提问是难点", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)

# ── 面板 4：延迟 ──
ax = axes[1][1]
p50 = [summary["results"][m]["latency_p50_ms"] for m in modes]
p95 = [summary["results"][m]["latency_p95_ms"] for m in modes]
x = range(len(modes))
b1 = ax.bar([i - w / 2 for i in x], p50, w, color="#1565C0", label="P50")
b2 = ax.bar([i + w / 2 for i in x], p95, w, color="#64B5F6", label="P95")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{b.get_height():.0f}",
            ha="center", fontsize=9.5)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("毫秒 (ms)", fontsize=11)
ax.set_title("④ 检索延迟：精排约 +420ms，仍远低于 LLM", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout(rect=(0, 0, 1, 0.96))
out = r"D:\文本需求拆分\docs\检索评测可视化.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120, facecolor="white")
print("saved:", out)
