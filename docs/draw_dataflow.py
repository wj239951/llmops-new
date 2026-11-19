# -*- coding: utf-8 -*-
"""用户输入数据流转图：原文 → 清洗 → OCR → 合并 → 归一/改写 → 检索 → 生成 → 落库。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

W, H = 178, 300
fig, ax = plt.subplots(figsize=(20, 26), dpi=100)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")

STAGE_X, STAGE_W = 3, 30
DATA_X, DATA_W = 36, 86
NOTE_X, NOTE_W = 126, 48

COL = {
    "in":  ("#E8F1FB", "#4A78B5"),
    "proc": ("#FFF4E5", "#C4832B"),
    "ret": ("#F3ECFA", "#7A54B0"),
    "gen": ("#EAF5EC", "#3F8A55"),
    "store": ("#EAF6F8", "#2E8A9A"),
}

state = {"y": 16.0}


def stage(title, sub, data_lines, note_lines, key="proc", data_title=""):
    y = state["y"]
    h_stage = 6 + 2.9 * 2
    for i, line in enumerate(data_lines):
        h_stage = max(h_stage, 6 + 2.9 * (i + 1))
    h = max(h_stage, 12)

    fill, edge = COL[key]
    # 阶段名
    ax.add_patch(FancyBboxPatch((STAGE_X, y), STAGE_W, h, boxstyle="round,pad=0,rounding_size=0.8",
                                linewidth=1.4, edgecolor=edge, facecolor=fill))
    ax.text(STAGE_X + STAGE_W / 2, y + 3.4, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#1f1f1f")
    ax.text(STAGE_X + STAGE_W / 2, y + 7.4, sub, ha="center", va="center",
            fontsize=8.4, color="#666")

    # 数据形态（深色底，等宽感）
    ax.add_patch(FancyBboxPatch((DATA_X, y), DATA_W, h, boxstyle="round,pad=0,rounding_size=0.8",
                                linewidth=1.2, edgecolor="#D6D6D6", facecolor="#FAFAFA"))
    if data_title:
        ax.text(DATA_X + 2, y + 3.0, data_title, ha="left", va="center", fontsize=8.4, color="#999")
    for i, line in enumerate(data_lines):
        ax.text(DATA_X + 2, y + 6.2 + i * 2.9, line, ha="left", va="center",
                fontsize=8.8, color="#2f2f2f")

    # 去向/说明
    ax.add_patch(FancyBboxPatch((NOTE_X, y), NOTE_W, h, boxstyle="round,pad=0,rounding_size=0.8",
                                linewidth=1.0, edgecolor="#E0E0E0", facecolor="#FFFFFF"))
    for i, line in enumerate(note_lines):
        ax.text(NOTE_X + 2, y + 4.2 + i * 2.9, line, ha="left", va="center",
                fontsize=8.4, color="#555")

    state["y"] = y + h + 7.5
    return y, y + h


def arrow(label=""):
    y0 = state["y"] - 7.5
    ax.add_patch(FancyArrowPatch((STAGE_X + STAGE_W / 2, y0), (STAGE_X + STAGE_W / 2, y0 + 6.5),
                                 arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color="#666"))
    if label:
        ax.text(STAGE_X + STAGE_W / 2 + 1.5, y0 + 3.4, label, fontsize=8.2, color="#888",
                ha="left", va="center")


# ───── 标题 ─────
ax.text(W / 2, 6.5, "用户输入数据流转图", ha="center", fontsize=23, fontweight="bold", color="#141414")
ax.text(W / 2, 11.5, "同一个请求里，用户原文如何一步步变成检索查询、模型上下文与最终回答",
        ha="center", fontsize=10.5, color="#666")

stage("1 用户输入", "键盘 / 粘贴 / 上传图片",
      ["【设备】ＺＸ－１２００  巡检周期？？？   ",
       "＋ 图片（可选）：设备铭牌照片"],
      ["前端：Chat Studio", "原文先留在浏览器"],
      key="in", data_title="浏览器里的原始内容")

arrow("POST /api/chat/stream（JSON）")

stage("2 请求体", "FastAPI + Bearer 鉴权",
      ['{ "query": "【设备】ＺＸ－１２００  巡检周期？？？",',
       '  "image_base64": "data:image/png;base64,...",',
       '  "use_rag": true, "use_rerank": true }'],
      ["后端入口", "req.query 原样保留"],
      key="in", data_title="HTTP JSON")

arrow("ChatService._prepare()")

stage("3 输入清洗", "全角→半角 / 空白 / 重复标点",
      ["【设备】ZX-1200 巡检周期?",
       "（全角字母数字归一、连续空格压缩、？？？→?）"],
      ["clean_query（仅内存）", "失败自动回退规则"],
      key="proc", data_title="清洗后的文本")

arrow("可选：有图才走")

stage("4 图片 OCR", "RapidOCR ≈1.6 s（CPU）",
      ["型号 ZX-1200",
       "额定功率 11 kw",
       "锁模力 1200     出厂编号 2024-0517"],
      ["meta: chars=67, ms=1617", "低置信度框丢弃"],
      key="proc", data_title="识别出的文字（按行聚合）")

arrow("merge_query()：图片在前、提问在后")

stage("5 拼装检索查询", "final_query",
      ["型号 ZX-1200 / 额定功率 11 kw / 出厂编号 2024-0517",
       "【设备】ZX-1200 巡检周期?"],
      ["图片提供实体", "提问决定意图"],
      key="proc", data_title="合并后的检索文本")

arrow("query_alias.normalize_terms()")

stage("6 术语归一", "口语 → 语料术语",
      ["密闭车间作业需要注意什么",
       "→ 受限空间作业需要注意什么"],
      ["只在检索侧生效", "工作流里留痕：步骤6-0"],
      key="proc", data_title="替换后的查询")

arrow("query_rewrite.rewrite()：有历史时")

stage("7 多轮改写", "代词消解 / 补全省略",
      ["它多久巡检一次",
       "→ ZX-1200 多久巡检一次"],
      ["改写无果 → 原句兜底重检", "精确 token 不丢"],
      key="proc", data_title="自包含查询")

arrow("rag.search()")

stage("8 混合检索", "dense + BM25 → RRF",
      ["dense:  bge-m3(1024d) → top-k",
       "sparse: jieba + BM25   → top-k",
       "RRF(k=60) → 候选 6 条"],
      ["Chroma + MySQL 文档表", "exact 命中可绕过阈值"],
      key="ret", data_title="两路召回与融合")

arrow("精排 + 阈值门控")

stage("9 精排与门控", "bge-reranker-v2-m3",
      ["top1 = 0.985  ≥ 0.92 → hit",
       "候选：05_受限空间作业安全规程"],
      ["< 0.92 → weak_hit / no_hit", "no_hit → 固定话术（不调 LLM）"],
      key="ret", data_title="打分结果")

arrow("_grounded_prompt()")

stage("10 上下文组装", "文档级去重 + top3",
      ["[知识片段1] 05_受限空间作业安全规程 …",
       "[知识片段2] 32_受限空间气体检测 …",
       "约束：只依据片段回答，标注【依据：知识片段N】"],
      ["最终 prompt 送 LLM", "无命中则不组装"],
      key="gen", data_title="送给模型的上下文")

arrow("LLM 流式生成")

stage("11 生成与校验", "DeepSeek / Ollama + grounding",
      ["回答：进入前需检测氧含量 19.5%–21%【依据：知识片段1】",
       "校验：19.5 / 21 均出现在片段中 → 通过"],
      ["数字不在片段 → 追加提示", "SSE 增量推送前端"],
      key="gen", data_title="回答文本")

arrow("落库 + 收尾")

stage("12 输出与落库", "MySQL 4 张表",
      ["chat_logs：query 原文 + answer + sources + timings",
       "conversation_messages：用户原话 / 助手回答",
       "Chroma：不改动（只读检索）"],
      ["注意：存的是用户原话", "清洗/改写结果不落库"],
      key="store", data_title="持久化内容")

# ───── 右侧要点 ─────
tip_y = state["y"] + 4
ax.text(3, tip_y, "三个容易混淆的边界", fontsize=11.5, fontweight="bold", color="#333")
tips = [
    ("原文 vs 派生物", "只有 req.query（原话）进库；clean / merged / search 都是过程量"),
    ("图片文字只进检索", "OCR 结果拼进检索查询与 prompt 上下文，不改变回答措辞"),
    ("术语归一单向", "只替换检索词；用户看到的问题与回答保持原样"),
]
for i, (t1, t2) in enumerate(tips):
    ax.text(3, tip_y + 6 + i * 5.0, f"· {t1}：{t2}", fontsize=8.8, color="#555")

out = r"D:\文本需求拆分\docs\用户输入数据流转图.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
ax.set_ylim(H, 0)
plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
print("saved:", out, "| content bottom:", round(state["y"], 1))
