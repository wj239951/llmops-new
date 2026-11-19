# -*- coding: utf-8 -*-
"""项目架构图 v4：纵向顺序布局，band 高度由内容决定，箭头走固定竖向通道。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

W = 134
COL = {
    "fe": ("#E8F1FB", "#4A78B5"), "api": ("#EAF5EC", "#3F8A55"),
    "orc": ("#FFF4E5", "#C4832B"), "rag": ("#F3ECFA", "#7A54B0"),
    "data": ("#EAF6F8", "#2E8A9A"), "ext": ("#FBE9E9", "#B4514F"),
    "mod": ("#F0F0F0", "#6B6B6B"),
}
TITLE_H, LINE_H, PAD_TOP, PAD_BOT = 4.2, 2.7, 3.0, 2.4


def box_h(n):
    return PAD_TOP + TITLE_H + n * LINE_H + PAD_BOT


fig_h = 30
fig, ax = plt.subplots(figsize=(18.4, fig_h), dpi=100)
ax.axis("off")

state = {"y": 10.0}


def band(label, sub=""):
    """开一个层：返回内容起始 y，并记录 band 起点。"""
    y = state["y"]
    state["band_start"] = y
    state["band_label_y"] = y + 3.4
    ax.text(3.0, y + 3.4, label, fontsize=13.5, fontweight="bold", color="#2b2b2b", va="center")
    if sub:
        ax.text(W - 3.0, y + 3.4, sub, fontsize=9.2, color="#8a8a8a", va="center", ha="right")
    state["content_y"] = y + 8.5
    return state["content_y"]


def close_band():
    bottom = state["max_bottom"]
    ax.add_patch(FancyBboxPatch((1.5, state["band_start"]), W - 3, bottom + 6 - state["band_start"],
                                boxstyle="round,pad=0,rounding_size=1.0",
                                linewidth=0, facecolor="#F7F8FA", zorder=0))
    state["y"] = bottom + 9
    return bottom


def box(x, y, w, title, lines=(), key="orc", fs=11.5, lfs=9.2):
    fill, edge = COL[key]
    h = box_h(len(lines))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + PAD_TOP + TITLE_H * 0.4, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#1f1f1f", zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + PAD_TOP + TITLE_H + i * LINE_H + 0.4, line,
                ha="center", va="center", fontsize=lfs, color="#454545", zorder=3)
    state["max_bottom"] = max(state.get("max_bottom", y), y + h)
    return y + h


def arrow(x0, y0, x1, y1, text="", color="#575757", ls="-", lw=1.5, dx=1.0, dy=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=lw, color=color, linestyle=ls, shrinkA=1, shrinkB=1, zorder=1))
    if text:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, fontsize=8.6, color="#6a6a6a",
                ha="left", va="center", zorder=3)


# ───────── 标题 ─────────
ax.text(W / 2, 3.8, "工业 LLMOps 平台 · 项目架构", ha="center", va="center",
        fontsize=24, fontweight="bold", color="#141414")
ax.text(W / 2, 8.0, "Vue3 前端 ｜ FastAPI 接入 ｜ 混合检索（Chroma + BM25）｜ 云端精排 ｜ 严格 grounded 生成",
        ha="center", va="center", fontsize=10.8, color="#666")

# ───────── ① 前端 ─────────
y = band("① 用户与前端", "Vue3 + Element Plus + Vite ｜ SSE 流式渲染")
b1 = box(5, y, 30, "Chat Studio", ["提问 / 图片上传（新增）", "RAG·精排·清洗开关", "SSE 增量渲染 + 溯源"], "fe")
box(38, y, 28, "Prompt 编排", ["系统提示词模板", "provider / model / 温度"], "fe")
box(69, y, 20, "知识库管理", ["文档增删改查", "上传 → 解析 → 入库"], "fe")
box(92, y, 20, "运行日志", ["调用记录 / 耗时", "token / 失败原因"], "fe")
fe_bottom = close_band()

# ───────── ② 接入层 ─────────
y = band("② 接入层（FastAPI）", "Bearer 口令鉴权 ｜ CORS ｜ SSE 关闭代理缓冲")
box(5, y, 29, "POST /api/chat", ["非流式（流式失败时降级）"], "api", fs=11, lfs=9)
b2 = box(37, y, 29, "POST /api/chat/stream", ["SSE：status → meta → delta → done"], "api", fs=11, lfs=9)
box(69, y, 20, "/api/chat/options", ["服务端能力探测"], "api", fs=10.6, lfs=8.8)
box(92, y, 20, "资源类接口", ["会话 / 知识库 / 日志 / Prompt"], "api", fs=10.6, lfs=8.4)
api_bottom = close_band()

# ───────── ③ 编排层 ─────────
y = band("③ 对话编排（ChatService）", "一轮对话：预处理 → 检索 → 生成 → 校验 → 落库")
rowA = y
box(4, rowA, 23, "1 输入清洗", ["规则通道 / 模型通道", "全角·空白·HTML·断行", "失败回退 rule"], "orc")
box(30, rowA, 23, "2 图片 OCR（新增）", ["RapidOCR ~2s（CPU）", "行聚合 + 置信度过滤", "与提问合并后检索"], "orc")
box(56, rowA, 20, "3 会话记忆", ["多轮历史窗口", "MEMORY_MAX_TURNS"], "orc")
box(79, rowA, 22, "4 Prompt / 模型", ["模板或默认提示词", "DeepSeek 云 / Ollama"], "orc")
box(104, rowA, 25, "5 术语归一", ["口语 → 语料术语", "密闭车间 → 受限空间"], "orc")
rowA_bottom = rowA + box_h(3)

rowB = rowA_bottom + 8
state["max_bottom"] = rowB
box(4, rowB, 25, "6 查询改写", ["多轮代词消解", "保留型号等精确 token", "改写无果 → 原句兜底"], "orc")
box(32, rowB, 30, "7 混合检索 + 精排", ["dense(bge-m3) + BM25(jieba)", "RRF 融合 k=60 → 候选 6 条", "精排 bge-reranker-v2-m3"], "orc")
box(65, rowB, 24, "8 阈值门控", ["余弦 0.56 / 精排 0.92", "弱命中余量 0.06", "→ hit / weak_hit / no_hit"], "orc")
box(92, rowB, 20, "9 上下文组装", ["文档级去重 + top3", "grounded prompt", "无命中 → 固定话术"], "orc", fs=10.8)
box(115, rowB, 14, "10 生成", ["流式增量"], "orc", fs=10.6, lfs=8.8)
rowB_bottom = rowB + box_h(3)

rowC = rowB_bottom + 8
state["max_bottom"] = max(state["max_bottom"], rowC)
box(32, rowC, 30, "11 大模型生成", ["DeepSeek API / Ollama", "temperature 0.7", "流式返回增量"], "orc")
box(65, rowC, 24, "12 数字级校验", ["回答中的数字必须在片段出现", "缺失则追加提示", "grounding"], "orc")
rowC_bottom = rowC + box_h(3)
orc_bottom = close_band()

# ───────── ④ 检索子系统 ─────────
y = band("④ 检索子系统（rag_service / hybrid_service）", "双通道召回 → RRF 融合 → 精排 → 阈值")
box(5, y, 25, "向量通道", ["Chroma 持久化库", "bge-m3（1024 维）", "Ollama / SiliconFlow 二选一"], "rag", fs=11)
box(33, y, 25, "稀疏通道 BM25", ["jieba 分词 + 前缀词典", "型号/编号 exact 命中", "exact 可绕过阈值"], "rag", fs=11)
box(61, y, 21, "RRF 融合", ["k = 60", "两路名次倒数求和", "候选 K = 6"], "rag", fs=11)
box(85, y, 24, "精排（云端）", ["SiliconFlow bge-reranker", "sigmoid 分数 0–1", "失败自动回退余弦"], "rag", fs=11)
rag_bottom = close_band()

# ───────── ⑤ 数据与存储 ─────────
y = band("⑤ 数据与存储", "")
box(5, y, 25, "MySQL", ["llm_knowledge_documents", "conversations / messages", "chat_logs / prompts"], "data", fs=11, lfs=8.6)
box(33, y, 25, "Chroma", ["向量集合（持久化）", "doc_id 与 MySQL 关联", "本地目录 chroma_db"], "data", fs=11, lfs=8.6)
box(61, y, 21, "知识库文件", ["PDF / Word / TXT", "file_parser 解析", "切分后入库"], "data", fs=11, lfs=8.6)
box(85, y, 24, "评测资产（eval/）", ["golden 题库 183 + 30", "基线 / gate / e2e", "拒答与数字校验"], "data", fs=11, lfs=8.6)
data_bottom = close_band()

# ───────── ⑥ 外部服务 ─────────
y = band("⑥ 外部服务与模型资产", "")
box(4, y, 24, "DeepSeek 云端 API", ["deepseek-chat"], "ext", fs=10.4, lfs=8.6)
box(30, y, 24, "SiliconFlow", ["bge-m3 / bge-reranker"], "ext", fs=10.4, lfs=8.6)
box(56, y, 22, "Ollama 本地", ["deepseek-r1:7b / bge-m3"], "ext", fs=10.4, lfs=8.6)
box(80, y, 22, "clean_server :8010", ["清洗模型服务（可选）"], "ext", fs=10.0, lfs=8.4)
box(104, y, 25, "Qwen3.5-0.8B 清洗模型", ["+ 结构蒸馏 LoRA（实验）"], "mod", fs=9.8, lfs=8.4)
ext_bottom = close_band()

# ───────── 箭头 ─────────
arrow(20, fe_bottom + 0.5, 20, api_bottom - box_h(1) - 0.5 + 0.5 - 3.2, "HTTP / SSE")
arrow(51, fe_bottom + 0.5, 51, 0, "")

# 用固定 y 重画（上面第二根需要 API 顶部坐标）
plt.close(fig)
print("layout computed; rebuilding with arrows")

# ───────── 重新绘制（含箭头）─────────
fig, ax = plt.subplots(figsize=(18.4, fig_h), dpi=100)
ax.axis("off")
state = {"y": 10.0}
ax.text(W / 2, 3.8, "工业 LLMoPs 平台 · 项目架构", ha="center", va="center", fontsize=24,
        fontweight="bold", color="#141414")
ax.text(W / 2, 8.0, "Vue3 前端 ｜ FastAPI 接入 ｜ 混合检索（Chroma + BM25）｜ 云端精排 ｜ 严格 grounded 生成",
        ha="center", va="center", fontsize=10.8, color="#666")

y = band("① 用户与前端", "Vue3 + Element Plus + Vite ｜ SSE 流式渲染")
box(5, y, 30, "Chat Studio", ["提问 / 图片上传（新增）", "RAG·精排·清洗开关", "SSE 增量渲染 + 溯源"], "fe")
box(38, y, 28, "Prompt 编排", ["系统提示词模板", "provider / model / 温度"], "fe")
box(69, y, 20, "知识库管理", ["文档增删改查", "上传 → 解析 → 入库"], "fe")
box(92, y, 20, "运行日志", ["调用记录 / 耗时", "token / 失败原因"], "fe")
fe_top, fe_bottom = y, close_band()

y = band("② 接入层（FastAPI）", "Bearer 口令鉴权 ｜ CORS ｜ SSE 关闭代理缓冲")
box(5, y, 29, "POST /api/chat", ["非流式（流式失败时降级）"], "api", fs=11, lfs=9)
box(37, y, 29, "POST /api/chat/stream", ["SSE：status → meta → delta → done"], "api", fs=11, lfs=9)
box(69, y, 20, "/api/chat/options", ["服务端能力探测"], "api", fs=10.6, lfs=8.8)
box(92, y, 20, "资源类接口", ["会话 / 知识库 / 日志 / Prompt"], "api", fs=10.6, lfs=8.4)
api_top, api_bottom = y, close_band()

y = band("③ 对话编排（ChatService）", "一轮对话：预处理 → 检索 → 生成 → 校验 → 落库")
rowA = y
box(4, rowA, 23, "1 输入清洗", ["规则通道 / 模型通道", "全角·空白·HTML·断行", "失败回退 rule"], "orc")
box(30, rowA, 23, "2 图片 OCR（新增）", ["RapidOCR ~2s（CPU）", "行聚合 + 置信度过滤", "与提问合并后检索"], "orc")
box(56, rowA, 20, "3 会话记忆", ["多轮历史窗口", "MEMORY_MAX_TURNS"], "orc")
box(79, rowA, 22, "4 Prompt / 模型", ["模板或默认提示词", "DeepSeek 云 / Ollama"], "orc")
box(104, rowA, 25, "5 术语归一", ["口语 → 语料术语", "密闭车间 → 受限空间"], "orc")
rowA_b = rowA + box_h(3)
rowB = rowA_b + 8
box(4, rowB, 25, "6 查询改写", ["多轮代词消解", "保留型号等精确 token", "改写无果 → 原句兜底"], "orc")
box(32, rowB, 30, "7 混合检索 + 精排", ["dense(bge-m3) + BM25(jieba)", "RRF 融合 k=60 → 候选 6 条", "精排 bge-reranker-v2-m3"], "orc")
box(65, rowB, 24, "8 阈值门控", ["余弦 0.56 / 精排 0.92", "弱命中余量 0.06", "→ hit / weak_hit / no_hit"], "orc")
box(92, rowB, 20, "9 上下文组装", ["文档级去重 + top3", "grounded prompt", "无命中 → 固定话术"], "orc", fs=10.8)
box(115, rowB, 14, "10 生成", ["流式增量"], "orc", fs=10.6, lfs=8.8)
rowB_b = rowB + box_h(3)
rowC = rowB_b + 8
box(32, rowC, 30, "11 大模型生成", ["DeepSeek API / Ollama", "temperature 0.7", "流式返回增量"], "orc")
box(65, rowC, 24, "12 数字级校验", ["回答中的数字必须在片段出现", "缺失则追加提示", "grounding"], "orc")
rowC_b = rowC + box_h(3)
orc_top, orc_bottom = rowA, close_band()

y = band("④ 检索子系统（rag_service / hybrid_service）", "双通道召回 → RRF 融合 → 精排 → 阈值")
box(5, y, 25, "向量通道", ["Chroma 持久化库", "bge-m3（1024 维）", "Ollama / SiliconFlow 二选一"], "rag", fs=11)
box(33, y, 25, "稀疏通道 BM25", ["jieba 分词 + 前缀词典", "型号/编号 exact 命中", "exact 可绕过阈值"], "rag", fs=11)
box(61, y, 21, "RRF 融合", ["k = 60", "两路名次倒数求和", "候选 K = 6"], "rag", fs=11)
box(85, y, 24, "精排（云端）", ["SiliconFlow bge-reranker", "sigmoid 分数 0–1", "失败自动回退余弦"], "rag", fs=11)
rag_top, rag_bottom = y, close_band()

y = band("⑤ 数据与存储", "")
box(5, y, 25, "MySQL", ["llm_knowledge_documents", "conversations / messages", "chat_logs / prompts"], "data", fs=11, lfs=8.6)
box(33, y, 25, "Chroma", ["向量集合（持久化）", "doc_id 与 MySQL 关联", "本地目录 chroma_db"], "data", fs=11, lfs=8.6)
box(61, y, 21, "知识库文件", ["PDF / Word / TXT", "file_parser 解析", "切分后入库"], "data", fs=11, lfs=8.6)
box(85, y, 24, "评测资产（eval/）", ["golden 题库 183 + 30", "基线 / gate / e2e", "拒答与数字校验"], "data", fs=11, lfs=8.6)
data_top, data_bottom = y, close_band()

y = band("⑥ 外部服务与模型资产", "")
box(4, y, 24, "DeepSeek 云端 API", ["deepseek-chat"], "ext", fs=10.4, lfs=8.6)
box(30, y, 24, "SiliconFlow", ["bge-m3 / bge-reranker"], "ext", fs=10.4, lfs=8.6)
box(56, y, 22, "Ollama 本地", ["deepseek-r1:7b / bge-m3"], "ext", fs=10.4, lfs=8.6)
box(80, y, 22, "clean_server :8010", ["清洗模型服务（可选）"], "ext", fs=10.0, lfs=8.4)
box(104, y, 25, "Qwen3.5-0.8B 清洗模型", ["+ 结构蒸馏 LoRA（实验）"], "mod", fs=9.8, lfs=8.4)
ext_top, ext_bottom = y, close_band()

# 主链路：前端 → 接入 → 编排
arrow(20, fe_bottom - 0.5, 20, api_top, "HTTP / SSE", dx=1.2)
arrow(51, fe_bottom - 0.5, 51, api_top)
# 接入 → 预处理行
for x in (15, 41, 66, 90, 116):
    arrow(x, api_bottom - 0.5, x, rowA)
# 预处理 → 检索行
for x in (15, 41, 116):
    arrow(x, rowA_b, x, rowB)
arrow(29, rowB + 6, 8, rowB + 6) if False else None
arrow(29, rowB + 6, 32, rowB + 6)
arrow(62, rowB + 6, 65, rowB + 6)
arrow(89, rowB + 6, 92, rowB + 6)
arrow(112, rowB + 6, 115, rowB + 6)
# 检索 → 生成（上下文）
arrow(47, rowB_b + 1, 47, rowC, "上下文")
# 生成 → 校验
arrow(62, rowC + 6, 65, rowC + 6)

# 编排 → 检索子系统（虚线上溯）
arrow(45, rag_top - 0.5, 45, rowB_b + 1, "召回候选", color="#7A54B0", ls="--", lw=1.2, dx=-11, dy=-1)
arrow(17, rag_top - 0.5, 17, rowB_b + 1, "", color="#7A54B0", ls="--", lw=1.2)
arrow(71, rag_top - 0.5, 71, rowC_b + 1, "", color="#7A54B0", ls="--", lw=1.2)

# 数据层（虚线）
arrow(17, data_top - 0.5, 17, rag_bottom - 0.5, "读文档", color="#2E8A9A", ls="--", lw=1.2, dx=-10)
arrow(45, data_top - 0.5, 45, rag_bottom - 0.5, "读写向量", color="#2E8A9A", ls="--", lw=1.2, dx=1)

# 外部服务（点线）
arrow(16, ext_top - 0.5, 16, rag_bottom - 0.5, "embed / rerank", color="#B4514F", ls=":", lw=1.2, dx=-16, dy=2)
arrow(42, ext_top - 0.5, 42, rag_bottom - 0.5, "", color="#B4514F", ls=":", lw=1.2)
arrow(67, ext_top - 0.5, 67, rowC_b + 1, "LLM", color="#B4514F", ls=":", lw=1.2)
arrow(91, ext_top - 0.5, 91, rowA_b + 1, "清洗", color="#B4514F", ls=":", lw=1.2, dx=-5)

# 图例
legend_y = ext_bottom + 5
ax.text(3.0, legend_y, "图例：实线 = 请求主链路　虚线 = 检索/向量读写　点线 = 外部服务调用　｜　"
                       "阈值：余弦 0.56、精排 0.92（弱命中余量 0.06）　｜　图片 OCR 为本次新增能力",
        fontsize=8.8, color="#6a6a6a")

ax.set_xlim(0, W)
ax.set_ylim(legend_y + 4, 0)

out = r"D:\文本需求拆分\docs\项目架构图.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
print("saved:", out)
print("total height:", round(legend_y + 4, 1))
