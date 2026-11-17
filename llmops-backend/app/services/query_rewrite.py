# -*- coding: utf-8 -*-
"""多轮检索查询改写（contextualization）：把「那它的更换周期呢」补成可独立检索的完整问句。

为什么加这一层：
- 检索只吃当前一句（chat_service._apply_rag 用 clean_query），历史只进了生成；
- 第二轮的「那它呢」会丢主体，向量与 BM25 同时跑偏，闸门按无答案处理（docs/13 的多轮题库可复现）；
- 但不能每轮都改：单轮、自带型号、明确换话题的追问，改写了反而引入漂移。

设计取舍：
- 只改「检索查询」，不改用户输入：原文仍用于展示、写库与生成；
- 三种模式：off 关闭 / rule 规则（零延迟，从最近轮次抽型号回填指代）/ llm 走模型（更准，每轮多一次调用）；
- 失败一律回退：llm 失败退 rule，rule 失败退原句，任何情况都不阻断问答；
- 改写结果必须保留原句里的型号/编号（exact bypass 的命门），丢了就补回去；
- 顺序：输入清洗 → 本层 → 查询分解（分解需要完整句）。
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time

from app.core.config import settings

logger = logging.getLogger(__name__)

_ID_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
# 指代：只用于「替换」，不包含语气词
_PRONOUN_RE = re.compile(r"(它们|它|其|该|此|这个|那个|这些|那些|上述|前面|刚才)")
# 指代/省略信号：只用于「要不要改写」的判断，含语气词与换话题的苗头
_REFER_RE = re.compile(_PRONOUN_RE.pattern + r"|还有|呢|换一个|另外")

_MAX_HISTORY_TURNS = 4
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="qrw")

_SYSTEM_PROMPT = (
    "你是检索查询改写助手。根据对话历史，把用户最新一句改写成不依赖上下文、可独立检索的完整问题。"
    "规则：只改写不回答；保持原疑问类型；只使用历史里出现过的实体；"
    "本来就完整的问题原样返回；只输出 JSON，格式为 {\"query\": \"...\", \"changed\": true|false}。"
    "示例："
    "历史用户：注塑机 ZX-2026 的螺杆链条用什么润滑脂？"
    "最新：那它的更换周期呢 → {\"query\": \"ZX-2026 螺杆链条润滑脂的更换周期\", \"changed\": true}"
    "最新：液压油有哪些品牌 → {\"query\": \"液压油有哪些品牌\", \"changed\": false}"
)


def exact_tokens(text: str) -> list[str]:
    """型号/编号型 token（含数字且长度 ≥ 4），与 rag_service.exact_tokens 口径一致。

    这里不复用 rag_service 的实现：那个模块 import 时就会打开 Chroma，
    改写在 chat 链路的前段，不该为此付出向量库初始化的代价。
    """
    return [t for t in _ID_TOKEN_RE.findall((text or "").lower())
            if len(t) >= 4 and any(c.isdigit() for c in t)]


def _recent_user_turns(history, limit: int = _MAX_HISTORY_TURNS) -> list[str]:
    turns = [str(m.get("content") or "") for m in (history or []) if m.get("role") == "user"]
    return [t for t in turns if t.strip()][-limit:]


def should_rewrite(query: str, history) -> bool:
    """只在「多轮 + 有指代/省略信号」时才值得花一次调用。"""
    if not _recent_user_turns(history):
        return False
    q = (query or "").strip()
    if not q:
        return False
    if exact_tokens(q) and not _PRONOUN_RE.search(q):
        return False          # 自带型号且无指代：已经是自包含查询，改了是纯风险
    # 多轮里不带型号的问句几乎一定是省略（「合模机构间隙检测多久一次？」），补主体收益远大于风险；
    # 实测把「≤12 字」这种长度阈值去掉后，20 段多轮的第 2 轮改写触发从 18 段变成 20 段
    return True


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _rule_rewrite(query: str, history) -> str:
    """规则改写：从最近轮次抽型号，其次抽关键词，回填进指代/省略位置。"""
    turns = _recent_user_turns(history)
    if not turns:
        return query
    subject = ""
    for turn in reversed(turns):
        toks = exact_tokens(turn)
        if toks:
            subject = toks[-1]
            break
    if not subject:
        try:
            import jieba.analyse
            kws = jieba.analyse.extract_tags(turns[-1], topK=3)
            subject = kws[0] if kws else ""
        except Exception:      # noqa: BLE001 - 分词失败不影响主链路
            subject = ""
    if not subject:
        return query
    q = re.sub(r"^(那么|那|然后|所以)[，,]?\s*", "", query.strip())
    if _PRONOUN_RE.search(q):
        return _PRONOUN_RE.sub(subject, q, count=1)
    return f"{subject} {q}".strip()


def _llm_rewrite(query: str, history) -> str:
    from app.services.llm_service import LLMService
    lines = [f"用户：{t[:200]}" for t in _recent_user_turns(history)]
    prompt = "对话历史：\n" + "\n".join(lines) + f"\n\n最新问题：{query}"
    resp = LLMService().chat(query=prompt, system_prompt=_SYSTEM_PROMPT, temperature=0.0)
    data = _parse_json((resp.get("answer") or "").strip())
    out = data.get("query")
    if not isinstance(out, str) or not out.strip():
        raise ValueError("改写结果为空")
    if len(out) > max(60, len(query) * 3):
        raise ValueError("改写结果异常变长")
    return out.strip()


def _keep_exact_tokens(original: str, rewritten: str) -> str:
    """型号/编号是 exact bypass 的命门：改写丢了就补回去，不能让它静默失效。"""
    missing = [t for t in exact_tokens(original) if t not in rewritten.lower()]
    return f"{rewritten} {' '.join(missing)}" if missing else rewritten


def rewrite(query: str, history=None) -> tuple[str, dict]:
    """返回 (检索用查询, 元信息)。任何异常都回退，不向上抛。"""
    mode = (settings.RAG_QUERY_REWRITE_MODE or "off").lower()
    meta = {"mode": mode, "applied": False, "changed": False, "ms": 0, "fallback": None}
    started = time.perf_counter()

    if mode == "off" or not should_rewrite(query, history):
        meta["ms"] = int((time.perf_counter() - started) * 1000)
        return query, meta

    meta["applied"] = True
    out = ""
    if mode == "llm":
        try:
            future = _POOL.submit(_llm_rewrite, query, history)
            out = future.result(timeout=settings.RAG_QUERY_REWRITE_TIMEOUT)
            meta["via"] = "llm"
        except Exception as exc:      # noqa: BLE001 - 超时/网络/解析失败一视同仁
            logger.warning("查询改写失败，按 %s 回退：%s", settings.RAG_QUERY_REWRITE_FALLBACK, exc)
            meta["fallback"] = settings.RAG_QUERY_REWRITE_FALLBACK
    if not out and (mode == "rule" or meta["fallback"] == "rule"):
        try:
            out = _rule_rewrite(query, history)
            meta.setdefault("via", "rule")
        except Exception:      # noqa: BLE001
            out = ""

    out = _keep_exact_tokens(query, out or query)
    meta["query"] = out
    meta["changed"] = out.strip() != (query or "").strip()
    meta["ms"] = int((time.perf_counter() - started) * 1000)
    return out, meta
