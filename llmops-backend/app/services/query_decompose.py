# -*- coding: utf-8 -*-
"""检索查询分解：把复合问题拆成主题子问句，各自过精排闸门后合并。

为什么加这一层（2026-11-12 服务器实测）：
- 「驾驶公务车辆出差怎么报销，有什么制度」跨两篇文档，交叉编码器按
  (整句, 单片段) 打分只有 0.60，0.76 闸门全滤掉 → 零召回；
- 按主题拆开后子问句对各自文档打 0.95/0.96，立刻过闸；
- 降全局阈值无解：0.60 低于无答案题 q43 的 0.743，降阈值会破拒答。

设计取舍：
- 默认关闭（RAG_QUERY_DECOMPOSE_ENABLED=false），只作用于精排分支；
- 分解失败 / 解析失败 / 模型异常一律回退 [原句]，等价于没有这一层；
- 拆 + 主题归一：口语子问句要改写成正式主题表述（「驾驶公务车辆出差怎么报销」
  实测 0.40，归一成「差旅费用报销有什么制度规定」后 0.96）；
  单主题问题原样直通，保持阈值标定尺度不变；
- 按标点机械拆不行（拆出的前半句仍是复合句），所以用 LLM 按主题拆，
  每问多一次云端调用；提示词必须带少样本示例——纯指令会被模型忽略，
  实测仍按标点照抄原句。
"""
from __future__ import annotations

import json
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_SUBQUERIES = 3

_SYSTEM_PROMPT = (
    "你是知识库检索查询分解助手。把复合问题拆成独立主题子问句，每个只问一件事，"
    "并把口语措辞归一成正式主题表述；单主题问题原样返回。不要回答，不要扩写。"
    "只输出 JSON 字符串数组。示例："
    "输入「驾驶公务车辆出差怎么报销，有什么制度」"
    "输出 [\"差旅费用报销有什么制度规定\", \"公务车辆派遣管理有什么制度\"]；"
    "输入「ZX-2026 的额定电压是多少」输出 [\"ZX-2026 的额定电压是多少\"]。"
)


def decompose(query: str) -> list[str]:
    """返回子问句列表；单主题、开关关闭或任何失败都返回 [query]。"""
    if not settings.RAG_QUERY_DECOMPOSE_ENABLED:
        return [query]
    from app.services.llm_service import LLMService
    try:
        resp = LLMService().chat(query=query, system_prompt=_SYSTEM_PROMPT, temperature=0.0)
        text = resp["answer"].strip()
        try:
            subs = json.loads(text)
        except json.JSONDecodeError:
            # 模型偶尔包 ```json 围栏或带前后废话：截取最外层数组再解析
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end <= start:
                raise
            subs = json.loads(text[start:end + 1])
        if not isinstance(subs, list):
            raise ValueError("分解结果不是数组")
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
        if not subs:
            raise ValueError("分解结果为空")
    except Exception as exc:
        logger.warning("查询分解失败，按原句检索：%s", exc)
        return [query]
    subs = list(dict.fromkeys(subs))[:MAX_SUBQUERIES]
    if subs == [query]:
        return [query]
    logger.info("查询分解：%s -> %s", query, subs)
    return subs