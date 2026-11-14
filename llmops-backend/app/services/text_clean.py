# -*- coding: utf-8 -*-
"""输入清洗：把用户粘贴的脏文本规范化后再进检索 / 生成。

为什么要在 RAG 前面加这一层：
- 复制粘贴的工业文档常带零宽字符、全角空格、HTML 实体、连续重复字（`我我我`）、
  以及 PDF 折断的硬换行。这些东西直接进 bge-m3 会把向量带偏，命中率掉得很难看；
- 型号 / 编号类查询一旦被全角字符污染（`ＺＸ－２０２６`），混合检索的精确编号豁免也会失效。

两条实现通道（与精排同样的「可选 + 失败降级」语义）：
- ``rule``：纯标准库，进程内几毫秒完成，零依赖、零显存；
- ``model``：调一个独立的小服务（``tools/clean_server.py``，加载蒸馏出的
  Qwen3.5-0.8B-TextPreprocessor 0.8B 模型），判断力更强但需要 GPU/额外进程。

默认 ``off``。开启后失败一律按 ``INPUT_CLEAN_FALLBACK`` 回退，不影响主链路。

清洗规则来源：https://github.com/wj239951/Qwen3.5-0.8B-TextPreprocessor
（本文件是精简移植版：保留对 RAG 有实际收益的步骤，去掉了段落级硬换行合并等长尾逻辑，
两边行为差异见 docs/08。）
"""
from __future__ import annotations

import html
import logging
import re
import time
import unicodedata

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── 字符表 ──────────────────────────────────────────────────────────────
# 不可能是单字节编码字节值的零宽字符：可以最先删
_INVISIBLE = frozenset(
    "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
    "\u2060\u2061\u2062\u2063\u2064\u2066\u2067\u2068\u2069"
    "\ufeff\u00ad\u180e\u034f\u115f\u1160\uffa0\u061c"
)
# 全角字母数字 → 半角（中文标点如 ，。！？ 不在此列，必须保留全角）
_FULLWIDTH = {
    **{chr(0xFF10 + i): chr(0x30 + i) for i in range(10)},
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},
    **{
        "\uff20": "@", "\uff03": "#", "\uff04": "$", "\uff05": "%", "\uff06": "&",
        "\uff0a": "*", "\uff0b": "+", "\uff0d": "-", "\uff0f": "/", "\uff1c": "<",
        "\uff1e": ">", "\uff1d": "=", "\uff3b": "[", "\uff3d": "]", "\uff3e": "^",
        "\uff3f": "_", "\uff5c": "|", "\uff5e": "~", "\uff3c": "\\",
    },
}
_SPACE_LIKE = {
    "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u200a": " ",
    "\u202f": " ", "\u205f": " ", "\u3000": " ", "\t": " ",
}
_HALF_TO_CJK = {",": "，", ";": "；", ":": "：", "!": "！", "?": "？", "(": "（", ")": "）"}

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_RUN = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff])\1{2,}")
_LATIN_RUN = re.compile(r"([A-Za-z])\1{2,}")

_HTML_TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]{0,9})(?:\s[^<>]{0,300})?/?>")
_HTML_PARA = re.compile(r"</(?:p|div|h[1-6]|li|tr|section|article|blockquote|pre)>", re.I)
_HTML_LINE = re.compile(r"<(?:br|hr)\s*/?>", re.I)

# URL / 邮箱 / 行内代码：原样保护，避免宽度与标点规则改写
_PROTECT = [
    re.compile(r"```.*?```", re.S),
    re.compile(r"`[^`\n]{1,200}`"),
    re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"'，。；）】]+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
]
_SPAN_OPEN, _SPAN_CLOSE = "\ue000", "\ue001"

_TERMINAL = "。！？!?：:；;，,、…）)]}】」』”\"'"
_MOJIBAKE_HINTS = frozenset("ÃÂâãäåæçèéêëìíîïðñòóôõöùúûüýþÿÅÆØÐÞŠŽŸ")
_RUN_RE = re.compile(r"[^ \n\t\r\f\v\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]{4,}")


def is_cjk(ch: str) -> bool:
    return bool(ch) and bool(_CJK.match(ch))


# ── 各步骤 ──────────────────────────────────────────────────────────────
def _strip_invisible(text: str) -> str:
    out = []
    for ch in text:
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        if ch in _INVISIBLE:
            continue
        if unicodedata.category(ch) in ("Cc", "Cf", "Cs"):
            continue
        out.append(ch)
    return "".join(out)


def _repair_segment(seg: str) -> str | None:
    """UTF-8 被按 CP1252/Latin-1 解码的乱码还原；只在高置信时动手。"""
    if not seg.strip():
        return None
    base = _quality(seg)
    best: tuple[float, str] | None = None
    for enc in ("cp1252", "latin-1"):
        try:
            cand = seg.encode(enc, errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
            continue
        if "\ufffd" in cand:
            continue
        score = _quality(cand)
        if best is None or score > best[0]:
            best = (score, cand)
    if best is None or best[0] <= base + 0.15:
        return None
    return best[1]


def _quality(text: str) -> float:
    n = len(text) or 1
    cjk = sum(1 for ch in text if _CJK.match(ch)) / n
    hint = sum(1 for ch in text if ch in _MOJIBAKE_HINTS) / n
    return cjk - hint * 2 - (1 if "\ufffd" in text else 0)


def _repair_mojibake(text: str) -> str:
    if not any(ch in _MOJIBAKE_HINTS for ch in text):
        return text
    fixed = _repair_segment(text)
    if fixed is not None:
        return fixed

    # 整段不可编码时按行重试，再按「疑似乱码片段」重试
    # （常见于"正文正常 + 个别片段乱码"的混合文本）
    changed = False
    lines: list[str] = []
    for line in text.split("\n"):
        if any(c in _MOJIBAKE_HINTS for c in line):
            candidate = _repair_segment(line)
            if candidate is not None:
                lines.append(candidate)
                changed = True
                continue
        lines.append(line)
    if changed:
        return "\n".join(lines)

    def _one(m: re.Match) -> str:
        run = m.group(0)
        if not any(c in _MOJIBAKE_HINTS for c in run):
            return run
        return _repair_segment(run) or run

    return _RUN_RE.sub(_one, text)


def _strip_html(text: str) -> str:
    tags = _HTML_TAG.findall(text)
    if len(tags) < 2 and not _HTML_LINE.search(text) and not _HTML_PARA.search(text):
        if "&" in text and re.search(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d{2,5});", text):
            return html.unescape(text)
        return text
    text = html.unescape(text)
    text = _HTML_PARA.sub("\n\n", text)
    text = _HTML_LINE.sub("\n", text)
    return _HTML_TAG.sub("", text)


def _normalize_width(text: str) -> str:
    for src, dst in _SPACE_LIKE.items():
        text = text.replace(src, dst)
    text = "".join(" " if (ch != " " and unicodedata.category(ch) == "Zs") else ch for ch in text)
    return "".join(_FULLWIDTH.get(ch, ch) for ch in text)


def _normalize_punct(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        target = _HALF_TO_CJK.get(ch)
        if target is None or i == 0 or not is_cjk(chars[i - 1]):
            continue
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if is_cjk(nxt) or nxt in ("", "\n", " "):
            chars[i] = target
    text = "".join(chars)
    text = re.sub(r"([!?！？])\1+", r"\1", text)
    text = re.sub(r"。{3,}", "……", text)
    text = re.sub(r"。{2}", "。", text)
    text = re.sub(r"…{2,}", "……", text)
    text = re.sub(r"([，,、；;：:])\1+", r"\1", text)
    return text


def _collapse_repeats(text: str) -> str:
    """同一汉字连打 ≥3 次压成 1 次；字母 ≥3 次压成 2 次。数字不动。"""
    text = _CJK_RUN.sub(r"\1", text)
    return _LATIN_RUN.sub(r"\1\1", text)


def _join_hard_wraps(text: str) -> str:
    """合并排版折断的行：上一行够长、且没说完，下一行不像新句子时才接上。"""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        if out and out[-1] and len(out[-1]) >= 24 and out[-1][-1] not in _TERMINAL:
            nxt = line.lstrip()
            if nxt and not re.match(r"^(?:[-*+•·]|\d{1,2}[.)、]|#|>|\|)", nxt) \
                    and re.search(r"[0-9A-Za-z\u4e00-\u9fff]", nxt):
                left_cjk, right_cjk = is_cjk(out[-1][-1]), is_cjk(nxt[0])
                out[-1] = out[-1] + ("" if (left_cjk or right_cjk) else " ") + nxt
                continue
        out.append(line)
    return "\n".join(out)


def _normalize_space(text: str) -> str:
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[ ]*\n[ ]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repair_mojibake_only(text: str) -> str:
    """只做乱码还原（UTF-8 被按单字节编码解码）。

    模型通道用得到：蒸馏出的 0.8B 模型没有"修乱码"这个能力（训练时刻意剔除了这类样本，
    否则小模型会靠背文本蒙混、在真实输入上凭空编造），所以这一步必须在调模型之前用代码做掉。
    """
    if not text:
        return text
    return _repair_mojibake(text.replace("锟斤拷", ""))


def clean_rule(text: str) -> str:
    """纯规则清洗（无依赖，毫秒级）。空输入返回空串。"""
    if not text or not text.strip():
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\v", "\n").replace("\f", "\n")
    # 乱码要在删不可见字符之前修：原始字节里就含 C1 与软连字符
    text = text.replace("锟斤拷", "")
    text = _repair_mojibake(text)
    text = _strip_invisible(text)
    text = _strip_html(text)
    # 保护 URL / 邮箱 / 代码片段
    spans: list[str] = []

    def _keep(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"{_SPAN_OPEN}{len(spans) - 1}{_SPAN_CLOSE}"

    for pattern in _PROTECT:
        text = pattern.sub(_keep, text)
    text = _normalize_width(text)
    text = _normalize_punct(text)
    text = _collapse_repeats(text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _join_hard_wraps(text)
    text = _normalize_space(text)

    def _restore(m: re.Match) -> str:
        idx = int(m.group(1))
        return spans[idx] if 0 <= idx < len(spans) else m.group(0)

    return re.sub(rf"{_SPAN_OPEN}(\d+){_SPAN_CLOSE}", _restore, text).strip()


def clean_model(text: str, url: str, timeout: float) -> str:
    """调独立清洗服务（蒸馏模型）。失败抛异常，由调用方决定降级。"""
    resp = requests.post(f"{url.rstrip('/')}/clean", json={"text": text}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    cleaned = data.get("clean")
    if not isinstance(cleaned, str):
        raise ValueError(f"清洗服务返回格式不对：{str(data)[:120]}")
    return cleaned


class InputCleaner:
    """统一入口：按 ``INPUT_CLEAN_MODE`` 选择通道，失败按配置回退。"""

    def __init__(self, mode: str | None = None, server_url: str | None = None,
                 timeout: float | None = None, fallback: str | None = None,
                 max_chars: int | None = None):
        # 参数可注入，测试用；生产走 settings
        self.mode = (mode or settings.INPUT_CLEAN_MODE or "off").lower()
        self.server_url = server_url or settings.INPUT_CLEAN_SERVER_URL
        self.timeout = timeout if timeout is not None else settings.INPUT_CLEAN_TIMEOUT
        self.fallback = (fallback or settings.INPUT_CLEAN_FALLBACK or "off").lower()
        self.max_chars = max_chars if max_chars is not None else settings.INPUT_CLEAN_MAX_CHARS

    @property
    def enabled(self) -> bool:
        return self.mode in ("rule", "model")

    def preprocess(self, query: str) -> tuple[str, dict]:
        """返回 (清洗后的文本, 元信息)。任何异常都不会向上抛。"""
        meta = {
            "mode": self.mode,
            "applied": False,
            "changed": False,
            "fallback": None,
            "reason": None,
            "ms": 0,
        }
        if not self.enabled:
            meta["reason"] = "off"
            return query, meta
        if not query or not query.strip():
            meta["reason"] = "empty"
            return query, meta
        if self.max_chars and len(query) > self.max_chars:
            # 超长输入不清洗：模型通道会拖慢首字，且长文清洗收益递减
            meta["reason"] = "too_long"
            return query, meta

        started = time.perf_counter()
        try:
            if self.mode == "model":
                cleaned = clean_model(query, self.server_url, self.timeout)
            else:
                cleaned = clean_rule(query)
        except Exception as exc:  # noqa: BLE001 - 任何失败都不该打断问答
            logger.warning("输入清洗失败（mode=%s），按 %s 回退：%s", self.mode, self.fallback, exc)
            meta["fallback"] = "rule" if self.fallback == "rule" else "off"
            if self.fallback == "rule":
                try:
                    cleaned = clean_rule(query)
                except Exception:  # noqa: BLE001 - 规则也失败就放弃
                    cleaned = query
                    meta["fallback"] = "off"
            else:
                cleaned = query
            meta["reason"] = "error"
        meta["ms"] = int((time.perf_counter() - started) * 1000)
        if not cleaned:
            # 清洗后为空说明整段是噪声，保留原文更能让用户发现问题
            meta["reason"] = "empty_after_clean"
            return query, meta
        meta["applied"] = True
        meta["changed"] = cleaned != query
        if not meta["changed"]:
            meta["reason"] = "already_clean"
        return cleaned, meta
