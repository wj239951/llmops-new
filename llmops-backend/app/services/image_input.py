# 图片输入：用户随消息发来的图片 -> OCR -> 与提问合并后检索（docs/22）。
from __future__ import annotations

import base64
import binascii
import io
import logging
import re
import time

from app.core.config import settings

logger = logging.getLogger(__name__)

# OCR 引擎懒加载：首次调用才导入（依赖较重），失败后不再重试
_engine = None
_engine_error: str | None = None


def _get_engine():
    global _engine, _engine_error
    if _engine is not None or _engine_error is not None:
        return _engine
    try:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
        logger.info("图片输入：RapidOCR 引擎已加载")
    except Exception as exc:  # 没装依赖也要能跑，只是图片通道不可用
        _engine_error = f"{type(exc).__name__}: {exc}"
        logger.warning("图片输入：OCR 引擎不可用，图片通道降级忽略（%s）", _engine_error)
    return _engine


def decode_image(image_base64: str | None) -> bytes | None:
    """解析 data URI 或裸 base64；超限/非法返回 None。"""
    raw = (image_base64 or "").strip()
    if not raw:
        return None
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        return None
    if not data or len(data) > settings.IMAGE_INPUT_MAX_BYTES:
        return None
    return data


def tidy_text(text: str) -> str:
    """OCR 结果的轻量整理：行内空白压缩、去空行、相邻重复行去重。"""
    lines = []
    for line in (text or "").splitlines():
        line = re.sub(r"[ \t\u3000]+", " ", line).strip()
        if not line:
            continue
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    return "\n".join(lines)


def merge_query(image_text: str, query: str) -> str:
    """图片文字在前、用户提问在后：检索时图片内容提供上下文，提问决定意图。"""
    it = (image_text or "").strip()
    qt = (query or "").strip()
    if not it:
        return qt
    if not qt:
        return it
    return f"{it}\n\n{qt}"


def _lines_from_result(items) -> str:
    """按行聚合 OCR 框：中心 y 相近归为一行，行内按 x 排序。"""
    boxes = []
    for box, text, score in items:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 1.0
        if settings.IMAGE_INPUT_MIN_SCORE > 0 and score < settings.IMAGE_INPUT_MIN_SCORE:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        boxes.append({"cy": (min(ys) + max(ys)) / 2, "x": min(xs),
                      "h": max(ys) - min(ys), "t": text})
    if not boxes:
        return ""
    boxes.sort(key=lambda d: (d["cy"], d["x"]))
    lines, cur, cur_cy = [], [], None
    for box in boxes:
        if cur_cy is None or abs(box["cy"] - cur_cy) <= max(box["h"], 8) * 0.6:
            cur.append(box)
            cur_cy = sum(x["cy"] for x in cur) / len(cur)
        else:
            lines.append(sorted(cur, key=lambda d: d["x"]))
            cur, cur_cy = [box], box["cy"]
    if cur:
        lines.append(sorted(cur, key=lambda d: d["x"]))
    return "\n".join(" ".join(x["t"] for x in ln) for ln in lines)


def extract_image_text(image_base64: str | None) -> dict:
    """图片 -> 文本。返回的 dict 会进 ctx 并回传给前端。"""
    meta = {"applied": False, "engine": "none", "chars": 0, "ms": 0,
            "truncated": False, "text": "", "reason": None}
    if not image_base64:
        meta["reason"] = "no_image"
        return meta
    if not settings.IMAGE_INPUT_ENABLED:
        meta["reason"] = "disabled"
        return meta

    data = decode_image(image_base64)
    if data is None:
        meta["reason"] = "invalid_image"
        return meta

    engine = _get_engine()
    if engine is None:
        meta["reason"] = "engine_unavailable"
        return meta

    t0 = time.perf_counter()
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        # OCR 在超大图上很容易吃满内存；长边限制到 2000 足够看清手机拍的字
        if max(img.size) > 2000:
            img.thumbnail((2000, 2000))
        result, _ = engine(np.array(img))
    except Exception as exc:
        logger.warning("图片输入：OCR 失败 %s", exc)
        meta["reason"] = f"ocr_error: {type(exc).__name__}"
        return meta

    text = tidy_text(_lines_from_result(result or []))
    meta["engine"] = "rapidocr"
    meta["ms"] = int((time.perf_counter() - t0) * 1000)
    if len(text) > settings.IMAGE_INPUT_MAX_CHARS:
        text = text[: settings.IMAGE_INPUT_MAX_CHARS]
        meta["truncated"] = True
    meta["text"] = text
    meta["chars"] = len(text)
    meta["applied"] = bool(text)
    if not text:
        meta["reason"] = "no_text"
    return meta


def public_meta(meta: dict | None, max_chars: int = 500) -> dict | None:
    """给前端的精简版：不带完整超长文本。"""
    if not meta or not meta.get("applied"):
        return None
    return {
        "engine": meta.get("engine"),
        "chars": meta.get("chars"),
        "ms": meta.get("ms"),
        "truncated": meta.get("truncated"),
        "text": (meta.get("text") or "")[:max_chars],
    }
