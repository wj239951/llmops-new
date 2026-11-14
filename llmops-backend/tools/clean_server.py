# -*- coding: utf-8 -*-
"""独立输入清洗服务：把蒸馏出的 Qwen3.5-0.8B-TextPreprocessor 包成 HTTP 接口。

为什么单独起一个进程，而不是在后端里 import 模型：
- 后端是 FastAPI 多 worker + 单机 2G 内存的部署形态，塞一个 1.6GB 的 torch 模型会直接 OOM；
- 清洗是可选能力，起不来也不该拖垮问答主链路（后端会按 INPUT_CLEAN_FALLBACK 降级）。

启动（在 llmops-backend 目录下）：
    pip install -r requirements-clean.txt
    python tools/clean_server.py --model Qwen3.5-0.8B-TextPreprocessor --port 8010

后端侧配置：
    INPUT_CLEAN_MODE=model
    INPUT_CLEAN_SERVER_URL=http://127.0.0.1:8010

接口：
    GET  /health  -> {"ok": true, "model": "...", "loaded": true}
    POST /clean   -> {"text": "..."}  => {"clean": "...", "ms": 12, "mojibake_fixed": false}
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# 复用后端的乱码还原：模型不负责乱码，这一步必须用确定性代码先做掉
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.text_clean import repair_mojibake_only  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("clean_server")

# 与模型卡里的系统提示词逐字一致，改一个标点都会让效果偏移
SYSTEM_PROMPT = (
    "你是输入文本预处理器。对用户给出的原始文本做清洗规范化："
    "删除不可见字符与噪声、修复乱码、去掉 HTML 标签、统一全角半角与标点、"
    "压缩无意义重复、整理空白与断行。只改格式不改内容，"
    "直接输出清洗后的文本，不要解释。"
)

# 超过这个长度不收：模型首字延迟会明显上升，且长文清洗收益递减
MAX_CHARS = 3000

app = FastAPI(title="Text Clean Server", version="1.0.0")

_state: dict = {"model": None, "tokenizer": None, "path": "", "error": None, "lock": threading.Lock()}


class CleanRequest(BaseModel):
    text: str


def _load_model(model_path: str) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True, device_map="auto",
    ).eval()
    _state.update(model=model, tokenizer=tokenizer, path=model_path, error=None)
    logger.info("清洗模型加载完成：%s（device_map=auto，dtype=%s）", model_path, dtype)


def _warmup(model_path: str) -> None:
    try:
        _load_model(model_path)
        # 跑一次空转，把 kernel 编译 / 显存分配提前到启动阶段
        _generate("预热")
        logger.info("预热完成")
    except Exception as exc:  # noqa: BLE001
        _state["error"] = str(exc)
        logger.exception("清洗模型加载失败：%s", exc)


def _generate(text: str) -> str:
    import torch

    tokenizer, model = _state["tokenizer"], _state["model"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    budget = min(1024, max(64, int(len(text) * 1.4) + 64))
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=budget, do_sample=False, num_beams=1,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@app.get("/health")
def health():
    return {
        "ok": _state["model"] is not None,
        "model": _state["path"],
        "loaded": _state["model"] is not None,
        "error": _state["error"],
    }


@app.post("/clean")
def clean(req: CleanRequest):
    started = time.perf_counter()
    raw = req.text or ""
    if not raw.strip():
        return {"clean": "", "ms": 0, "mojibake_fixed": False}
    if len(raw) > MAX_CHARS:
        # 超长直接回原文，让后端自己决定要不要退规则清洗
        return {"clean": raw, "ms": 0, "mojibake_fixed": False, "skipped": "too_long"}
    if _state["model"] is None:
        raise RuntimeError(f"清洗模型未就绪：{_state['error'] or '尚未加载'}")

    fixed = repair_mojibake_only(raw)
    mojibake_fixed = fixed != raw
    with _state["lock"]:  # 单卡串行，避免并发下显存抖动
        cleaned = _generate(fixed)
    return {
        "clean": cleaned or fixed,
        "ms": int((time.perf_counter() - started) * 1000),
        "mojibake_fixed": mojibake_fixed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CLEAN_MODEL_PATH", "Qwen3.5-0.8B-TextPreprocessor"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLEAN_PORT", "8010")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    if not args.no_warmup:
        threading.Thread(target=_warmup, args=(args.model,), daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
