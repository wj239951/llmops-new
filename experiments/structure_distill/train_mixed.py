# -*- coding: utf-8 -*-
"""混训：结构蒸馏（图像->Markdown）+ 15% 清洗样本，防止原有清洗能力漂移。

与 train_remote.py 的区别：
1. 支持两类样本：{"image", "target"} 和 {"raw", "clean", "type": "text"}；
2. 清洗样本用原项目的 system prompt + 纯文本对话，只对 assistant 段算 loss；
3. 默认 900k 像素 / 2048 序列 —— 实测峰值显存约 1.8GB，6GB 卡可跑。
"""
import argparse
import json
import sys
import time

sys.path.append(r"D:\anaconda\Lib\site-packages")   # venv 里没有 PIL，借用 anaconda 的（保留可让 transformers 识别）

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForImageTextToText
from transformers.processing_utils import ProcessorMixin

ProcessorMixin.check_argument_for_proper_class = lambda self, attr, arg: None
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # noqa: E402
from peft import LoraConfig, get_peft_model                                  # noqa: E402

PROMPT = "Convert this document page into Markdown. Output only the Markdown content."
SYSTEM = ("你是输入文本预处理器。对用户给出的原始文本做清洗规范化："
          "删除不可见字符与噪声、修复乱码、去掉 HTML 标签、统一全角半角与标点、"
          "压缩无意义重复、整理空白与断行。只改格式不改内容，"
          "直接输出清洗后的文本，不要解释。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="D:/models/Qwen3.5-0.8B-TextPreprocessor")
    ap.add_argument("--data", default=r"D:\文本需求拆分\bench_eval\mixed_train.json")
    ap.add_argument("--out", default=r"D:\文本需求拆分\bench_eval\adapter_mixed")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-pixels", type=int, default=900_000)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()

    rows = json.load(open(args.data, encoding="utf-8"))
    n_img = sum(1 for r in rows if "image" in r)
    print(f"samples: {len(rows)} (image {n_img} / text {len(rows)-n_img})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    imgp = AutoImageProcessor.from_pretrained(args.model_dir)
    proc = Qwen3VLProcessor(image_processor=imgp, tokenizer=tok)

    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    size = {"shortest_edge": 65536, "longest_edge": args.max_pixels}

    def build_text(raw, clean):
        ptxt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": raw}],
            tokenize=False, add_generation_prompt=True)
        p_ids = tok(ptxt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        t_ids = tok(clean, return_tensors="pt", add_special_tokens=False)["input_ids"]
        t_ids = torch.cat([t_ids, torch.tensor([[tok.eos_token_id]])], dim=1)
        ids = torch.cat([p_ids, t_ids], dim=1)[:, :args.maxlen]
        labels = torch.cat([torch.full_like(p_ids, -100), t_ids], dim=1)[:, :args.maxlen]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": torch.ones_like(ids)}

    def build_image(path, target):
        img = Image.open(path).convert("RGB")
        side = int(args.max_pixels ** 0.5)
        if max(img.size) > side:
            img.thumbnail((side, side), Image.LANCZOS)
        text = tok.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
            tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=[img], return_tensors="pt", size=size)
        t_ids = tok(target, return_tensors="pt", add_special_tokens=False)["input_ids"]
        t_ids = torch.cat([t_ids, torch.tensor([[tok.eos_token_id]])], dim=1)
        ids = torch.cat([enc["input_ids"], t_ids], dim=1)[:, :args.maxlen]
        labels = torch.cat([torch.full_like(enc["input_ids"], -100), t_ids], dim=1)[:, :args.maxlen]
        mm = torch.cat([enc["mm_token_type_ids"], torch.zeros_like(t_ids)], dim=1)[:, :args.maxlen]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": torch.ones_like(ids), "mm_token_type_ids": mm,
                "pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"]}

    step, t0, running, skipped = 0, time.time(), [], 0
    for epoch in range(args.epochs):
        for item in rows:
            try:
                if item.get("type") == "text":
                    batch = build_text(item["raw"], item["clean"])
                else:
                    batch = build_image(item["image"], item["target"])
                batch = {k: v.to(model.device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward()
                opt.step()
            except torch.cuda.OutOfMemoryError:
                skipped += 1
                torch.cuda.empty_cache()
                print(f"skip oom (total {skipped})", flush=True)
                continue
            finally:
                opt.zero_grad(set_to_none=True)
                if step % 50 == 0:
                    torch.cuda.empty_cache()
            step += 1
            running.append(loss.item())
            if step % 10 == 0:
                avg = sum(running[-25:]) / len(running[-25:])
                print(f"[e{epoch+1} {step}] loss={avg:.4f} len={batch['input_ids'].shape[1]} "
                      f"gpu={torch.cuda.memory_allocated()/1e9:.1f}GB {time.time()-t0:.0f}s", flush=True)
            if args.steps and step >= args.steps:
                print("debug stop", flush=True)
                return
    model.save_pretrained(args.out)
    print("adapter saved:", args.out, flush=True)


if __name__ == "__main__":
    main()
