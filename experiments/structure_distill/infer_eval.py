# -*- coding: utf-8 -*-
"""用微调后的 adapter 跑 30 页评测集，输出 .md（文件名 = 图片名去扩展名）。"""
import argparse
import glob
import os
import time

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForImageTextToText
from transformers.processing_utils import ProcessorMixin

ProcessorMixin.check_argument_for_proper_class = lambda self, attr, arg: None
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # noqa: E402
from peft import PeftModel                                                     # noqa: E402

PROMPT = "Convert this document page into Markdown. Output only the Markdown content."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/root/autodl-tmp/model")
    ap.add_argument("--adapter", default="/root/autodl-tmp/adapter")
    ap.add_argument("--images", default="/root/autodl-tmp/eval/images")
    ap.add_argument("--out", default="/root/autodl-tmp/pred_adapter")
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("--max-new", type=int, default=1200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    imgp = AutoImageProcessor.from_pretrained(args.model_dir)
    proc = Qwen3VLProcessor(image_processor=imgp, tokenizer=tok)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map="auto")
    if args.adapter and os.path.isdir(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter)
        print("adapter loaded:", args.adapter, flush=True)
    model.eval()

    files = sorted(glob.glob(os.path.join(args.images, "*")))
    print("pages:", len(files), flush=True)
    for i, f in enumerate(files, 1):
        img = Image.open(f).convert("RGB")
        if max(img.size) > args.max_side:
            img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        text = proc.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = proc(text=[text], images=[img], return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 no_repeat_ngram_size=6)
        txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        stem = os.path.splitext(os.path.basename(f))[0]
        open(os.path.join(args.out, stem + ".md"), "w", encoding="utf-8").write(txt)
        print(f"[{i}/{len(files)}] {stem[:40]} {len(txt)} chars {time.time()-t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
