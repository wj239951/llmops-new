# -*- coding: utf-8 -*-
"""清洗能力回归测试：基线模型 vs 结构 adapter，在 rule_test / teacher_test 上对比。

指标沿用原项目 src/eval/eval_model.py 的定义：
exact（完全一致）、sim（字符相似度）、loss（漏删率）、add（新增率）。
"""
import json
import sys
import time
from difflib import SequenceMatcher

sys.path.append(r"D:\anaconda\Lib\site-packages")
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel

MP = "D:/models/Qwen3.5-0.8B-TextPreprocessor"
ADAPTER = r"D:\文本需求拆分\bench_eval\adapter"
REPO = r"D:\文本需求拆分\.tmp\model-repo\Qwen3.5-0.8B-TextPreprocessor-main"
SETS = {"rule": (REPO + r"\dataset\rule_test.jsonl", 50),
        "teacher": (REPO + r"\dataset\teacher_test.jsonl", 40)}
SYSTEM = ("你是输入文本预处理器。对用户给出的原始文本做清洗规范化："
          "删除不可见字符与噪声、修复乱码、去掉 HTML 标签、统一全角半角与标点、"
          "压缩无意义重复、整理空白与断行。只改格式不改内容，"
          "直接输出清洗后的文本，不要解释。")


def load(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def char_metrics(pred, ref):
    if not ref:
        return 0.0, 0.0, 1.0
    m = SequenceMatcher(None, ref, pred, autojunk=False)
    matched = sum(b.size for b in m.get_matching_blocks())
    return (max(0.0, (len(ref) - matched) / len(ref)),
            max(0.0, (len(pred) - matched) / max(1, len(ref))), m.ratio())


def score(preds, refs):
    n = len(refs)
    ex = sum(1 for p, r in zip(preds, refs) if p.strip() == r.strip())
    rows = [char_metrics(p.strip(), r.strip()) for p, r in zip(preds, refs)]
    return {"n": n, "exact": ex / n, "sim": sum(r[2] for r in rows) / n,
            "loss": sum(r[0] for r in rows) / n, "add": sum(r[1] for r in rows) / n}


def main():
    tok = AutoTokenizer.from_pretrained(MP)
    model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    def gen(raw, ref):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": raw}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        mx = min(512, max(64, int(len(ref) * 1.6) + 16))
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=mx, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    report = {}
    for name, (path, limit) in SETS.items():
        rows = load(path, limit)
        refs = [r["clean"] for r in rows]
        raws = [r["raw"] for r in rows]
        t0 = time.time()
        preds_base = [gen(r, c) for r, c in zip(raws, refs)]
        print(f"[{name}] base done {time.time()-t0:.0f}s", flush=True)
        report[f"{name}_base"] = score(preds_base, refs)
        report[f"{name}_base"]["preds"] = preds_base
        report[f"{name}_rows"] = [{"raw": r, "ref": c, "base": p} for r, c, p in zip(raws, refs, preds_base)]

    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print("adapter loaded", flush=True)

    for name in SETS:
        rows = report[f"{name}_rows"]
        t0 = time.time()
        preds = [gen(r["raw"], r["ref"]) for r in rows]
        print(f"[{name}] adapter done {time.time()-t0:.0f}s", flush=True)
        report[f"{name}_adapter"] = score(preds, [r["ref"] for r in rows])
        for r, p in zip(rows, preds):
            r["adapter"] = p

    out = r"D:\文本需求拆分\bench_eval\clean_regression.json"
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n| 数据集 | 模型 | n | 完全一致 | 相似度 | 漏删率 | 新增率 |")
    print("|---|---|---|---|---|---|---|")
    for name in SETS:
        for tag in ("base", "adapter"):
            m = report[f"{name}_{tag}"]
            print(f"| {name} | {tag} | {m['n']} | {m['exact']:.1%} | {m['sim']:.3f} "
                  f"| {m['loss']:.3f} | {m['add']:.3f} |")
    print("saved:", out)


if __name__ == "__main__":
    main()
