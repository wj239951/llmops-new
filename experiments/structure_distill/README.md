# 多模态结构蒸馏实验脚本

配套报告：[docs/21.多模态结构蒸馏实验.md](../../docs/21.多模态结构蒸馏实验.md)

| 脚本 | 作用 |
|---|---|
| train_mixed.py | 图像 + 清洗文本混训（LoRA，视觉塔冻结） |
| infer_eval.py | 30 页评测集推理，输出每页 .md |
| eval_clean.py | 清洗能力回归（基线 vs adapter） |
| ocr_baseline.py | RapidOCR 两段式基线（对照组） |
| marathon_to_jsonl.py | MarathonDocBench 标注 -> 训练 JSONL |
| synth_pages_v3.py | 合成页面生成（模板渲染 + 自动真值） |

评测结果见 eval_results/structure_distill/。
