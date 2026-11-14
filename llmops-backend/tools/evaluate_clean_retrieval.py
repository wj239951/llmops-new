# -*- coding: utf-8 -*-
"""输入清洗对检索层的影响：端到端离线评测。

背景：docs/08 只量化了「清洗还原率」，没证明清洗能让检索变好——因为 45 条评测题本身
是干净手写问句，清洗对它们是 no-op。本脚本补上这块证据：

    干净问句 ──┐
    脏化问句 ──┼─► bge-m3 向量 ─► cosine top-K ─► 与 expected_doc_ids 比对
    脏化+规则 ─┤
    脏化+模型 ─┘

四种条件用同一份向量库、同一个切块器、同一套指标，差异只来自「送进编码器的问句」。

语料与切块完全复刻线上：`mysql-init/llmops_db.sql` 的 llm_knowledge_documents 表，
`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=[...])`，
embedding 走 Ollama 的 bge-m3（与 `EMBEDDING_PROVIDER=ollama` 一致）。

指标（与 tools/evaluate_rag.py 同定义）：
- recall@K：正确文档是否出现在 K 条候选里
- hit@3   ：前 3 条候选里是否命中正确文档
- MRR@K   ：正确文档排名的倒数（同篇多 chunk 只算首次出现）
- top1    ：第一条候选是否来自正确文档

用法（在 llmops-backend 目录执行）：
    .venv\\Scripts\\python.exe tools\\evaluate_clean_retrieval.py
    .venv\\Scripts\\python.exe tools\\evaluate_clean_retrieval.py --without-model
    .venv\\Scripts\\python.exe tools\\evaluate_clean_retrieval.py --candidate-k 6 --suite all
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import requests

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
SQL_PATH = REPO_DIR / "mysql-init" / "llmops_db.sql"
EXPERIMENTS = REPO_DIR / "experiments"
RESULTS = EXPERIMENTS / "results"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.text_clean import clean_rule  # noqa: E402

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

FULLWIDTH_OFFSET = 0xFEE0
ZERO_WIDTH = ["\u200b", "\u200c", "\u2060", "\ufeff"]

# 条件名 -> 题库里的字段名（干净原句存在 question 里）
QUERY_KEY = {"clean": "question", "dirty": "dirty", "rule": "rule", "model": "model"}


# ---------------------------------------------------------------- 语料
def parse_docs(sql_path: Path) -> list[dict]:
    """从 mysqldump 里抠出 llm_knowledge_documents 的 (id, title, content, source)。

    只解析这一条 INSERT；MySQL 的转义（\\' \\" \\\\ \\n \\r \\t）需要先还原。
    """
    text = sql_path.read_text(encoding="utf-8")
    marker = "INSERT INTO `llm_knowledge_documents` VALUES "
    start = text.index(marker) + len(marker)
    raw = text[start:]
    rows: list[list[str | None]] = []
    buf: list[str | None] = []
    i = 0
    in_str = False
    token: list[str] = []
    while i < len(raw):
        ch = raw[i]
        if in_str:
            if ch == "\\" and i + 1 < len(raw):
                nxt = raw[i + 1]
                token.append({"n": "\n", "r": "\r", "t": "\t", "0": "\0"}.get(nxt, nxt))
                i += 2
                continue
            if ch == "'":
                in_str = False
                buf.append("".join(token))
                token = []
                i += 1
                continue
            token.append(ch)
            i += 1
            continue
        if ch == "'":
            in_str = True
            i += 1
            continue
        if ch == "(":
            buf = []
            i += 1
            continue
        if ch == ")":
            rows.append(buf)
            buf = []
            i += 1
            continue
        if ch == ",":
            i += 1
            continue
        if ch == ";":
            break
        if ch.isdigit() and not buf:
            m = re.match(r"\d+", raw[i:])
            attr = m.group(0)
            buf.append(attr)
            i += len(attr)
            continue
        i += 1

    docs = []
    for row in rows:
        if len(row) < 3:
            continue
        docs.append({
            "id": int(row[0]) if row[0] and str(row[0]).isdigit() else None,
            "title": row[1] or "",
            "content": row[2] or "",
            "source": row[3] if len(row) > 3 else None,
        })
    return [d for d in docs if d["id"] is not None]


def build_chunks(docs: list[dict]) -> list[dict]:
    """复刻 rag_service 的切块配置。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS,
    )
    chunks: list[dict] = []
    for doc in docs:
        for idx, piece in enumerate(splitter.split_text(doc["content"])):
            if piece.strip():
                chunks.append({
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "chunk": idx,
                    "content": piece,
                })
    return chunks


# ---------------------------------------------------------------- 向量
class OllamaEmbedder:
    def __init__(self, model: str = "bge-m3", base_url: str = "http://127.0.0.1:11434", batch: int = 16):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch = batch

    def encode(self, texts: list[str], progress: str = "") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i: i + self.batch]
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": chunk},
                timeout=120,
            )
            resp.raise_for_status()
            out.extend(resp.json()["embeddings"])
            if progress:
                print(f"  向量化 {progress}: {min(i + self.batch, len(texts))}/{len(texts)}", end="\r")
        if progress:
            print()
        return out


def normalize(vecs: list[list[float]]) -> list[list[float]]:
    import math

    out = []
    for v in vecs:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


def top_k(query_vec: list[float], mat: list[list[float]], chunks: list[dict], k: int) -> list[dict]:
    scored = []
    for i, vec in enumerate(mat):
        scored.append((sum(a * b for a, b in zip(query_vec, vec)), i))
    scored.sort(reverse=True)
    return [
        {"doc_id": chunks[i]["doc_id"], "title": chunks[i]["title"],
         "chunk": chunks[i]["chunk"], "score": s}
        for s, i in scored[:k]
    ]


# ---------------------------------------------------------------- 脏化
def make_dirty(query: str, rng: random.Random, severity: str = "mild") -> tuple[str, list[str]]:
    """给干净问句注入真实会遇到的噪声。

    mild：零宽字符 + 全角化 + 多余空格 + 手抖重复（聊天框里手打/输入法造成的）
    heavy：mild 之外再加 HTML 包裹、`&nbsp;`、句中硬换行（网页/PDF 复制粘贴造成的）
    """
    kinds: list[str] = []
    text = query

    chars = list(text)
    for _ in range(rng.randint(1, 3)):
        chars.insert(rng.randrange(len(chars) + 1), rng.choice(ZERO_WIDTH))
    text = "".join(chars)
    kinds.append("zero_width")

    out = []
    for ch in text:
        if ch.isascii() and ch.isalnum() and rng.random() < 0.5:
            out.append(chr(ord(ch) + FULLWIDTH_OFFSET))
        else:
            out.append(ch)
    text = "".join(out)
    kinds.append("fullwidth")

    if rng.random() < 0.8:
        text = re.sub(r"([，。？、])", r"\1  ", text)
    kinds.append("spaces")

    cjk_pos = [i for i, ch in enumerate(text) if "\u4e00" <= ch <= "\u9fff"]
    if cjk_pos:
        pos = rng.choice(cjk_pos)
        text = text[: pos + 1] + text[pos] * rng.randint(2, 3) + text[pos + 1:]
        kinds.append("repeat")

    if severity == "heavy":
        # 网页 / PDF 粘贴的典型残留：实体化空格 + 块级标签 + 句中硬折行。
        # 折行只切在**汉字之间**——真实排版不会把 `E-073`、`&nbsp;` 从中间劈开，
        # 早期版本按中点盲切，制造了清洗器无法还原的假故障（见 docs/09 的说明）。
        text = text.replace(" ", "&nbsp;")
        cut = [i for i in range(4, len(text) - 4)
               if "\u4e00" <= text[i] <= "\u9fff" and "\u4e00" <= text[i - 1] <= "\u9fff"]
        if cut:
            pos = cut[len(cut) // 2]
            text = text[:pos] + "\n" + text[pos:]
            kinds.append("hard_wrap")
        text = f"<p>{text}</p>"
        kinds.append("html")
    return text, kinds


# ---------------------------------------------------------------- 指标
def first_doc_rank(cands: list[dict], expected: set[int]) -> int | None:
    seen: set[int] = set()
    rank = 0
    for cand in cands:
        if cand["doc_id"] in seen:
            continue
        seen.add(cand["doc_id"])
        rank += 1
        if cand["doc_id"] in expected:
            return rank
    return None


def score_one(cands: list[dict], expected: set[int], k: int, hit_k: int) -> dict:
    rank = first_doc_rank(cands, expected)
    return {
        "recall": 1.0 if rank is not None and rank <= k else 0.0,
        "hit": 1.0 if rank is not None and rank <= hit_k else 0.0,
        "mrr": (1.0 / rank) if rank else 0.0,
        "top1": 1.0 if rank == 1 else 0.0,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "recall": mean(r["recall"] for r in rows),
        "hit3": mean(r["hit"] for r in rows),
        "mrr": mean(r["mrr"] for r in rows),
        "top1": mean(r["top1"] for r in rows),
    }


# ---------------------------------------------------------------- 主流程
def load_suite(name: str, dirty_seed: int = 20261114, severity: str = "mild") -> list[dict]:
    files = {
        "main": ["eval_qa.jsonl"],
        "hard": ["eval_qa_hard.jsonl"],
        "token": ["eval_qa_token.jsonl"],
        "all": ["eval_qa.jsonl", "eval_qa_hard.jsonl", "eval_qa_token.jsonl"],
    }[name]
    rows: list[dict] = []
    for fn in files:
        with (EXPERIMENTS / fn).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rng = random.Random(dirty_seed)
    for row in rows:
        dirty, kinds = make_dirty(row["question"], rng, severity)
        row["dirty"] = dirty
        row["noise"] = kinds
        row["rule"] = clean_rule(dirty)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["main", "hard", "token", "all"])
    ap.add_argument("--noise", default="mild", choices=["mild", "heavy"], help="噪声强度：mild=手打/输入法，heavy=网页/PDF 粘贴")
    ap.add_argument("--candidate-k", type=int, default=6)
    ap.add_argument("--hit-k", type=int, default=3)
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--clean-server", default="http://127.0.0.1:8010")
    ap.add_argument("--without-model", action="store_true", help="不跑模型清洗通道")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    print(f"语料：{SQL_PATH}")
    docs = parse_docs(SQL_PATH)
    chunks = build_chunks(docs)
    print(f"  {len(docs)} 篇文档 -> {len(chunks)} 个片段（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")

    rows = load_suite(args.suite, severity=args.noise)
    print(f"题库：{len(rows)} 条（噪声强度 {args.noise}）")

    embedder = OllamaEmbedder(args.embed_model, args.ollama)
    print("向量化文档片段 …")
    doc_vecs = normalize(embedder.encode([c["content"] for c in chunks], "docs"))

    conditions = ["clean", "dirty", "rule"]
    if not args.without_model:
        conditions.append("model")
        try:
            health = requests.get(f"{args.clean_server}/health", timeout=5).json()
            if not health.get("loaded"):
                print(f"清洗服务未就绪（{health}），跳过 model 条件")
                conditions.remove("model")
        except Exception as exc:  # noqa: BLE001
            print(f"清洗服务不可用（{exc}），跳过 model 条件")
            conditions.remove("model")

    if "model" in conditions:
        t0 = time.perf_counter()
        cleaned = []
        for row in rows:
            resp = requests.post(f"{args.clean_server}/clean", json={"text": row["dirty"]}, timeout=60)
            resp.raise_for_status()
            cleaned.append(resp.json()["clean"])
        elapsed = (time.perf_counter() - t0) * 1000
        for row, clean in zip(rows, cleaned):
            row["model"] = clean
        print(f"模型清洗：{len(rows)} 条，共 {elapsed / 1000:.1f}s，平均 {elapsed / len(rows):.0f} ms/条")

    per_query: list[dict] = []
    summaries: dict[str, dict] = {}
    by_type: dict[str, dict[str, dict]] = defaultdict(dict)

    for cond in conditions:
        key = QUERY_KEY[cond]
        vecs = normalize(embedder.encode([row[key] for row in rows], cond))
        cond_rows = []
        for row, qvec in zip(rows, vecs):
            cands = top_k(qvec, doc_vecs, chunks, args.candidate_k)
            expected = set(row["expected_doc_ids"])
            scores = score_one(cands, expected, args.candidate_k, args.hit_k)
            cond_rows.append({**scores, "type": row.get("type", "unknown")})
            per_query.append({
                "id": row.get("id"), "type": row.get("type", "unknown"), "condition": cond,
                "query": row[key], "rank": first_doc_rank(cands, expected),
                "top1_doc": cands[0]["doc_id"] if cands else None,
                "expected": ",".join(str(x) for x in row["expected_doc_ids"]),
                **{k: round(v, 3) for k, v in scores.items()},
            })
        summaries[cond] = aggregate(cond_rows)
        for t in {r["type"] for r in cond_rows}:
            by_type[t][cond] = aggregate([r for r in cond_rows if r["type"] == t])

    restore = {"rule": sum(1 for r in rows if r["rule"] == r["question"]) / len(rows)}
    if "model" in conditions:
        restore["model"] = sum(1 for r in rows if r["model"] == r["question"]) / len(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"clean_retrieval_{args.suite}_{args.noise}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_query[0].keys()))
        writer.writeheader()
        writer.writerows(per_query)

    summary = {
        "suite": args.suite,
        "noise": args.noise,
        "n_questions": len(rows),
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "candidate_k": args.candidate_k,
        "hit_k": args.hit_k,
        "embed_model": args.embed_model,
        "metrics": summaries,
        "restore_rate": restore,
        "by_type": by_type,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = out_dir / f"clean_retrieval_{args.suite}_{args.noise}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'条件':<10}{'recall@' + str(args.candidate_k):>12}{'hit@' + str(args.hit_k):>10}{'MRR':>10}{'top1':>10}")
    for cond in conditions:
        m = summaries[cond]
        print(f"{cond:<10}{m['recall']:>11.1%}{m['hit3']:>10.1%}{m['mrr']:>10.3f}{m['top1']:>10.1%}")
    print(f"\n还原率：规则 {restore['rule']:.0%}" + (f"，模型 {restore['model']:.0%}" if "model" in restore else ""))
    print(f"明细 -> {csv_path}")
    print(f"汇总 -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
