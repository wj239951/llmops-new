# -*- coding: utf-8 -*-
"""混合检索：jieba / BM25 关键词召回 + bge-m3 向量召回的融合层。

背景（docs/04、docs/05 的实测结论）：
- 纯向量召回对「主题相近但没答到点上」的片段打分偏低，需要精排；
- 但另一类查询向量召回天然不占优：精确编号 / 型号 / 数字（RL-2077、E-073、SX-12V、25m…），
  它们靠子串是否命中区分，关键词通道更稳。
所以把仓库里第一代 jieba 关键词检索（原实现以注释形式留在 rag_service.py）改造成
BM25 稀疏召回，与向量召回做 RRF（Reciprocal Rank Fusion）融合，再把候选池交给
（可选的）交叉编码器精排。

设计取舍：
- 默认关闭（RAG_HYBRID_ENABLED=false）：向量检索已经是默认主链路，混合检索是可选增强。
- 索引构建零额外存储：直接读 Chroma collection 里已有的片段（正文 + metadata + 向量），
  在内存里建 BM25；文档增删后由 RAGService 调用 invalidate() 触发懒重建。
- 中文分词用 jieba.lcut_for_search（与第一代关键词检索一致）；英文/编号用正则整体保留，
  避免 RL-2077 被切成 "rl" / "2077" 两个无区分度的词。
- 融合只负责「进候选池」，最终排序仍交给精排或 cosine 阈值，保持阈值语义不变。
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)

_STOP_WORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Disable_word_list.txt")
_ZH_RE = re.compile(r"[\u4e00-\u9fff]+")
# 英文 / 型号 / 编号整体保留：rl-2077、e-073、zx-2026、0.15mpa、wns2-1.25-q
_LATIN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

_jieba = None
_stop_words: set[str] | None = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        import jieba  # 延迟导入：未开启混合检索时不为它付启动成本

        jieba.setLogLevel(logging.WARNING)
        _jieba = jieba
    return _jieba


def _get_stop_words() -> set[str]:
    global _stop_words
    if _stop_words is None:
        try:
            with open(_STOP_WORDS_FILE, encoding="utf-8") as f:
                _stop_words = set(f.read().split())
        except OSError:
            _stop_words = set()
    return _stop_words


def tokenize(text: str) -> list[str]:
    """中文用 jieba 搜索模式分词，英文/编号整体保留，过滤停用词。"""
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = _LATIN_RE.findall(lowered)
    jieba = _get_jieba()
    for segment in _ZH_RE.findall(lowered):
        tokens.extend(t for t in jieba.lcut_for_search(segment) if len(t) >= 2)
    stop_words = _get_stop_words()
    return [t for t in tokens if t not in stop_words]


@dataclass(frozen=True)
class HybridDoc:
    """Chroma 里一条片段的内存表示（与向量库一致，不额外落库）。"""

    vec_id: str
    doc_id: int | None
    title: str
    chunk: int | None
    content: str
    embedding: tuple[float, ...] | None = None


class BM25Index:
    """极简 BM25（Okapi），语料是 Chroma 里的全部片段。"""

    def __init__(self, docs: list[HybridDoc], k1: float = 1.5, b: float = 0.75):
        self._docs = docs
        self._k1 = k1
        self._b = b
        self._corpus = [tokenize(d.content) for d in docs]
        self._doc_len = [len(tokens) for tokens in self._corpus]
        self._avgdl = (sum(self._doc_len) / len(docs)) if docs else 0.0
        self._tf: list[Counter] = [Counter(tokens) for tokens in self._corpus]
        df: Counter = Counter()
        for tokens in self._corpus:
            df.update(set(tokens))
        n = len(docs)
        self._idf = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()
        }

    def search(self, query: str, k: int) -> list[tuple[HybridDoc, float]]:
        q_terms = set(tokenize(query))
        if not q_terms or not self._docs:
            return []
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for term in q_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self._k1 * (
                    1.0 - self._b + self._b * self._doc_len[i] / (self._avgdl or 1.0)
                )
                score += self._idf[term] * freq * (self._k1 + 1.0) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(self._docs[i], score) for score, i in scored[:k]]


def _build_docs(data: dict) -> list[HybridDoc]:
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    embeddings = data.get("embeddings")
    docs: list[HybridDoc] = []
    for i, vec_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        embedding = None
        if embeddings is not None and i < len(embeddings) and embeddings[i] is not None:
            embedding = tuple(float(x) for x in embeddings[i])
        docs.append(
            HybridDoc(
                vec_id=vec_id,
                doc_id=meta.get("id"),
                title=meta.get("title", ""),
                chunk=meta.get("chunk"),
                content=documents[i] if i < len(documents) else "",
                embedding=embedding,
            )
        )
    return docs


class HybridIndex:
    """进程级懒加载索引：从 Chroma 现有片段建内存 BM25，文档变更后 invalidate 重建。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._docs: list[HybridDoc] | None = None
        self._bm25: BM25Index | None = None
        self._attempted = False
        self._load_error: str | None = None

    # ---------------------------------------------------------------- 加载
    def ensure_loaded(self) -> bool:
        if self._docs is not None:
            return True
        if self._attempted:
            return False
        with self._lock:
            if self._docs is not None:
                return True
            if self._attempted:
                return False
            self._attempted = True
            try:
                from app.services.rag_service import vectorstore

                data = vectorstore._collection.get(
                    include=["documents", "metadatas", "embeddings"]
                )
                docs = _build_docs(data)
                self._docs = docs
                self._bm25 = BM25Index(docs)
                logger.info("RAG 混合检索索引已构建：%d 条片段（BM25 + jieba）", len(docs))
                return True
            except Exception as exc:  # pragma: no cover - 取决于部署环境
                self._load_error = f"混合检索索引构建失败：{exc}"
                logger.warning("RAG 混合检索不可用，回退纯向量检索：%s", self._load_error)
                return False

    def invalidate(self) -> None:
        """文档增删改后调用：下次检索时按最新 Chroma 内容重建索引。"""
        with self._lock:
            self._docs = None
            self._bm25 = None
            self._attempted = False
            self._load_error = None

    @property
    def docs(self) -> list[HybridDoc]:
        return self._docs or []

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def bm25_search(self, query: str, k: int) -> list[tuple[HybridDoc, float]]:
        if not self.ensure_loaded() or self._bm25 is None:
            return []
        return self._bm25.search(query, k)


_index = HybridIndex()


def get_index() -> HybridIndex:
    return _index


def is_available() -> bool:
    """开启混合检索且索引可用；失败只尝试一次，避免每个请求都重建。"""
    return bool(settings.RAG_HYBRID_ENABLED) and _index.ensure_loaded()


def invalidate() -> None:
    _index.invalidate()


def bm25_search(query: str, k: int) -> list[tuple[HybridDoc, float]]:
    return _index.bm25_search(query, k)


def embed_query(text: str) -> list[float]:
    """复用向量库的 embedding 通道，给 BM25 独有候选补 cosine 分数（只在未开精排时需要）。"""
    from app.services.rag_service import embeddings

    return embeddings.embed_query(text)


def cosine(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def rrf_score(rank: int, rrf_k: int) -> float:
    """RRF 标准公式：每个通道按名次贡献 1/(k + rank)，名次从 1 开始。"""
    return 1.0 / (rrf_k + rank)
