# -*- coding: utf-8 -*-
"""RAG 第二阶段精排：用 bge-reranker-v2-m3 交叉编码器给候选片段重新打分。

为什么加这一层（docs/04 的实测结论）：
- bge-m3 双塔向量只能粗筛，正确问题与无答案问题的 top1 cosine 区间重叠
  （[0.5808, 0.6790]），单阈值无法同时保证「不漏答」和「能拒答」；
- 交叉编码器把 (query, passage) 成对编码，能区分「主题相近但没答到点上」的片段。

设计取舍：
- 默认关闭（RAG_RERANK_ENABLED=false）：torch + 权重体积大，不是所有部署都需要；
  开启后若依赖/权重缺失，自动回退纯 cosine 检索并打 warning，不影响服务启动。
- 候选池放宽到 RAG_RERANK_CANDIDATE_K（默认 6；45 题实测 K=6 已 100% 召回，
  K=12 质量相同但 CPU 耗时约 2.4 倍），先召回再精排，避免精排只能看到 top3。
- 精排分数是 sigmoid(logit)，取值 (0,1)，不是 cosine，阈值需另标（见 docs/05）。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import replace

from app.core.config import settings

logger = logging.getLogger(__name__)


class RerankerUnavailableError(RuntimeError):
    """依赖缺失 / 权重加载失败时抛出，由调用方回退 cosine 检索。"""


class CrossEncoderReranker:
    """bge-reranker 交叉编码器封装（线程安全的懒加载单例）。"""

    def __init__(self, model_ref: str | None = None):
        # model_ref 供离线评测脚本指定权重目录；服务运行时留空，走 settings
        self._model_ref_override = model_ref
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = settings.RAG_RERANK_DEVICE
        self._load_error: str | None = None
        self._load_attempted = False

    # ---------------------------------------------------------------- 加载
    def _model_ref(self) -> str:
        return (
            self._model_ref_override
            or settings.RAG_RERANK_MODEL_PATH
            or settings.RAG_RERANK_MODEL
        )

    def load(self) -> bool:
        """加载模型。成功返回 True；失败返回 False 并记住原因，只尝试一次。"""
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if self._load_attempted:
                return False
            self._load_attempted = True
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except Exception as exc:  # pragma: no cover - 取决于部署环境
                self._load_error = f"缺少 torch/transformers 依赖：{exc}"
                logger.warning("RAG rerank 不可用，回退 cosine 检索：%s", self._load_error)
                return False
            try:
                model_ref = self._model_ref()
                tokenizer = AutoTokenizer.from_pretrained(model_ref)
                model = AutoModelForSequenceClassification.from_pretrained(model_ref)
                device = self._device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                model.eval()
                self._torch = torch
                self._tokenizer = tokenizer
                self._model = model
                self._device = device
                logger.info("RAG rerank 模型已加载：%s（device=%s）", model_ref, device)
                return True
            except Exception as exc:  # pragma: no cover - 权重缺失/损坏
                self._load_error = f"模型加载失败（{self._model_ref()}）：{exc}"
                logger.warning("RAG rerank 不可用，回退 cosine 检索：%s", self._load_error)
                return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str:
        return self._device

    # ---------------------------------------------------------------- 推理
    def score(self, query: str, passages: list[str]) -> list[float]:
        """一次批量算 (query, passage) 相关性，返回 sigmoid(logit) ∈ (0,1)。"""
        if not self.load():
            raise RerankerUnavailableError(self._load_error or "reranker 未加载")
        pairs = [[query, p] for p in passages]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=settings.RAG_RERANK_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits.view(-1)
            return self._torch.sigmoid(logits.float()).tolist()


# 进程级单例：模型只加载一次，避免每个请求重复吃内存
_reranker = CrossEncoderReranker()


def get_reranker() -> CrossEncoderReranker:
    return _reranker


def is_available() -> bool:
    """开启精排且模型可用（首次调用会触发加载，失败只记一次，不会反复重试）。"""
    return bool(settings.RAG_RERANK_ENABLED) and _reranker.load()


def rerank_chunks(query: str, chunks, top_k: int | None = None):
    """给 RetrievedChunk 列表重打分并排序；score 换成精排概率（不再是 cosine）。

    默认返回全部候选（由调用方先按精排阈值过滤、再截断 top_k），
    避免「先截断后过滤」把第 4 名之后过线的片段丢掉。
    """
    if not chunks:
        return []
    scores = _reranker.score(query, [c.content for c in chunks])
    rescored = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored if top_k is None else rescored[:top_k]
