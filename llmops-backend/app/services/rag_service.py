# 知识库业务逻辑，包括保存知识、检索知识、拼接上下文。##检索增强索引
from dataclasses import dataclass
from fastapi import HTTPException
#from langchain_community import embeddings
from sqlalchemy.orm import Session
from app.models.knowledge import KnowledgeDocument
#from sqlalchemy import or_
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from app.core.config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
#import jieba
import logging
import os

# 在文件顶部读取
stop_words_file = os.path.join(str(os.path.dirname(__file__)), "Disable_word_list.txt")

logger = logging.getLogger(__name__)


class SiliconFlowEmbeddings(Embeddings):
    """硅基流动 bge-m3 向量化。直接用 openai 客户端对接其 OpenAI 兼容端点,
    避开 langchain OpenAIEmbeddings 对 tiktoken/transformers 的依赖(容器里都没装)。"""

    def __init__(self, model: str, base_url: str, api_key: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _build_embeddings():
    # 按部署环境选向量化通道:本地有 Ollama 走它;云上没 Ollama 走硅基流动(同 bge-m3 权重,老向量兼容)
    if settings.EMBEDDING_PROVIDER == "siliconflow":
        return SiliconFlowEmbeddings(
            model=settings.SILICONFLOW_EMBEDDING_MODEL,
            base_url=settings.SILICONFLOW_EMBEDDING_BASE_URL,
            api_key=settings.SILICONFLOW_API_KEY,
        )
    return OllamaEmbeddings(
        model=settings.OLLAMA_EMBEDDING_MODEL,
        base_url=settings.OLLAMA_EMBEDDING_BASE_URL,
    )


# 必须在 _open_vectorstore() 之前实例化，后者引用它
embeddings = _build_embeddings()

# chroma_db 绝对路径：不依赖进程 CWD，从任意目录启动都能定位到同一个库
_CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "chroma_db",
)

_COLLECTION = "knowledge_bge_m3"
_SPACE = {"hnsw:space": "cosine"}


def _open_vectorstore():
    """打开/重建向量库。space 只在建 collection 时生效，旧 space 改不了，必须重建。"""
    client = chromadb.PersistentClient(path=_CHROMA_DIR)
    try:
        col = client.get_collection(_COLLECTION)
    except Exception:
        col = None
    if col is not None and (col.metadata or {}).get("hnsw:space") != "cosine":
        print(f"[rag_service] collection '{_COLLECTION}' 是旧 l2 空间，删除重建为 cosine。")
        client.delete_collection(_COLLECTION)   # 只有确认 space 不对才删
        col = None
    if col is None:
        client.create_collection(_COLLECTION, metadata=_SPACE)
        print(f"[rag_service] 已创建 cosine collection '{_COLLECTION}'；"
              f"向量需从 MySQL 重新灌：python tools/reindex_knowledge.py。")
    return Chroma(
        client=client,
        collection_name=_COLLECTION,
        embedding_function=embeddings,
    )


vectorstore = _open_vectorstore()


def _invalidate_hybrid_index() -> None:
    """文档增删改后让混合检索索引懒重建（未开启混合检索时是空操作）。"""
    from app.services import hybrid_service

    hybrid_service.invalidate()


# 长文档切分成小片段再向量化：既能避免超出 bge-m3 上下文长度，又能让检索命中更精准的段落。
# separators 加了中文标点，优先在句子边界切分。
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,#分块长度
    chunk_overlap=50,#分块重叠
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
with open(stop_words_file, encoding="utf-8") as f:
    stop_words = set(f.read().split())


@dataclass
class RetrievedChunk:
    title: str
    content: str
    source: str | None
    score: float          # 相关度分数：cosine（= 1 - distance）或精排 sigmoid，越大越相关
    # 片段在 Chroma 里的定位（混合检索反查/去重用）；旧调用方只传前四个字段，故给默认值
    doc_id: int | None = None
    chunk: int | None = None


class RAGService:
    def add_document(self, db: Session, title: str, content: str, source: str | None = None):
        doc = KnowledgeDocument(title=title, content=content, source=source)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        self._sync_to_chroma(doc)
        return doc

    def list_documents(self, db: Session):
        return db.query(KnowledgeDocument).order_by(KnowledgeDocument.id.desc()).all()
    # ── 检索链路 ─────────────────────────────────────────────────────────
    # 第一代 jieba 关键词检索（原先整段注释在这里：jieba 分词 → MySQL LIKE → 命中词数排序）
    # 已重构进 app/services/hybrid_service.py：分词保留（jieba.lcut_for_search + 正则整体保留
    # 型号/编号），排序升级为 BM25，并与 bge-m3 向量召回做 RRF 融合；RAG_HYBRID_ENABLED 控制
    # 是否启用（默认关闭），依赖/索引不可用时自动回退纯向量。
    def search(self, db: Session, query: str, limit: int | None = None,
               use_rerank: bool | None = None):
        """检索：宽召回（可选混合）→（可选）精排 → 阈值过滤 → 截断。

        过滤收在这一层而不是调用方，是为了先取宽候选再截断：
        原先 k=3 直接截断，排在第 4 位之后的正确片段永远进不了候选集。

        候选池：RAG_HYBRID_ENABLED=true 且索引可用时 = 向量召回 + BM25 关键词召回 RRF 融合；
        否则 = 纯 bge-m3 向量召回。精排开启时（RAG_RERANK_ENABLED=true）再对候选池成对重打分。
        任一可选环节的依赖/权重/索引缺失都会自动回退，不影响服务可用性。
        """
        if not query.strip():
            return []
        feed_topk = limit if limit is not None else settings.RAG_FEED_TOPK
        rerank_requested, rerank_available, _ = self.rerank_state(use_rerank)
        if rerank_requested:
            candidate_k = max(settings.RAG_RERANK_CANDIDATE_K, feed_topk)
        else:
            candidate_k = max(feed_topk, settings.RAG_CANDIDATE_K)

        chunks: list[RetrievedChunk] | None = None
        if settings.RAG_HYBRID_ENABLED:
            # 延迟导入：未启用时 hybrid_service / jieba 不参与启动
            from app.services import hybrid_service
            if hybrid_service.is_available():
                try:
                    chunks = self._hybrid_candidates(db, query, candidate_k)
                except Exception as exc:  # pragma: no cover - 运行时保护
                    logger.warning("RAG 混合检索失败，回退纯向量检索：%s", exc)
        if chunks is None:
            try:
                chunks = self._dense_chunks(db, query, candidate_k)
            except Exception as exc:  # pragma: no cover - 运行时保护
                # embedding 服务不可用（云端余额不足 / 网络不通 / 本地 Ollama 没起）时
                # 不要让整个对话请求 500：降级为「本轮没有知识库内容」，让模型用通用知识回答，
                # 并把真实原因留在日志里，避免静默失败。
                logger.warning("RAG 向量检索失败，本轮按空上下文处理：%s", exc)
                return []

        if rerank_requested:
            # 延迟导入：没装 torch/transformers 时不影响服务启动
            from app.services import rerank_service
            if rerank_available:
                try:
                    reranked = rerank_service.rerank_chunks(query, chunks)
                    passed = [
                        c for c in reranked
                        if c.score >= settings.RAG_RERANK_SCORE_THRESHOLD
                    ]
                    return passed[:feed_topk]
                except Exception as exc:  # pragma: no cover - 运行时保护
                    logger.warning("RAG rerank 调用失败，回退 cosine 检索：%s", exc)
            # is_available() 为 False 时加载失败原因已在 rerank_service 里打过 warning

        # 纯 cosine 路径（默认）：混合检索只扩大候选池，排序/阈值仍按 cosine
        passed = [c for c in chunks if c.score >= settings.RAG_SCORE_THRESHOLD]
        passed.sort(key=lambda c: c.score, reverse=True)
        return passed[:feed_topk]

    def rerank_state(self, use_rerank: bool | None = None) -> tuple[bool, bool, str]:
        """返回（是否请求精排, 精排是否可用, 实际打分方式）。

        use_rerank=None 跟随服务端配置；True/False 是请求级强制开关（Chat Studio 的
        精排按钮）。请求强制开启但服务端缺少 torch/权重时 available=False，
        打分方式回退 cosine，调用方可据此在前端提示「精排不可用」。
        """
        requested = (
            bool(settings.RAG_RERANK_ENABLED) if use_rerank is None else bool(use_rerank)
        )
        available = False
        if requested:
            from app.services import rerank_service
            available = rerank_service.is_available(force=(use_rerank is True))
        return requested, available, ("rerank" if available else "cosine")

    def active_score_config(self, use_rerank: bool | None = None) -> tuple[float, float, str]:
        """search() 之后调用：返回当前生效的（阈值, 弱命中余量, 打分方式）。

        精排模型可用时阈值按精排 sigmoid 分数标定（docs/05），否则按 cosine（docs/04）。
        混合检索不改变打分尺度，只扩大候选池，因此打分方式仍归入 cosine 一类。
        """
        _requested, _available, score_mode = self.rerank_state(use_rerank)
        if score_mode == "rerank":
            return (
                settings.RAG_RERANK_SCORE_THRESHOLD,
                settings.RAG_RERANK_WEAK_MARGIN,
                "rerank",
            )
        return settings.RAG_SCORE_THRESHOLD, settings.RAG_WEAK_MARGIN, "cosine"

    # ---------------------------------------------------------------- 候选生成
    def _dense_chunks(self, db: Session, query: str, k: int) -> list[RetrievedChunk]:
        """纯向量召回：宽候选 → distance 转 cosine 分数（不截断、不过阈值）。"""
        results = vectorstore.similarity_search_with_score(query, k=k)
        chunks: list[RetrievedChunk] = []
        for r, score in results:
            doc = self._convert_to_knowledge_doc(db, r)
            if doc is None:
                # 命中但反查 MySQL 失败 = 孤儿向量（文档已删但向量没删干净）。
                # 不静默跳过，记 warning，否则和"真·未命中"在日志里长得一样、没法排查。
                logger.warning(
                    "RAG 命中片段在 MySQL 中找不到对应文档，已跳过（疑似孤儿向量，"
                    "建议清理 Chroma 里对应 id 的向量）：metadata=%s",
                    dict(r.metadata or {}),
                )
                continue
            chunks.append(RetrievedChunk(
                title=doc.title, content=r.page_content, source=doc.source,
                score=float(1.0 - score),  # distance → cosine：越大越相关，方便直接跟阈值比
                doc_id=r.metadata.get("id"), chunk=r.metadata.get("chunk"),
            ))
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    def _rrf_fuse(self, dense: list[RetrievedChunk], sparse, rrf_k: int) -> list[dict]:
        """把向量候选与 BM25 候选按 RRF 融合；同一片段两路都命中则分数相加。"""
        from app.services import hybrid_service

        rows: dict[tuple, dict] = {}

        def _key(doc_id, chunk, fallback):
            if doc_id is None:
                return ("fallback", fallback)
            return (doc_id, chunk)

        for rank, chunk in enumerate(dense, start=1):
            row = rows.setdefault(
                _key(chunk.doc_id, chunk.chunk, chunk.content),
                {"chunk": None, "hybrid_doc": None, "rrf": 0.0},
            )
            if row["chunk"] is None:
                row["chunk"] = chunk
            row["rrf"] += hybrid_service.rrf_score(rank, rrf_k)

        for rank, (hybrid_doc, _bm25_score) in enumerate(sparse, start=1):
            row = rows.setdefault(
                _key(hybrid_doc.doc_id, hybrid_doc.chunk, hybrid_doc.vec_id),
                {"chunk": None, "hybrid_doc": None, "rrf": 0.0},
            )
            if row["hybrid_doc"] is None:
                row["hybrid_doc"] = hybrid_doc
            row["rrf"] += hybrid_service.rrf_score(rank, rrf_k)

        return sorted(rows.values(), key=lambda row: row["rrf"], reverse=True)

    def _hybrid_candidates(self, db: Session, query: str, candidate_k: int) -> list[RetrievedChunk]:
        """混合检索：向量 + BM25 两路召回 → RRF 融合 → 补齐 BM25 独有候选的 metadata/cosine。"""
        from app.services import hybrid_service

        dense = self._dense_chunks(db, query, settings.RAG_HYBRID_DENSE_K)
        sparse = hybrid_service.bm25_search(query, settings.RAG_HYBRID_SPARSE_K)
        fused = self._rrf_fuse(dense, sparse, settings.RAG_HYBRID_RRF_K)

        doc_cache: dict[int, KnowledgeDocument | None] = {}
        query_vector: list[float] | None = None
        chunks: list[RetrievedChunk] = []
        for row in fused:
            if len(chunks) >= candidate_k:
                break
            chunk = row["chunk"]
            if chunk is not None:
                chunks.append(chunk)
                continue
            hybrid_doc = row["hybrid_doc"]
            assert hybrid_doc is not None
            doc = None
            if hybrid_doc.doc_id is not None:
                if hybrid_doc.doc_id not in doc_cache:
                    doc_cache[hybrid_doc.doc_id] = (
                        db.query(KnowledgeDocument)
                        .filter(KnowledgeDocument.id == hybrid_doc.doc_id)
                        .first()
                    )
                doc = doc_cache[hybrid_doc.doc_id]
                if doc is None:
                    logger.warning(
                        "RAG 混合检索命中片段在 MySQL 中找不到对应文档，已跳过（疑似孤儿向量）："
                        "metadata=%s",
                        {"id": hybrid_doc.doc_id, "title": hybrid_doc.title, "chunk": hybrid_doc.chunk},
                    )
                    continue
            score = 0.0
            if hybrid_doc.embedding is not None:
                if query_vector is None:
                    try:
                        query_vector = hybrid_service.embed_query(query)
                    except Exception as exc:  # pragma: no cover - 运行时保护
                        logger.warning("RAG 混合检索补 cosine 分数失败：%s", exc)
                        query_vector = []
                score = hybrid_service.cosine(query_vector, hybrid_doc.embedding)
            chunks.append(RetrievedChunk(
                title=doc.title if doc is not None else hybrid_doc.title,
                content=hybrid_doc.content,
                source=doc.source if doc is not None else None,
                score=float(score),
                doc_id=hybrid_doc.doc_id,
                chunk=hybrid_doc.chunk,
            ))
        return chunks

    def build_context(self, docs):
        return "\n\n".join(
            f"[知识片段{i}] 标题：{doc.title}\n内容：{doc.content}"
            for i, doc in enumerate(docs, start=1)
        )

    def update_document(self, db: Session, knowledge_id: int, title: str | None = None, content: str | None = None,
                        source: str | None = None):
        """根据 ID 更新知识库文档"""
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == knowledge_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"知识库文档不存在，ID: {knowledge_id}")

        if title is not None:
            doc.title = title
        if content is not None:
            doc.content = content
        if source is not None:
            doc.source = source

        db.commit()
        db.refresh(doc)
        # 一篇文档对应多个 chunk 向量，按 metadata 里的文档 id 删除它的全部片段
        vectorstore._collection.delete(where={"id": knowledge_id})
        self._sync_to_chroma(doc)
        return doc

    def delete_document(self, db, knowledge_id):
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == knowledge_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"知识库文档不存在，ID: {knowledge_id}")
        db.delete(doc)
        db.commit()
        vectorstore._collection.delete(where={"id": knowledge_id})
        _invalidate_hybrid_index()
        return {"message": "知识库文档删除成功"}

    def _convert_to_knowledge_doc(self, db: Session, r):
        """
        从 r.metadata["id"] 拿到 MySQL 主键
        用这个 id 去 db 里查 KnowledgeDocument
        查到就返回，查不到返回 None
        """
        doc_id = r.metadata.get("id")
        if doc_id is None:
            return None
        return db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()

    def _sync_to_chroma(self, doc):
        """把一条知识库文档切分成片段后写入 Chroma，每个片段一个向量；片段 id 用 '文档id_序号'"""
        # 只向量化正文，不拼标题前缀：标题已通过 metadata 反查 + build_context 单独展示。
        # 之前 f"{doc.title}：{doc.content}" 会让每条向量共享同一段标题开头，
        # 结构性抬高 cosine 基线（"查询全挤在 0.36–0.44" 的第三个原因）。
        text = doc.content
        chunks = text_splitter.split_text(text)
        ids = [f"{doc.id}_{i}" for i in range(len(chunks))]
        # metadata 里的 id 存 MySQL 主键，检索时用它反查文档；chunk 记录片段序号
        metadatas = [{"id": doc.id, "title": doc.title, "chunk": i} for i in range(len(chunks))]
        # 必须走 LangChain 包装层的 add_texts，它才会用配置的 bge-m3(1024维) 生成向量；
        # 若直接调 _collection.add(documents=...)，chromadb 会用自带的默认模型(384维)，导致读写维度不一致。
        vectorstore.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=ids,
        )
        _invalidate_hybrid_index()
        