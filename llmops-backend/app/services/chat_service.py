# 聊天核心业务流程，负责读取 Prompt、处理 RAG、调用模型、写入日志。
import logging
import time
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.chat_log import ChatLog
from app.models.conversation import Conversation
from app.schemas.chat import ChatRequest
from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService
from app.services import grounding, query_rewrite
from app.services.rag_service import RAGService
from app.services.text_clean import InputCleaner
from app.services.image_input import extract_image_text, merge_query, public_meta
from app.core.config import settings

logger = logging.getLogger(__name__)


def _canned_stream(answer: str, provider: str, model_name: str):
    """把固定话术包装成与 LLM 流式同构的分片，复用下游的 delta / done 处理。"""
    yield {"content": answer}
    yield {
        "done": True,
        "answer": answer,
        "reasoning": "知识库未检索到相关内容，直接返回固定话术。",
        "model_provider": provider,
        "model_name": model_name,
        "total_tokens": 0,
    }


# RAG 命中阈值（cosine 相似度，越大越相关）。
class ChatService:

    def __init__(self):
        self.llm = LLMService()
        self.prompt_service = PromptService()
        self.rag = RAGService()
    def new_conversation_id(self) -> str:
        return str(uuid.uuid4())
    ##会话记忆
    def _load_history(self, db: Session, conversation_id: str | None, limit: int) -> list[dict]:
        # 从 ChatLog 中读取同一会话最近若干轮成功对话，作为上下文记忆。
        if not conversation_id or limit <= 0:
            return []
        logs = (
            db.query(ChatLog)
            .filter(
                ChatLog.conversation_id == conversation_id,
                ChatLog.status == "success",
            )
            .order_by(ChatLog.created_at.desc(), ChatLog.id.desc())
            .limit(limit)
            .all()
        )
        logs.reverse()
        history: list[dict] = []
        for log in logs:
            history.append({"role": "user", "content": log.user_input})
            if log.assistant_output:
                history.append({"role": "assistant", "content": log.assistant_output})
        return history
    ##
    def _ensure_conversation(self, db: Session, conversation_id: str | None, query: str):
        # 确保 conversations 表中有该会话记录；首次对话时自动用问题前30字做标题。
        if not conversation_id:
            return
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            title = query[:30] if len(query) > 30 else query
            conv = Conversation(id=conversation_id, title=title)
            db.add(conv)
        elif conv.title == "新会话":
            conv.title = query[:30] if len(query) > 30 else query
        conv.updated_at = datetime.utcnow()
        db.commit()
    ##
    # ── 同步 / 流式共用的准备阶段 ──────────────────────────────
    @staticmethod
    def _resolve_model_name(provider: str, model_name: str | None) -> str:
        if model_name:
            return model_name
        if provider == "ollama":
            return settings.OLLAMA_MODEL
        if provider == "deepseek":
            return settings.DEEPSEEK_MODEL
        return "unknown"

    def _prepare(self, db: Session, req: ChatRequest) -> dict:
        """解析 Prompt / 模型参数 / 会话记忆，得到本轮上下文（RAG 单独处理）。"""
        prompt_name = None
        system_prompt = req.system_prompt or "你是工业 LLMOps 平台中的智能助手，请准确、简洁地回答。"
        provider = req.model_provider or "ollama"
        model_name = req.model_name
        temperature = req.temperature
        top_p = req.top_p

        # 输入清洗：脏文本会把 bge-m3 向量带偏，型号类查询还会被全角字符搞失效。
        # 用户原文保留在 req.query（写库、展示都用它），清洗结果只用于检索与提问。
        clean_query, clean_meta = self._clean_query(req)

        # 图片输入：OCR 提取文字后与清洗后的提问合并，检索与提问都用合并结果
        image_meta = extract_image_text(getattr(req, "image_base64", None))
        final_query = clean_query
        if image_meta.get("text"):
            final_query = merge_query(image_meta["text"], clean_query)

        workflow_reasoning = [
            f"用户问题：{req.query}",
            f"模型提供商：{provider}",
            f"模型名称：{model_name or '默认模型'}",
            "步骤1：读取用户输入。",
            self._clean_note(clean_meta),
            (
                f"步骤2b：识别用户图片，提取 {image_meta['chars']} 字"
                f"（{image_meta['engine']}，{image_meta['ms']} ms），已与提问合并。"
                if image_meta.get("applied")
                else "步骤2b：本轮无图片输入。"
            ),
            "步骤3：加载系统提示词。",
        ]
        if req.prompt_id:
            prompt = self.prompt_service.get(db, req.prompt_id)
            if prompt:
                prompt_name = prompt.name
                system_prompt = prompt.system_prompt
                provider = req.model_provider or prompt.model_provider or provider
                model_name = req.model_name or prompt.model_name
                temperature = prompt.temperature
                top_p = prompt.top_p
                workflow_reasoning.append(f"步骤4：使用 Prompt 模板：{prompt_name}。")
            else:
                workflow_reasoning.append("步骤4：未找到指定 Prompt，使用默认系统提示词。")
        else:
            workflow_reasoning.append("步骤4：未选择 Prompt，使用默认系统提示词。")

        history: list[dict] = []
        if req.memory_enabled and req.conversation_id:
            history = self._load_history(db, req.conversation_id, settings.MEMORY_MAX_TURNS)
            workflow_reasoning.append(
                f"步骤5：加载会话记忆，conversation_id={req.conversation_id}，"
                f"共 {len(history) // 2} 轮历史对话。"
            )
        else:
            workflow_reasoning.append("步骤5：未启用会话记忆或无 conversation_id，本轮为独立对话。")

        return {
            "system_prompt": system_prompt,
            "provider": provider,
            "model_name": self._resolve_model_name(provider, model_name),
            "temperature": temperature,
            "top_p": top_p,
            "prompt_name": prompt_name,
            "workflow_reasoning": workflow_reasoning,
            "history": history,
            "clean_query": clean_query,
            "clean_meta": clean_meta,
            "image_meta": image_meta,
            "final_query": final_query,
            "rag_docs": [],
            "rag_status": "off",
            "score_mode": "off",
            "rerank_requested": False,
            "rerank_available": False,
            "timings": {},
        }

    def _clean_query(self, req: ChatRequest) -> tuple[str, dict]:
        """按请求开关与全局配置决定本轮走哪条清洗通道，返回 (清洗后文本, 元信息)。

        两级开关，语义互不越权：
        - ``use_clean``（整体开关）：False 一律不清洗；True 时即使服务端默认 off 也按规则通道开。
        - ``use_clean_model``（通道开关）：True 时把本轮升级到模型通道（精细清洗）。
          服务端 ``INPUT_CLEAN_MODEL_ALLOWED=false``、或本轮已被 ``use_clean=false`` 关掉时
          不升级——不报错，只把 ``fallback="rule"`` 与 ``fallback_reason="not_allowed"`` 记进
          元信息，让前端能区分「服务端不允许」与「模型服务挂了」。
          注意它只升级通道、不负责开启清洗：服务端默认 off 时仍需 ``use_clean=true``。

        每次请求都重新读 settings 再构造：清洗是纯策略对象（无 I/O、无缓存），
        这样改 .env 或测试里 monkeypatch 都能立刻生效，不用重启。
        """
        mode = (settings.INPUT_CLEAN_MODE or "off").lower()
        if req.use_clean is False:
            mode = "off"
        elif req.use_clean is True and mode == "off":
            # 服务端没配默认策略时，前端强制开启按规则清洗（模型通道需要额外进程，不隐式启用）
            mode = "rule"

        # 请求级通道升级：只有清洗整体开着、且用户显式点「精细清洗」才走模型
        want_model = bool(req.use_clean_model) and mode != "off"
        not_allowed = want_model and not settings.INPUT_CLEAN_MODEL_ALLOWED
        if want_model and not not_allowed:
            mode = "model"

        cleaned, meta = InputCleaner(mode=mode).preprocess(req.query)
        meta["requested_model"] = want_model
        if not_allowed:
            meta["fallback"] = "rule"
            meta["fallback_reason"] = "not_allowed"
        return cleaned, meta

    @staticmethod
    def _clean_note(meta: dict) -> str:
        mode = meta.get("mode") or "off"
        if mode == "off" or not meta.get("applied"):
            reason = {
                "empty": "输入为空",
                "too_long": "输入过长，跳过清洗",
                "error": "清洗失败，已回退原文",
                "off": "未启用",
            }.get(meta.get("reason") or "", "未执行")
            return f"步骤2：输入清洗未生效（{reason}）。"
        note = "内容有改动" if meta.get("changed") else "内容本来就干净"
        fallback = ""
        if meta.get("fallback") == "rule":
            why = "服务端未允许模型通道" if meta.get("fallback_reason") == "not_allowed" else "模型通道失败"
            fallback = f"，{why}，已回退规则"
        return (f"步骤2：输入清洗完成（mode={mode}，{note}，"
                f"{meta.get('ms', 0)} ms{fallback}）。")

    @staticmethod
    def _clean_public(meta: dict, clean_query: str) -> dict:
        """给前端的清洗结果：只在内容真的变了时附带清洗后文本。"""
        public = {k: meta.get(k) for k in
                  ("mode", "applied", "changed", "fallback", "fallback_reason",
                   "reason", "ms", "requested_model")}
        if meta.get("changed"):
            public["clean_query"] = clean_query[:200]
        return public

    @staticmethod
    def _rewrite_public(meta: dict | None) -> dict | None:
        """给前端的改写信息：只暴露判定与结果，不暴露内部异常。"""
        if not meta or not meta.get("applied"):
            return None
        return {
            "mode": meta.get("mode"),
            "via": meta.get("via"),
            "changed": bool(meta.get("changed")),
            "query": meta.get("query") if meta.get("changed") else None,
            "fallback": meta.get("fallback"),
            "ms": meta.get("ms", 0),
        }

    def _apply_rag(self, db: Session, req: ChatRequest, ctx: dict) -> None:
        """执行 RAG 检索并改写 final_query（无命中时保持原问题）。"""
        workflow_reasoning = ctx["workflow_reasoning"]
        # answer_text：给模型看的提问（只用清洗后的用户文本，保持聚焦）
        # query：拿去检索的文本（含图片识别结果——型号、编号这类实体往往只在图里）
        answer_text = ctx.get("clean_query") or req.query
        query = ctx.get("final_query") or answer_text
        if not req.use_rag:
            workflow_reasoning.append("步骤6：未启用 RAG，直接调用模型。")
            return
        # 多轮：检索用自包含查询，生成仍用原文——改写只作用于检索侧
        search_query, rewrite_meta = query_rewrite.rewrite(query, ctx.get("history") or [])
        ctx["search_query"] = search_query
        ctx["rewrite_meta"] = rewrite_meta
        if rewrite_meta.get("changed"):
            workflow_reasoning.append(
                f"步骤6-1：多轮改写检索查询（{rewrite_meta.get('via') or rewrite_meta['mode']}，"
                f"{rewrite_meta['ms']} ms）→ {search_query}"
            )
        # search() 内部已完成「取宽候选 → 按阈值过滤 → 截断」，此处不再重复过滤
        # timing 只写不读：把检索/精排耗时直接记到本轮 ctx 里，供日志与响应展示
        rag_docs = self.rag.search(
            db, search_query, use_rerank=req.use_rerank, timing=ctx.get("timings"),
        )
        if not rag_docs and search_query.strip() != query.strip():
            # 改写没召回到东西：用原句兜底再检一次，避免"改坏"比"不改"更差
            rag_docs = self.rag.search(
                db, query, use_rerank=req.use_rerank, timing=ctx.get("timings"),
            )
            rewrite_meta["fallback"] = "origin"
            workflow_reasoning.append("步骤6-2：改写未召回，已用原句兜底重检。")
        ctx["rag_docs"] = rag_docs
        rerank_requested, rerank_available, score_mode = self.rag.rerank_state(req.use_rerank)
        ctx["score_mode"] = score_mode
        ctx["rerank_requested"] = rerank_requested
        ctx["rerank_available"] = rerank_available
        context = self.rag.build_context(rag_docs)
        if context:
            top1 = rag_docs[0].score
            # 阈值按当前打分方式取：精排 sigmoid 分数 / cosine 分数（见 rag_service）
            threshold, weak_margin, score_mode = self.rag.active_score_config(req.use_rerank)
            # 型号/编号逐字命中（match=exact）时，分数低但确定性高，直接判 hit；
            # 其余仍按「阈值 + 弱命中余量」判定，避免把勉强过线的片段当可靠依据。
            exact_hit = getattr(rag_docs[0], "match", "vector") == "exact"
            # 闸门勉强过线时标成 weak_hit，不让前端把它当成可靠依据
            ctx["rag_status"] = "hit" if exact_hit or top1 >= threshold + weak_margin else "weak_hit"
            ctx["final_query"] = self._grounded_prompt(context, answer_text)
            rerank_note = (
                "（精排请求但服务端不可用，已回退向量）"
                if rerank_requested and not rerank_available else ""
            )
            workflow_reasoning.append(
                f"步骤6：启用 RAG，检索到 {len(rag_docs)} 条知识片段"
                f"（打分={score_mode}，top1={top1:.4f}，判定 {ctx['rag_status']}）{rerank_note}。"
            )
        else:
            ctx["rag_status"] = "no_hit"
            if settings.RAG_NO_HIT_REPLY_ENABLED:
                # 没有依据就不让模型自由发挥：固定话术，顺带省掉一次生成
                ctx["canned_answer"] = settings.RAG_NO_HIT_REPLY_TEXT
                workflow_reasoning.append(
                    "步骤6：启用 RAG，但未检索到有效知识片段（返回固定话术，不调用模型）。"
                )
            else:
                workflow_reasoning.append(
                    "步骤6：启用 RAG，但未检索到有效知识片段（本次回答不含知识库内容）。"
                )

    @staticmethod
    def _guard_numbers(answer: str, ctx: dict) -> tuple[str, dict | None]:
        """数字级 grounding：回答里出现、片段里找不到的数字，追加提示（docs/19）。"""
        if not settings.RAG_NUMERIC_GUARD_ENABLED or not ctx.get("rag_docs"):
            return answer, None
        context = "\n".join(f"{d.title}\n{d.content}" for d in ctx["rag_docs"])
        result = grounding.check_numbers(answer, context)
        if result["ok"]:
            return answer, result
        warn = settings.RAG_NUMERIC_GUARD_TEXT.format(nums="、".join(result["missing"]))
        return f"{answer}\n\n{warn}", result

    @staticmethod
    def _grounded_prompt(context: str, query: str) -> str:
        """命中的提示词。RAG_STRICT_GROUNDING=true 时禁止模型用常识补充。"""
        if not settings.RAG_STRICT_GROUNDING:
            return (
                "请基于以下知识库内容回答问题。\n"
                "如果知识库内容不足，请说明不足，再结合通用知识补充。\n\n"
                f"{context}\n\n用户问题：{query}"
            )
        return (
            "请只依据下面的知识库片段回答问题，数字与结论必须来自片段原文。\n"
            "片段里没有的信息，直接回答“知识库未收录该信息”，不要用常识补充，不要猜测或编造。\n"
            "如果引用了片段，在回答末尾标注编号，例如【依据：知识片段1】。\n\n"
            f"{context}\n\n用户问题：{query}"
        )

    @staticmethod
    def _source_items(rag_docs: list) -> list[dict]:
        return [
            {
                "title": d.title,
                "snippet": d.content[:200],
                "source": d.source,
                "score": d.score,
                # vector=向量相似度；exact=型号/编号逐字命中（分数低但确定性高）
                "match": getattr(d, "match", "vector"),
            }
            for d in rag_docs
        ]

    @staticmethod
    def _thought_text(provider: str) -> str:
        if provider == "ollama":
            return "已调用本地 Ollama 模型 deepseek-r1:7b，并完成 Prompt/RAG/会话记忆/日志流程。"
        return "已调用 DeepSeek 云端 API 模式，并完成 Prompt/RAG/会话记忆/日志流程。"

    def _save_success_log(self, db: Session, req: ChatRequest, ctx: dict, answer: str,
                          total_tokens: int, latency_ms: int, timings: dict | None = None) -> None:
        db.add(
            ChatLog(
                conversation_id=req.conversation_id,
                user_input=req.query,
                assistant_output=answer,
                model_provider=ctx["provider"],
                model_name=ctx["model_name"],
                prompt_name=ctx["prompt_name"],
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                timings=timings or None,
                status="success",
            )
        )
        db.commit()
        self._ensure_conversation(db, req.conversation_id, req.query)

    def _save_failed_log(self, db: Session, req: ChatRequest, ctx: dict | None,
                         partial_answer: str, latency_ms: int, error: Exception,
                         timings: dict | None = None) -> None:
        db.add(
            ChatLog(
                conversation_id=req.conversation_id,
                user_input=req.query,
                assistant_output=partial_answer or "",
                model_provider=(ctx or {}).get("provider") or req.model_provider or "unknown",
                model_name=(ctx or {}).get("model_name") or req.model_name or "unknown",
                prompt_name=(ctx or {}).get("prompt_name"),
                total_tokens=0,
                latency_ms=latency_ms,
                timings=timings or None,
                status="failed",
                error_message=str(error),
            )
        )
        db.commit()

    def chat(self, db: Session, req: ChatRequest):
        """非流式对话（保留给旧前端 / 流式失败时的降级路径）。"""
        start = time.time()
        timings: dict = {}
        t_prepare = time.perf_counter()
        ctx = self._prepare(db, req)
        timings["prepare_ms"] = int((time.perf_counter() - t_prepare) * 1000)
        timings["clean_ms"] = (ctx.get("clean_meta") or {}).get("ms", 0)
        ctx["timings"] = timings
        self._apply_rag(db, req, ctx)
        try:
            if ctx.get("canned_answer"):
                ctx["workflow_reasoning"].append("步骤7：知识库无命中，直接返回固定话术（未调用模型）。")
                result = {
                    "answer": ctx["canned_answer"],
                    "reasoning": "知识库未检索到相关内容，直接返回固定话术。",
                    "model_provider": ctx["provider"],
                    "model_name": ctx["model_name"],
                    "total_tokens": 0,
                }
                timings["llm_ms"] = 0
            else:
                ctx["workflow_reasoning"].append("步骤7：调用大模型生成回答。")
                t_llm = time.perf_counter()
                result = self.llm.chat(
                    query=ctx["final_query"],
                    system_prompt=ctx["system_prompt"],
                    provider=ctx["provider"],
                    model_name=ctx["model_name"],
                    temperature=ctx["temperature"],
                    top_p=ctx["top_p"],
                    history=ctx["history"],
                )
                timings["llm_ms"] = int((time.perf_counter() - t_llm) * 1000)
            latency_ms = int((time.time() - start) * 1000)
            timings["total_ms"] = latency_ms
            answer = result["answer"]
            guarded, guard = self._guard_numbers(answer, ctx)
            self._save_success_log(
                db, req, ctx, guarded, result["total_tokens"], latency_ms, timings,
            )
            ctx["workflow_reasoning"].append("步骤8：模型回答生成完成。")
            ctx["workflow_reasoning"].append("步骤9：将调用日志写入数据库。")
            ctx["workflow_reasoning"].append(f"步骤10：本次调用耗时 {latency_ms} ms。")
            reasoning = result.get("reasoning") or "\n".join(ctx["workflow_reasoning"])
            return {
                "answer": guarded,
                "reasoning": reasoning,
                "thought": self._thought_text(result["model_provider"]),
                "model_provider": result["model_provider"],
                "model_name": result["model_name"],
                "total_tokens": result["total_tokens"],
                "latency_ms": latency_ms,
                "conversation_id": req.conversation_id,
                "rag_status": ctx["rag_status"],
                "sources": self._source_items(ctx["rag_docs"]),
                "score_mode": ctx.get("score_mode"),
                "rerank_requested": ctx.get("rerank_requested"),
                "rerank_available": ctx.get("rerank_available"),
                "clean": self._clean_public(ctx.get("clean_meta") or {}, ctx.get("clean_query") or ""),
                "image": public_meta(ctx.get("image_meta")),
                "rewrite": self._rewrite_public(ctx.get("rewrite_meta")),
                "grounding": guard,
                "timings": timings,
            }
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            timings["total_ms"] = latency_ms
            self._save_failed_log(db, req, ctx, "", latency_ms, e, timings)
            raise

    def stream(self, db: Session, req: ChatRequest):
        """流式对话：产出事件字典，由 API 层转成 SSE。

        事件顺序：status(preparing) → status(retrieving) → meta → status(generating)
        → reasoning/delta* → done；异常时产出 error（并写 failed 日志）。
        """
        start = time.time()
        timings: dict = {}
        ctx: dict | None = None
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        first_token_ms: int | None = None
        try:
            yield {"event": "status", "data": {"phase": "preparing"}}
            t_prepare = time.perf_counter()
            ctx = self._prepare(db, req)
            timings["prepare_ms"] = int((time.perf_counter() - t_prepare) * 1000)
            timings["clean_ms"] = (ctx.get("clean_meta") or {}).get("ms", 0)
            ctx["timings"] = timings
            if req.use_rag:
                yield {"event": "status", "data": {"phase": "retrieving"}}
            self._apply_rag(db, req, ctx)
            yield {
                "event": "meta",
                "data": {
                    "conversation_id": req.conversation_id,
                    "model_provider": ctx["provider"],
                    "model_name": ctx["model_name"],
                    "prompt_name": ctx["prompt_name"],
                    "rag_status": ctx["rag_status"],
                    "sources": self._source_items(ctx["rag_docs"]),
                    "score_mode": ctx.get("score_mode"),
                    "rerank_requested": ctx.get("rerank_requested"),
                    "rerank_available": ctx.get("rerank_available"),
                    "clean": self._clean_public(
                        ctx.get("clean_meta") or {}, ctx.get("clean_query") or ""
                    ),
                    "image": public_meta(ctx.get("image_meta")),
                    "rewrite": self._rewrite_public(ctx.get("rewrite_meta")),
                },
            }
            ctx["workflow_reasoning"].append(
                "步骤7：知识库无命中，直接返回固定话术（未调用模型）。"
                if ctx.get("canned_answer")
                else "步骤7：调用大模型生成回答（流式）。"
            )
            yield {"event": "status", "data": {"phase": "generating"}}

            t_llm = time.perf_counter()
            if ctx.get("canned_answer"):
                model_stream = _canned_stream(
                    ctx["canned_answer"], ctx["provider"], ctx["model_name"]
                )
            else:
                model_stream = self.llm.chat_stream(
                    query=ctx["final_query"],
                    system_prompt=ctx["system_prompt"],
                    provider=ctx["provider"],
                    model_name=ctx["model_name"],
                    temperature=ctx["temperature"],
                    top_p=ctx["top_p"],
                    history=ctx["history"],
                )
            result: dict | None = None
            try:
                for chunk in model_stream:
                    # done 分片里也带 answer/reasoning 汇总，不能当增量再发一遍
                    if chunk.get("done"):
                        result = chunk
                        continue
                    reasoning = chunk.get("reasoning") or ""
                    if reasoning:
                        if first_token_ms is None:
                            first_token_ms = int((time.time() - start) * 1000)
                        reasoning_parts.append(reasoning)
                        yield {"event": "reasoning", "data": {"delta": reasoning}}
                    content = chunk.get("content") or ""
                    if content:
                        if first_token_ms is None:
                            first_token_ms = int((time.time() - start) * 1000)
                        answer_parts.append(content)
                        yield {"event": "delta", "data": {"content": content}}
            finally:
                close = getattr(model_stream, "close", None)
                if close is not None:
                    close()

            result = result or {}
            answer = result.get("answer") or "".join(answer_parts)
            # 数字级 grounding：编出来的数值在这里补一条提示（流式已吐出的内容无法撤回，只能追加）
            guarded, guard = self._guard_numbers(answer, ctx)
            if guarded != answer:
                warn = guarded[len(answer):]
                yield {"event": "delta", "data": {"content": warn}}
                answer = guarded
            reasoning = (
                result.get("reasoning")
                or "".join(reasoning_parts)
                or "\n".join(ctx["workflow_reasoning"])
            )
            total_tokens = int(result.get("total_tokens") or 0)
            latency_ms = int((time.time() - start) * 1000)
            timings["llm_ms"] = int((time.perf_counter() - t_llm) * 1000)
            timings["first_token_ms"] = first_token_ms
            timings["total_ms"] = latency_ms
            self._save_success_log(db, req, ctx, answer, total_tokens, latency_ms, timings)
            provider = result.get("model_provider") or ctx["provider"]
            done = {
                "answer": answer,
                "reasoning": reasoning,
                "thought": self._thought_text(provider),
                "model_provider": provider,
                "model_name": result.get("model_name") or ctx["model_name"],
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
                "first_token_ms": first_token_ms,
                "conversation_id": req.conversation_id,
                "prompt_name": ctx["prompt_name"],
                "rag_status": ctx["rag_status"],
                "sources": self._source_items(ctx["rag_docs"]),
                "score_mode": ctx.get("score_mode"),
                "rerank_requested": ctx.get("rerank_requested"),
                "rerank_available": ctx.get("rerank_available"),
                "clean": self._clean_public(ctx.get("clean_meta") or {}, ctx.get("clean_query") or ""),
                "image": public_meta(ctx.get("image_meta")),
                "rewrite": self._rewrite_public(ctx.get("rewrite_meta")),
                "grounding": guard,
                "timings": timings,
            }
            logger.info(
                "chat.stream done provider=%s model=%s ttft_ms=%s latency_ms=%s tokens=%s timings=%s",
                done["model_provider"], done["model_name"], first_token_ms, latency_ms, total_tokens, timings,
            )
            yield {"event": "done", "data": done}
        except Exception as exc:
            logger.exception("chat.stream failed")
            latency_ms = int((time.time() - start) * 1000)
            timings["total_ms"] = latency_ms
            self._save_failed_log(db, req, ctx, "".join(answer_parts), latency_ms, exc, timings)
            yield {"event": "error", "data": {"message": str(exc), "latency_ms": latency_ms}}
