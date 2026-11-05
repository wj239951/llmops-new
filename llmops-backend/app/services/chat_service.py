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
from app.services.rag_service import RAGService
from app.core.config import settings

logger = logging.getLogger(__name__)

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

        workflow_reasoning = [
            f"用户问题：{req.query}",
            f"模型提供商：{provider}",
            f"模型名称：{model_name or '默认模型'}",
            "步骤1：读取用户输入。",
            "步骤2：加载系统提示词。",
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
                workflow_reasoning.append(f"步骤3：使用 Prompt 模板：{prompt_name}。")
            else:
                workflow_reasoning.append("步骤3：未找到指定 Prompt，使用默认系统提示词。")
        else:
            workflow_reasoning.append("步骤3：未选择 Prompt，使用默认系统提示词。")

        history: list[dict] = []
        if req.memory_enabled and req.conversation_id:
            history = self._load_history(db, req.conversation_id, settings.MEMORY_MAX_TURNS)
            workflow_reasoning.append(
                f"步骤4：加载会话记忆，conversation_id={req.conversation_id}，"
                f"共 {len(history) // 2} 轮历史对话。"
            )
        else:
            workflow_reasoning.append("步骤4：未启用会话记忆或无 conversation_id，本轮为独立对话。")

        return {
            "system_prompt": system_prompt,
            "provider": provider,
            "model_name": self._resolve_model_name(provider, model_name),
            "temperature": temperature,
            "top_p": top_p,
            "prompt_name": prompt_name,
            "workflow_reasoning": workflow_reasoning,
            "history": history,
            "final_query": req.query,
            "rag_docs": [],
            "rag_status": "off",
            "score_mode": "off",
            "rerank_requested": False,
            "rerank_available": False,
        }

    def _apply_rag(self, db: Session, req: ChatRequest, ctx: dict) -> None:
        """执行 RAG 检索并改写 final_query（无命中时保持原问题）。"""
        workflow_reasoning = ctx["workflow_reasoning"]
        if not req.use_rag:
            workflow_reasoning.append("步骤5：未启用 RAG，直接调用模型。")
            return
        # search() 内部已完成「取宽候选 → 按阈值过滤 → 截断」，此处不再重复过滤
        rag_docs = self.rag.search(db, req.query, use_rerank=req.use_rerank)
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
            ctx["final_query"] = (
                "请基于以下知识库内容回答问题。\n"
                "如果知识库内容不足，请说明不足，再结合通用知识补充。\n\n"
                f"{context}\n\n用户问题：{req.query}"
            )
            rerank_note = (
                "（精排请求但服务端不可用，已回退向量）"
                if rerank_requested and not rerank_available else ""
            )
            workflow_reasoning.append(
                f"步骤5：启用 RAG，检索到 {len(rag_docs)} 条知识片段"
                f"（打分={score_mode}，top1={top1:.4f}，判定 {ctx['rag_status']}）{rerank_note}。"
            )
        else:
            ctx["rag_status"] = "no_hit"
            workflow_reasoning.append("步骤5：启用 RAG，但未检索到有效知识片段（本次回答不含知识库内容）。")

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
                          total_tokens: int, latency_ms: int) -> None:
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
                status="success",
            )
        )
        db.commit()
        self._ensure_conversation(db, req.conversation_id, req.query)

    def _save_failed_log(self, db: Session, req: ChatRequest, ctx: dict | None,
                         partial_answer: str, latency_ms: int, error: Exception) -> None:
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
                status="failed",
                error_message=str(error),
            )
        )
        db.commit()

    def chat(self, db: Session, req: ChatRequest):
        """非流式对话（保留给旧前端 / 流式失败时的降级路径）。"""
        start = time.time()
        ctx = self._prepare(db, req)
        self._apply_rag(db, req, ctx)
        try:
            ctx["workflow_reasoning"].append("步骤6：调用大模型生成回答。")
            result = self.llm.chat(
                query=ctx["final_query"],
                system_prompt=ctx["system_prompt"],
                provider=ctx["provider"],
                model_name=ctx["model_name"],
                temperature=ctx["temperature"],
                top_p=ctx["top_p"],
                history=ctx["history"],
            )
            latency_ms = int((time.time() - start) * 1000)
            self._save_success_log(db, req, ctx, result["answer"], result["total_tokens"], latency_ms)
            ctx["workflow_reasoning"].append("步骤7：模型回答生成完成。")
            ctx["workflow_reasoning"].append("步骤8：将调用日志写入数据库。")
            ctx["workflow_reasoning"].append(f"步骤9：本次调用耗时 {latency_ms} ms。")
            reasoning = result.get("reasoning") or "\n".join(ctx["workflow_reasoning"])
            return {
                "answer": result["answer"],
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
            }
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            self._save_failed_log(db, req, ctx, "", latency_ms, e)
            raise

    def stream(self, db: Session, req: ChatRequest):
        """流式对话：产出事件字典，由 API 层转成 SSE。

        事件顺序：status(preparing) → status(retrieving) → meta → status(generating)
        → reasoning/delta* → done；异常时产出 error（并写 failed 日志）。
        """
        start = time.time()
        ctx: dict | None = None
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        first_token_ms: int | None = None
        try:
            yield {"event": "status", "data": {"phase": "preparing"}}
            ctx = self._prepare(db, req)
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
                },
            }
            ctx["workflow_reasoning"].append("步骤6：调用大模型生成回答（流式）。")
            yield {"event": "status", "data": {"phase": "generating"}}

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
            reasoning = (
                result.get("reasoning")
                or "".join(reasoning_parts)
                or "\n".join(ctx["workflow_reasoning"])
            )
            total_tokens = int(result.get("total_tokens") or 0)
            latency_ms = int((time.time() - start) * 1000)
            self._save_success_log(db, req, ctx, answer, total_tokens, latency_ms)
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
            }
            logger.info(
                "chat.stream done provider=%s model=%s ttft_ms=%s latency_ms=%s tokens=%s",
                done["model_provider"], done["model_name"], first_token_ms, latency_ms, total_tokens,
            )
            yield {"event": "done", "data": done}
        except Exception as exc:
            logger.exception("chat.stream failed")
            latency_ms = int((time.time() - start) * 1000)
            self._save_failed_log(db, req, ctx, "".join(answer_parts), latency_ms, exc)
            yield {"event": "error", "data": {"message": str(exc), "latency_ms": latency_ms}}
