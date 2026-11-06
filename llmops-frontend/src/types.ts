// 与后端 Pydantic schema 对齐的前端类型定义。
// 后端序列化后日期字段是 ISO 字符串，这里统一用 string。

/** Prompt 模板（对应 PromptOut） */
export interface Prompt {
  id: number
  name: string
  description: string | null
  system_prompt: string
  model_provider: string
  model_name: string | null
  temperature: number
  top_p: number
  enabled: boolean
  created_at: string
  updated_at: string
}

/** 新建/修改 Prompt 的入参（对应 PromptCreate / PromptUpdate） */
export interface PromptPayload {
  name?: string
  description?: string | null
  system_prompt?: string
  model_provider?: string
  model_name?: string | null
  temperature?: number
  top_p?: number
  enabled?: boolean
}

/** 知识库文档（对应 KnowledgeOut） */
export interface KnowledgeDocument {
  id: number
  title: string
  content: string
  source: string | null
  created_at: string
}

/** 知识库分页响应（对应 KnowledgePage） */
export interface KnowledgePage {
  items: KnowledgeDocument[]
  total: number
  page: number
  page_size: number
}

/** 新建/修改知识库文档的入参（对应 KnowledgeCreate / KnowledgeUpdate） */
export interface KnowledgePayload {
  title?: string
  content?: string
  source?: string | null
}

/** 聊天请求（对应 ChatRequest） */
export interface ChatRequestPayload {
  query: string
  prompt_id?: number | null
  system_prompt?: string | null
  model_provider?: string | null
  model_name?: string | null
  temperature?: number
  top_p?: number
  use_rag?: boolean
  /** 精排开关：null/省略=跟随服务端配置，true/false=本次请求强制开/关 */
  use_rerank?: boolean | null
  conversation_id?: string | null
  memory_enabled?: boolean
}

/** 引用来源（RAG 命中片段） */
export interface Source {
  title: string
  snippet: string
  source: string | null
  score?: number
}

/** 聊天返回（对应 ChatResponse） */
export interface ChatResponse {
  answer: string
  reasoning: string | null
  thought: string
  model_provider: string
  model_name: string
  total_tokens: number
  latency_ms: number
  conversation_id: string | null
  rag_status: RagStatus
  sources: Source[]
  /** 本次实际生效的打分方式：off / cosine / rerank */
  score_mode?: string | null
  /** 用户是否请求了精排 */
  rerank_requested?: boolean | null
  /** 服务端精排是否真的可用（false 时已回退余弦） */
  rerank_available?: boolean | null
  /** 阶段耗时（ms），用于展示“时间花在哪” */
  timings?: PhaseTimings | null
}

/** RAG 判定四态：off=未启用，no_hit=启用但无过阈值片段，
 *  weak_hit=top1 仅勉强过阈值（不可当可靠依据），hit=正常命中 */
export type RagStatus = 'off' | 'no_hit' | 'weak_hit' | 'hit'

/** SSE 流式聊天的 meta 事件（event: meta），连接建立后先行下发 */
export interface ChatStreamMeta {
  conversation_id: string | null
  model_provider: string
  model_name: string
  prompt_name: string | null
  rag_status: RagStatus
  sources: Source[]
  score_mode?: string | null
  rerank_requested?: boolean | null
  rerank_available?: boolean | null
}

/** SSE 流式聊天的 done 事件（event: done），流结束时一次性给出最终值 */
export interface ChatStreamDone {
  answer: string
  reasoning: string | null
  thought: string
  model_provider: string
  model_name: string
  total_tokens: number
  latency_ms: number
  first_token_ms: number
  conversation_id: string | null
  prompt_name: string | null
  rag_status: RagStatus
  sources: Source[]
  score_mode?: string | null
  rerank_requested?: boolean | null
  rerank_available?: boolean | null
  timings?: PhaseTimings | null
}

/** 聊天页可选项（对应 GET /api/chat/options） */
export interface ChatOptions {
  rerank_enabled: boolean
  rerank_model: string
  hybrid_enabled: boolean
  score_threshold_cosine: number
  score_threshold_rerank: number
}

/** 会话（对应 Conversation） */
export interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
}

/** 会话内的一条消息（getMessages 返回项） */
export interface ChatMessage {
  user_input: string
  assistant_output: string
  model_provider: string
  model_name: string
  created_at: string
}

/** 阶段耗时（毫秒）。老数据没有这一列，所以字段都可选 */
export interface PhaseTimings {
  prepare_ms?: number
  retrieve_ms?: number
  rerank_ms?: number
  llm_ms?: number
  first_token_ms?: number | null
  total_ms?: number
}

/** 运行日志条目（logs 接口返回项） */
export interface ChatLog {
  id: number
  user_input: string
  assistant_output: string
  model_provider: string
  model_name: string
  prompt_name: string | null
  total_tokens: number
  latency_ms: number
  timings?: PhaseTimings | null
  status: string
  error_message: string | null
  created_at: string
}

/** 运行日志分页返回 */
export interface LogListResponse {
  items: ChatLog[]
  total: number
  page: number
  page_size: number
}

/** 运行日志查询参数 */
export interface LogQueryParams {
  page?: number
  page_size?: number
  status?: string
}
