//聊天接口封装
import { http } from './request'
import type {
  ChatOptions,
  ChatRequestPayload,
  ChatResponse,
  ChatStreamDone,
  ChatStreamMeta,
} from '../types'

export function sendChat(data: ChatRequestPayload, signal?: AbortSignal) {
  return http.post<ChatResponse>('/chat', data, { signal })
}

/** 聊天页可选项：精排是否可用、模型名与两套阈值（只读服务端配置，不加载模型） */
export function getChatOptions() {
  return http.get<ChatOptions>('/chat/options')
}

/** SSE status 事件的阶段（对应后端 event: status 的 phase 字段） */
export type ChatStreamPhase = 'preparing' | 'retrieving' | 'generating'

/** 流式聊天的事件回调集合 */
export interface ChatStreamHandlers {
  onStatus?: (phase: ChatStreamPhase) => void
  onMeta?: (meta: ChatStreamMeta) => void
  onReasoning?: (delta: string) => void
  onDelta?: (content: string) => void
  onDone?: (done: ChatStreamDone) => void
  onError?: (message: string) => void
}

/** 解析单帧 SSE 文本（形如 "event: delta\ndata: {...}"），返回事件名与 data 解析结果 */
function parseSseFrame(rawFrame: string): { event: string; data: unknown } | null {
  let event = ''
  const dataLines: string[] = []
  for (const line of rawFrame.split('\n')) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      // SSE 规范：去掉 "data:" 前缀后若紧跟一个空格则去掉这一个空格，其余内容原样保留
      let value = line.slice('data:'.length)
      if (value.startsWith(' ')) value = value.slice(1)
      dataLines.push(value)
    }
  }
  if (!event && dataLines.length === 0) return null
  const dataText = dataLines.join('\n')
  if (dataText === '') return { event, data: null }
  try {
    return { event, data: JSON.parse(dataText) as unknown }
  } catch {
    // data 不是合法 JSON 时保留原始文本，交给上层按字符串处理（不中断整条流）
    return { event, data: dataText }
  }
}

/**
 * 调用后端 SSE 流式聊天接口 POST /api/chat/stream，按事件增量回调。
 * 协议：`event: <name>\ndata: <json>\n\n`，data 可能多行（用换行连接后 JSON.parse）。
 * 鉴权与 request.ts 一致：Authorization: Bearer <localStorage token>；
 * 401 行为也与 axios 响应拦截器一致：清 token + location.reload。
 * AbortSignal 直接透传给 fetch，取消时会抛 AbortError。
 */
export async function sendChatStream(
  payload: ChatRequestPayload,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const token = localStorage.getItem('token')
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  }
  if (token) headers.Authorization = `Bearer ${token}`

  const baseURL = import.meta.env.VITE_API_BASE_URL as string
  const resp = await fetch(`${baseURL}/chat/stream`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal,
  })

  if (resp.status === 401) {
    // 与 src/api/request.ts 响应拦截器保持一致：口令失效清 token 回门禁页重来
    localStorage.removeItem('token')
    location.reload()
    throw new Error('登录已过期，请重新登录')
  }
  if (!resp.ok) {
    throw new Error(`流式请求失败：HTTP ${resp.status}`)
  }
  if (!resp.body) {
    throw new Error('流式请求失败：响应中没有可读取的 body')
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  /** 把缓冲区按空行分帧并分发给 handlers；flush=true 时把剩余内容也当成最后一帧 */
  const dispatchFrames = (flush: boolean) => {
    // 兼容服务端可能的 \r\n 换行
    buffer = buffer.replace(/\r\n/g, '\n')
    const frames = buffer.split('\n\n')
    buffer = flush ? '' : (frames.pop() ?? '')
    for (const frame of frames) {
      const parsed = parseSseFrame(frame)
      if (!parsed) continue
      const data = (parsed.data ?? {}) as Record<string, unknown>
      switch (parsed.event) {
        case 'status':
          handlers.onStatus?.(data.phase as ChatStreamPhase)
          break
        case 'meta':
          handlers.onMeta?.(parsed.data as ChatStreamMeta)
          break
        case 'reasoning':
          if (typeof data.delta === 'string') handlers.onReasoning?.(data.delta)
          break
        case 'delta':
          if (typeof data.content === 'string') handlers.onDelta?.(data.content)
          break
        case 'done':
          handlers.onDone?.(parsed.data as ChatStreamDone)
          break
        case 'error':
          handlers.onError?.(
            typeof data.message === 'string' ? data.message : '流式请求出错',
          )
          break
        default:
          // 未知事件名直接忽略，保证向后兼容
          break
      }
    }
  }

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      dispatchFrames(false)
    }
    // flush：把 TextDecoder 与帧缓冲区里的残余内容处理掉
    buffer += decoder.decode()
    dispatchFrames(true)
  } finally {
    reader.releaseLock()
  }
}
