//知识库接口封装
import { http } from './request'
import type { KnowledgeDocument, KnowledgePage, KnowledgePayload } from '../types'

/** 全量列表（保留给脚本/兼容旧调用；数据量大时请用 listKnowledgePage） */
export function listKnowledge() {
  return http.get<KnowledgeDocument[]>('/knowledge')
}

/** 分页列表：表格用，一次只取一页，避免把全部正文拉回前端 */
export function listKnowledgePage(page = 1, pageSize = 20) {
  return http.get<KnowledgePage>('/knowledge/paged', {
    params: { page, page_size: pageSize },
  })
}

export function createKnowledge(data: KnowledgePayload) {
  return http.post<KnowledgeDocument>('/knowledge', data)
}

export function updateKnowledge(id: number, form: KnowledgePayload) {
  return http.put<KnowledgeDocument>(`/knowledge/${id}`, form)
}

export function deleteKnowledge(id: number) {
  return http.delete<{ message: string }>(`/knowledge/${id}`)
}

export function uploadKnowledge(file: File, title?: string) {
  const formData = new FormData()
  formData.append('file', file)
  if (title) formData.append('title', title)
  return http.post<KnowledgeDocument>('/knowledge/upload', formData)
}
