<template>
<div>
    <h1>运行日志</h1>
    <div class="card">
      <el-select v-model="status" placeholder="状态筛选" clearable style="width: 140px" @change="onFilterChange">
        <el-option label="全部" value="" />
        <el-option label="成功" value="success" />
        <el-option label="失败" value="failed" />
      </el-select>
      <el-button style="margin-left: 8px" @click="load">刷新</el-button>
      <el-table :data="rows" style="margin-top: 12px">
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="model_provider" label="模型商" width="110" />
        <el-table-column prop="model_name" label="模型" width="150" />
        <el-table-column prop="user_input" label="用户输入" />
        <!-- 阶段耗时瀑布条：检索 / 精排 / 生成 各占一段，鼠标悬停看具体毫秒 -->
        <el-table-column label="阶段耗时" width="300">
          <template #default="{ row }">
            <div v-if="row.timings" class="phase-wrap" :title="timingText(row)">
              <div class="phase-bar">
                <span
                  v-for="p in phaseSegments(row)"
                  :key="p.key"
                  class="phase-seg"
                  :style="{ width: p.width + '%', background: p.color }"
                />
              </div>
              <span class="phase-total">{{ row.latency_ms }} ms</span>
            </div>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>
        <el-table-column prop="total_tokens" label="Tokens" width="90" />
        <el-table-column prop="status" label="状态" width="90" />
        <el-table-column prop="created_at" label="时间" width="180" />
      </el-table>
      <el-pagination
        style="margin-top: 12px; justify-content: flex-end"
        layout="total, prev, pager, next"
        :total="total"
        :page-size="pageSize"
        v-model:current-page="page"
        @current-change="load"
      />
   </div>
  </div>
 </template>
<script setup lang="ts">
import { onMounted, ref } from 'vue'
import type { ChatLog, LogQueryParams } from '../types'
import { listLogs } from '../api/logs'

const rows = ref<ChatLog[]>([])
const total = ref(0)       // 总条数，分页条用
const page = ref(1)        // 当前页
const pageSize = ref(10)   // 每页条数
const status = ref('')     // 状态筛选，空=全部

// 阶段与颜色：检索（蓝）→ 精排（橙）→ 生成（绿）→ 准备（灰）
const PHASES = [
  { key: 'retrieve_ms', label: '检索', color: '#409eff' },
  { key: 'rerank_ms', label: '精排', color: '#e6a23c' },
  { key: 'llm_ms', label: '生成', color: '#67c23a' },
  { key: 'prepare_ms', label: '准备', color: '#909399' },
] as const

function totalOf(row: ChatLog): number {
  const t = row.timings || {}
  const sum = PHASES.reduce((acc, p) => acc + Number(t[p.key] || 0), 0)
  // 兜底：各阶段之和可能略小于总耗时（日志写库、序列化等），用总耗时当分母更直观
  return Math.max(sum, Number(t.total_ms || 0), 1)
}

function phaseSegments(row: ChatLog) {
  const t = row.timings || {}
  const total = totalOf(row)
  return PHASES
    .map((p) => ({ key: p.key, label: p.label, color: p.color, ms: Number(t[p.key] || 0) }))
    .filter((p) => p.ms > 0)
    .map((p) => ({ ...p, width: Math.max((p.ms / total) * 100, 1) }))
}

function timingText(row: ChatLog): string {
  const t = row.timings || {}
  const parts = PHASES
    .filter((p) => Number(t[p.key] || 0) > 0)
    .map((p) => `${p.label} ${Number(t[p.key])} ms`)
  const ttft = t.first_token_ms ? `首字 ${t.first_token_ms} ms` : ''
  return [...parts, ttft, `总 ${row.latency_ms} ms`].filter(Boolean).join(' · ')
}

async function load() {
  const params: LogQueryParams = { page: page.value, page_size: pageSize.value }
  if (status.value) params.status = status.value  // 选了才传，不传后端返回全部
  const data = await listLogs(params)
  rows.value = data.items
  total.value = data.total
}
function onFilterChange() {  // 切换状态时回到第1页，避免停在空白页
  page.value = 1
  load()
}
onMounted(load)
</script>
<style scoped>
.phase-wrap { display: flex; align-items: center; gap: 8px; }
.phase-bar {
  flex: 1;
  display: flex;
  height: 10px;
  border-radius: 5px;
  overflow: hidden;
  background: #f0f2f5;
}
.phase-seg { display: block; height: 100%; }
.phase-total { font-size: 12px; color: #606266; white-space: nowrap; }
.muted { color: #c0c4cc; }
</style>
