<template>
  <div>
    <h1>知识库</h1>
    <div class="card">
      <h3>手动输入</h3>
      <el-input v-model="form.title" placeholder="文档标题" style="margin-bottom:
10px" />
      <el-input v-model="form.content" type="textarea" :rows="8" placeholder="文档内容" />
      <el-button type="primary" style="margin-top: 10px" @click="save">{{ KnowledgeId ? '保存修改' : '新增文档' }}</el-button>
    </div>
    <div class="card">
      <h3>文件上传</h3>
      <el-upload
        ref="uploadRef"
        :auto-upload="false"
        :on-change="handleFileChange"
        :limit="1"
        accept=".txt,.md,.pdf,.py,.json"
      >
        <el-button type="primary">选择文件</el-button>
        <template #tip>
          <div class="el-upload__tip">支持 txt/md/pdf/py/json 格式</div>
        </template>
      </el-upload>
      <el-button type="success" style="margin-top: 10px" @click="upload" :disabled="!selectedFile">上传并解析</el-button>
    </div>
    <div class="card">
        <el-table :data="rows">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="title" label="标题" />
        <el-table-column prop="source" label="来源" />
        <el-table-column prop="created_at" label="创建时间" />
        <el-table-column label="操作1" width="120">
          <template #default="{ row }">
            <el-button type="primary" size="small" @click="editRow(row)">修改</el-button>
          </template>
        </el-table-column>
         <el-table-column label="操作2" width="120">
          <template #default="{ row }">
            <el-button type="primary" size="small" @click="deleteRow(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <!-- 服务端分页：一次只取一页，避免几千篇文档时把全部正文拉进浏览器 -->
      <el-pagination
        v-model:current-page="page"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50]"
        layout="total, sizes, prev, pager, next"
        style="margin-top: 12px"
        @current-change="load"
        @size-change="handleSizeChange"
      />
     </div>
    </div>
</template>
<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import type { KnowledgeDocument } from '../types'
import { createKnowledge, listKnowledgePage, updateKnowledge, deleteKnowledge, uploadKnowledge } from '../api/knowledge'
import { ElMessage, ElMessageBox, type UploadFile, type UploadInstance } from 'element-plus'

const rows = ref<KnowledgeDocument[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(10)
const KnowledgeId = ref<number | null>(null) // 正在编辑的知识库ID
const selectedFile = ref<File | null>(null)
const uploadRef = ref<UploadInstance>()
const form = reactive({
  title: 'LLMOps 说明',
  content: 'Prompt 管理、RAG、模型调用、日志监控、工具调用是工业 LLMOps 的核心能力。本地版使用Ollama 调用 deepseek-R1:7b，云端使用 DeepSeek API。'
})

function handleFileChange(file: UploadFile) {
  selectedFile.value = file.raw ?? null
}

async function upload() {
  if (!selectedFile.value) return
  try {
    await uploadKnowledge(selectedFile.value)
    ElMessage.success('文件上传成功')
    selectedFile.value = null
    uploadRef.value?.clearFiles()
    page.value = 1
    await load()
  } catch {
    // 错误提示已由 request.ts 的响应拦截器统一处理
  }
}
async function save() { 
  if (KnowledgeId.value) { // 修改
    await updateKnowledge(KnowledgeId.value, form)
  } else { // 新增
    await createKnowledge(form)
  }
  await load()
  KnowledgeId.value = null
  form.title = ''
  form.content = ''
}
async function editRow(row: KnowledgeDocument) {
  KnowledgeId.value = row.id       // 记住正在编辑这条
  form.title = row.title           // 逐个字段赋值，避免污染 id/created_at
  form.content = row.content
}
async function deleteRow(row: KnowledgeDocument) {
  try {
    await ElMessageBox.confirm(
      `确定删除文档「${row.title}」吗？删除后不可恢复。`,
      '删除确认',
      { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning' }
    )
  } catch {
    return // 用户取消删除
  }
  await deleteKnowledge(row.id)
  // 删掉当前页最后一条时回退一页，避免停在空白页
  if (rows.value.length === 1 && page.value > 1) page.value -= 1
  await load()
}

function handleSizeChange() {
  page.value = 1
  load()
}

async function load() {
  const res = await listKnowledgePage(page.value, pageSize.value)
  rows.value = res.items
  total.value = res.total
  // 页码越界（例如删到只剩前几页）时回到最后一页
  if (rows.value.length === 0 && total.value > 0 && page.value > 1) {
    page.value = Math.max(1, Math.ceil(total.value / pageSize.value))
    const last = await listKnowledgePage(page.value, pageSize.value)
    rows.value = last.items
    total.value = last.total
  }
}
onMounted(load)
</script>
