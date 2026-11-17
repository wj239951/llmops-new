平台核心页面


模型选择
Prompt选择
RAG开关
用户输入(将用户输入与历史会话一同喂给模型)
思考过程展示
最终回答展示
Token统计
耗时统计
<template>
  <div>
    <h1>Chat Studio</h1>

    <div class="card">
      <el-alert
        title="默认走 DeepSeek 云端 API；部署环境有本地 Ollama 时可切本地通道。"
        type="info"
        show-icon
      />
      <el-form label-width="140px" style="margin-top: 18px">
        <el-form-item label="模型提供商">
          <el-select v-model="form.model_provider" style="width: 300px">
          <el-option label="本地 DeepSeek-R1:7b / Ollama" value="ollama" />
          <el-option label="DeepSeek 云端 API" value="deepseek" />
        </el-select>
        </el-form-item>

        <el-form-item label="模型名称">
          <el-select v-model="form.model_name" style="width: 300px">
            <el-option v-for="m in modelOptions" :key="m" :label="m" :value="m" />
          </el-select>
        </el-form-item>

        <el-form-item label="选择 Prompt">
          <el-select v-model="form.prompt_id" clearable style="width: 360px">
            <el-option
              v-for="p in prompts"
              :key="p.id"
              :label="p.name"
              :value="p.id"
              />
            </el-select>
           </el-form-item>

           <el-form-item label="启用 RAG">
            <el-switch v-model="form.use_rag" />
           </el-form-item>
           <el-form-item label="启用精排">
            <el-switch v-model="form.use_rerank" :disabled="!form.use_rag" />
            <span class="rerank-hint">
              服务端精排：{{ serverRerankEnabled ? '可用' : '不可用（请求后会自动回退向量余弦）' }}
            </span>
           </el-form-item>
           <el-form-item label="输入清洗">
            <el-switch v-model="form.use_clean" />
            <span class="rerank-hint">
              服务端清洗：{{ cleanModeText }}
            </span>
           </el-form-item>
           <el-form-item v-if="showModelCleanSwitch" label="精细清洗">
            <el-switch v-model="form.use_clean_model" :disabled="!form.use_clean" />
            <span class="rerank-hint">
              模型通道较慢（GPU 约 0.5s），失败或服务端未允许时自动回退规则清洗
            </span>
           </el-form-item>
            <el-form-item label="会话历史">
              <el-select v-model="form.conversation_id" clearable style="width: 300px" @change="switchConversation">
                <el-option
                  v-for="c in conversations"
                  :key="c.id"
                  :label="c.title + '  (' + c.created_at.replace('T', ' ').slice(0, 16) + ')'"
                  :value="c.id"
                />
              </el-select>
              <el-button type="primary" plain size="small" style="margin-left: 10px" @click="newConversation">
                新建会话
              </el-button>
              <el-button type="danger" plain size="small" style="margin-left: 6px" @click="deleteCurrentConversation">
                删除会话
              </el-button>
            </el-form-item>
            <el-form-item label="启用记忆">
              <el-switch v-model="form.memory_enabled" />
            </el-form-item>
           <el-form-item label="用户输入">
            <el-input v-model="form.query" type="textarea" :rows="5" />
           </el-form-item>
          
        
           <el-button type="primary" :loading="loading" @click="submit">
            发送
           </el-button>
           <el-button v-if="loading" @click="cancel">
            取消
           </el-button>
           <el-alert
             v-if="loading"
             :title="streamPhase"
             type="info"
             :closable="false"
             show-icon
             style="margin-top: 12px"
           />
          </el-form>
         </div>
         <div class="card" v-if="messages.length">
            <h3>历史消息</h3>
            <div v-for="(msg, i) in messages" :key="i" class="message-item">
              <div class="message-user"><strong>用户：</strong>{{ msg.user_input }}</div>
              <div class="message-assistant"><strong>助手：</strong>{{ msg.assistant_output }}</div>
            </div>
         </div>
         <div class="card" v-if="reasoning">
            <h3>思考过程</h3>
            <el-input
              v-model="reasoning"
              type="textarea"
              :rows="10"
              readonly
              
              />
            </div>
            <div class="card" v-if="answer">
                <h3>最终回答</h3>
                <el-input
                v-model="answer"
                type="textarea"
                :rows="8"
                readonly
                />
                <el-alert
                :title="thought"
                type="info"
                style="margin-top: 10px;"
                />
                <el-alert
:title="meta"
type="success"
style="margin-top: 10px"
/>
                <el-alert
                  v-if="scoreMode"
                  :title="scoreModeText"
                  :type="scoreMode === 'rerank' ? 'success' : 'info'"
                  :closable="false"
                  style="margin-top: 8px"
                />
                <el-alert
                  v-if="cleanInfo && cleanInfo.changed"
                  :title="`输入已清洗（${cleanInfo.mode}，${cleanInfo.ms ?? 0}ms）：${cleanInfo.clean_query ?? ''}`"
                  type="success"
                  :closable="false"
                  style="margin-top: 8px"
                />
                <el-alert
                  v-if="cleanInfo && cleanInfo.fallback"
                  :title="cleanFallbackText"
                  type="warning"
                  :closable="false"
                  style="margin-top: 8px"
                />
                <el-alert
                  v-if="timingsText"
                  :title="`耗时拆解：${timingsText}`"
                  type="info"
                  :closable="false"
                  style="margin-top: 8px"
                />
                <el-alert
                  v-if="rerankRequested && rerankAvailable === false"
                  title="已请求精排，但服务端精排不可用，本次已回退为向量余弦"
                  type="warning"
                  :closable="false"
                  style="margin-top: 8px"
                />
<div style="margin-top: 12px">
  <el-alert
    v-if="ragStatus === 'off'"
    title="本次未启用 RAG"
    type="info"
    :closable="false"
  />
  <el-alert
    v-else-if="ragStatus === 'no_hit'"
    title="本次未命中知识库，以下为通用知识"
    type="warning"
    :closable="false"
  />
  <template v-else-if="ragStatus === 'hit' || ragStatus === 'weak_hit'">
    <el-alert
      v-if="ragStatus === 'weak_hit'"
      title="弱命中：top1 相关度只是勉强过阈值，以下来源可能不构成可靠依据"
      type="warning"
      :closable="false"
      style="margin-bottom: 8px"
    />
    <el-button text type="primary" size="small" @click="sourcesExpanded = !sourcesExpanded">
      {{ sourcesExpanded ? '收起' : '展开' }} {{ sources.length }} 个参考来源
    </el-button>
    <div v-show="sourcesExpanded" style="margin-top: 8px">
      <div v-for="(s, i) in sources" :key="i" class="source-item">
        <span class="source-title">{{ s.title || '来源 ' + (i + 1) }}</span>
        <span v-if="s.score != null" class="source-score">score={{ s.score?.toFixed(4) }}</span>
        <a v-if="s.source" :href="s.source" target="_blank" class="source-link">{{ s.source }}</a>
        <p v-if="s.snippet" class="source-snippet">{{ s.snippet }}</p>
      </div>
    </div>
  </template>
</div>
</div>
</div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import type { ChatMessage, ChatRequestPayload, CleanInfo, Conversation, PhaseTimings, Prompt, RagStatus, Source } from '../types'
import { getChatOptions, sendChat, sendChatStream } from '../api/chat'
import { listPrompts } from '../api/prompt'
import { listConversations, createConversation, getMessages, deleteConversation } from '../api/conversations'

const loading = ref(false)
const controller = ref<AbortController | null>(null)
const answer = ref('')
const reasoning = ref('')
const meta = ref('')
const thought = ref('')
const sources = ref<Source[]>([])
const sourcesExpanded = ref(false)
const ragStatus = ref<RagStatus>('off')
// 精排开关与服务端能力：serverRerankEnabled 来自 GET /chat/options（不触发模型加载）
const serverRerankEnabled = ref(false)
const serverRerankModel = ref('')
// 输入清洗：服务端默认模式 + 本次请求的清洗结果
const serverCleanMode = ref('off')
// 服务端是否允许请求级「精细清洗」（INPUT_CLEAN_MODEL_ALLOWED）；false 时按钮不显示
const serverCleanModelAllowed = ref(false)
const cleanInfo = ref<CleanInfo | null>(null)
const cleanModeText = computed(() => {
  if (serverCleanMode.value === 'model') return '模型（失败回退规则）'
  if (serverCleanMode.value === 'rule') return '规则清洗'
  return '未启用（可在 .env 里开 INPUT_CLEAN_MODE）'
})
// 默认通道本身已经是模型时，再给一个「精细清洗」开关没有意义
const showModelCleanSwitch = computed(
  () => serverCleanModelAllowed.value && serverCleanMode.value !== 'model'
)
// 回退提示要区分「服务端不允许」与「模型通道挂了」，两者用户的处理动作不同
const cleanFallbackText = computed(() => {
  const info = cleanInfo.value
  if (!info) return ''
  const tail = info.fallback === 'rule' ? '已回退规则清洗' : '本次未清洗'
  return info.fallback_reason === 'not_allowed'
    ? `输入清洗：服务端未允许精细清洗（${tail}）`
    : `输入清洗：模型通道不可用（${tail}）`
})
// 本次请求的实际打分信息（来自 meta / done / 非流式响应）
const scoreMode = ref('')
// 本次请求的阶段耗时文案（检索 / 精排 / 生成 / 首字），空串表示没有数据
const timingsText = ref('')
const rerankRequested = ref(false)
const rerankAvailable = ref<boolean | null>(null)
const scoreModeText = computed(() => {
  if (scoreMode.value === 'rerank') return '本次打分方式：精排（cross-encoder）'
  if (scoreMode.value === 'cosine') return '本次打分方式：向量余弦'
  return '本次打分方式：未启用（未走 RAG）'
})
const prompts = ref<Prompt[]>([])
const conversations = ref<Conversation[]>([])
const messages = ref<ChatMessage[]>([])
// 流式阶段提示（loading 时显示在发送/取消按钮下方）
const streamPhase = ref('正在准备…')
const form = reactive({
  query: '请介绍一下你自己',
  prompt_id: undefined as number | undefined,
  model_provider: 'deepseek',
  model_name: 'deepseek-chat',
  use_rag: true,
  use_rerank: false,
  use_clean: false,
  use_clean_model: false,
  memory_enabled: true,
  conversation_id: ''
})
const modelOptions = computed(() =>
  form.model_provider === 'ollama'
    ? ['deepseek-r1:7b']
    : ['deepseek-chat', 'deepseek-reasoner']
)
watch(
    () => form.model_provider,
    (value) => {
      form.model_name = value === 'ollama' ? 'deepseek-r1:7b' : 'deepseek-chat'
    }
)
/** 从 meta / done / 非流式响应里读取本次实际打分信息 */
function applyScoreInfo(info: {
  score_mode?: string | null
  rerank_requested?: boolean | null
  rerank_available?: boolean | null
}) {
  scoreMode.value = info.score_mode ?? ''
  rerankRequested.value = info.rerank_requested ?? false
  rerankAvailable.value = info.rerank_available ?? null
}

/** 从 meta / done / 非流式响应里读取本次输入清洗结果 */
function applyCleanInfo(info: { clean?: CleanInfo | null }) {
  cleanInfo.value = info.clean ?? null
}

/** 毫秒转短文本：1200 → 1.2s */
function fmtMs(ms?: number | null): string {
  if (!ms || ms <= 0) return ''
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`
}

/** 阶段耗时拆解：检索 / 精排 / 生成 / 首字，只显示有值的项 */
function formatTimings(t?: PhaseTimings | null): string {
  if (!t) return ''
  return [
    t.retrieve_ms ? `检索 ${fmtMs(t.retrieve_ms)}` : '',
    t.rerank_ms ? `精排 ${fmtMs(t.rerank_ms)}` : '',
    t.llm_ms ? `生成 ${fmtMs(t.llm_ms)}` : '',
    t.first_token_ms ? `首字 ${fmtMs(t.first_token_ms)}` : '',
  ].filter(Boolean).join(' · ')
}

/** 把后端 status 事件的 phase 翻译成用户可读的阶段文案 */
function streamPhaseText(phase: 'preparing' | 'retrieving' | 'generating'): string {
  if (phase === 'retrieving') return '正在检索知识库…'
  if (phase === 'generating') return '正在生成…'
  return '正在准备…'
}
async function loadPrompts() {prompts.value = await listPrompts()
}
async function loadConversations() {
  conversations.value = await listConversations()
}
async function loadChatOptions() {
  try {
    const opts = await getChatOptions()
    serverRerankEnabled.value = opts.rerank_enabled
    serverRerankModel.value = opts.rerank_model
    form.use_rerank = opts.rerank_enabled
    serverCleanMode.value = opts.clean_mode ?? 'off'
    serverCleanModelAllowed.value = opts.clean_model_allowed ?? false
    form.use_clean = Boolean(opts.clean_enabled)
  } catch {
    // 可选接口失败不影响聊天；开关保持关闭并显示不可用
    serverRerankEnabled.value = false
    serverCleanMode.value = 'off'
    serverCleanModelAllowed.value = false
  }
}
async function newConversation() {
  const newConv = await createConversation()
  conversations.value.unshift(newConv)
  form.conversation_id = newConv.id
  messages.value = []
  answer.value = ''
  reasoning.value = ''
  thought.value = ''
  meta.value = ''
  sources.value = []
  ragStatus.value = 'off'
  scoreMode.value = ''
  timingsText.value = ''
  rerankRequested.value = false
  rerankAvailable.value = null
  cleanInfo.value = null
}
async function switchConversation() {
  if (!form.conversation_id) {
    messages.value = []
    return
  }
  try {
    const res = await getMessages(form.conversation_id)
    messages.value = res
  } catch {
    messages.value = []
  }
}
async function deleteCurrentConversation() {
  if (!form.conversation_id) return
  await deleteConversation(form.conversation_id)
  form.conversation_id = ''
  messages.value = []
  answer.value = ''
  reasoning.value = ''
  thought.value = ''
  meta.value = ''
  sources.value = []
  ragStatus.value = 'off'
  scoreMode.value = ''
  timingsText.value = ''
  rerankRequested.value = false
  rerankAvailable.value = null
  cleanInfo.value = null
  await loadConversations()
}
async function submit() {
  loading.value = true
  streamPhase.value = '正在准备…'
  answer.value = ''
  reasoning.value = ''
  thought.value = ''
  meta.value = ''
  sources.value = []
  sourcesExpanded.value = false
  ragStatus.value = 'off'
  scoreMode.value = ''
  timingsText.value = ''
  rerankRequested.value = false
  rerankAvailable.value = null
  cleanInfo.value = null
  const ctrl = new AbortController()
  // RAG 关闭时精排无意义，显式传 null 表示"本次不干预，跟随服务端"
  const payload: ChatRequestPayload = {
    ...form,
    use_rerank: form.use_rag ? form.use_rerank : null,
    // 清洗与 RAG 无关，任何时候都按开关显式下发（false 表示本次不要清洗）
    use_clean: form.use_clean,
    // 清洗关掉时通道开关无意义，显式传 false，避免服务端要做一次「被忽略」的判断
    use_clean_model: form.use_clean && form.use_clean_model,
  }
  controller.value = ctrl

  // 是否已收到实质输出（reasoning/delta）：决定失败后能否安全降级到非流式
  let hasSubstantiveOutput = false
  // 是否收到 done：只有 done 才算流式正常完成
  let doneReceived = false
  let streamError = ''

  try {
    try {
      await sendChatStream(
        payload,
        {
          onStatus(phase) {
            streamPhase.value = streamPhaseText(phase)
          },
          onMeta(m) {
            if (m.conversation_id) form.conversation_id = m.conversation_id
            sources.value = m.sources ?? []
            ragStatus.value = m.rag_status ?? 'off'
            meta.value = `${m.model_provider} / ${m.model_name}`
            applyScoreInfo(m)
            applyCleanInfo(m)
          },
          onReasoning(delta) {
            hasSubstantiveOutput = true
            reasoning.value += delta
          },
          onDelta(content) {
            hasSubstantiveOutput = true
            answer.value += content
          },
          onDone(d) {
            doneReceived = true
            answer.value = d.answer
            reasoning.value = d.reasoning ?? ''
            thought.value = d.thought
            sources.value = d.sources ?? []
            ragStatus.value = d.rag_status ?? 'off'
            if (d.conversation_id) form.conversation_id = d.conversation_id
            meta.value = `${d.model_provider} / ${d.model_name} / tokens=${d.total_tokens} / ${d.latency_ms}ms`
            applyScoreInfo(d)
            applyCleanInfo(d)
            timingsText.value = formatTimings(d.timings)
          },
          onError(message) {
            streamError = message || '流式生成失败'
            ElMessage.error(streamError)
          },
        },
        ctrl.signal,
      )
    } catch (error) {
      if (ctrl.signal.aborted) throw error
      streamError = error instanceof Error ? error.message : '流式请求失败'
      // 已经有部分输出时及时提示；首包前的失败交给下面的降级分支统一提示
      if (hasSubstantiveOutput) ElMessage.error(streamError)
    }

    if (ctrl.signal.aborted) {
      if (!hasSubstantiveOutput) answer.value = '已取消'
      return
    }

    if (!doneReceived) {
      if (!hasSubstantiveOutput) {
        // 首个实质输出前就失败：降级调用原有非流式 /chat，保持旧行为
        ElMessage.warning('流式输出不可用，已自动降级为非流式请求')
        streamPhase.value = '正在重试（非流式）…'
        const res = await sendChat(payload, ctrl.signal)
        answer.value = res.answer
        reasoning.value = res.reasoning ?? ''
        thought.value = res.thought
        sources.value = res.sources ?? []
        ragStatus.value = res.rag_status ?? 'off'
        meta.value = `${res.model_provider} / ${res.model_name} / tokens=${res.total_tokens} / ${res.latency_ms}ms`
        applyScoreInfo(res)
        applyCleanInfo(res)
        timingsText.value = formatTimings(res.timings)
      } else if (!streamError) {
        // 已经有内容但流没给 done 就断开：保留已有内容，提示可能不完整
        ElMessage.error('流式输出中断，回答可能不完整')
      }
    }

    streamPhase.value = '正在整理结果…'
    await loadConversations()
    if (form.conversation_id) {
      await switchConversation()
    }
  } catch {
    if (ctrl.signal.aborted) {
      if (!hasSubstantiveOutput) answer.value = '已取消'
    }
    // 其它错误：流式路径上面已用 ElMessage 提示；降级 sendChat 的错误由 request.ts 拦截器统一 toast
  } finally {
    loading.value = false
    streamPhase.value = ''
    controller.value = null
  }
}

function cancel() {
  controller.value?.abort()
}
onMounted(() => { loadPrompts(); loadConversations(); loadChatOptions() })
</script>

<style scoped>
.rerank-hint {
  margin-left: 10px;
  font-size: 12px;
  color: #909399;
}
.source-item {
  padding: 8px 0;
  border-bottom: 1px solid #eee;
}
.source-title {
  font-weight: 600;
  font-size: 14px;
}
.source-score {
  margin-left: 8px;
  font-size: 12px;
  color: #909399;
}
.source-link {
  display: block;
  font-size: 12px;
  color: #409eff;
  word-break: break-all;
}
.source-snippet {
  font-size: 13px;
  color: #666;
  margin: 4px 0 0;
}
.message-item {
  padding: 10px 0;
  border-bottom: 1px solid #eee;
}
.message-user {
  margin-bottom: 6px;
  color: #333;
}
.message-assistant {
  color: #555;
  white-space: pre-wrap;
}
</style>
