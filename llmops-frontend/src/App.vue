系统主框架
菜单布局
页面容器
导航栏
系统标题
<template>
  <!-- 演示口令门禁：挡住纯路人 -->
  <div v-if="!authed" class="gate">
    <div class="gate-box">
      <h2>工业 LLMOps 演示</h2>
      <p class="gate-sub">请输入演示口令进入（口令见简历 / 说明）</p>
      <el-input v-model="pwd" type="password" placeholder="演示口令" show-password style="max-width: 260px" @keyup.enter="unlock" />
      <el-button type="primary" style="margin-top: 14px" @click="unlock">进入</el-button>
      <p v-if="err" class="gate-err">口令错误，请重试</p>
    </div>
  </div>

  <div v-else class="layout">
    <aside>
      <h2>工业 LLMOps</h2>
      <p class="sub">Ollama + deepseek-r1</p>
      <RouterLink to="/">Chat studio</RouterLink>
      <RouterLink to="/prompts">Prompt 编排</RouterLink>
      <RouterLink to="/knowledge">知识库</RouterLink>
      <RouterLink to="/logs">运行日志</RouterLink>
    </aside>
    <main> <RouterView /> </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

// 口令不写死在前端包里：拿用户输入真发一次请求问后端，200 才放行
// 有 token 即视为已解锁；token 同时随请求发往后端做鉴权
const authed = ref(!!localStorage.getItem('token'))
const pwd = ref('')
const err = ref(false)

async function unlock() {
  try {
    const res = await fetch(`${import.meta.env.VITE_API_BASE_URL}/prompts`, {
      headers: { Authorization: `Bearer ${pwd.value}` },
    })
    if (res.ok) {
      localStorage.setItem('token', pwd.value)
      authed.value = true
    } else {
      err.value = true
      pwd.value = ''
    }
  } catch {
    err.value = true
    pwd.value = ''
  }
}
</script>

<style>
body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f5f7fb; }
.layout { display: flex; min-height: 100vh; }
aside { width: 250px; background: #cdb4cd; color: #0a2025; padding: 20px; }
aside h2 { margin-bottom: 4px; }
.sub { color: #0fb0d8; font-size: 12px; margin-bottom: 20px; margin-top: 0; }
aside a { display: block; color: #0fb0d8; text-decoration: none; margin: 16px 0; }
aside a.router-link-exact-active { color: #ffffff; font-weight: bold; }
main { flex: 1; padding: 24px; }
.card { background: #fff; border-radius: 12px; padding: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 16px; }

/* 门禁页 */
.gate { display: flex; align-items: center; justify-content: center; min-height: 100vh; }
.gate-box { background: #fff; border-radius: 12px; padding: 36px 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); text-align: center; width: 340px; }
.gate-box h2 { margin: 0 0 6px; color: #1F4E79; }
.gate-sub { color: #8A939E; font-size: 13px; margin: 0 0 20px; }
.gate-err { color: #C22B1D; font-size: 13px; margin-top: 12px; }
</style>
