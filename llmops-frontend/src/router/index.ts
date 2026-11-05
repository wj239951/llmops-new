//路由管理
// URL映射
// 页面切换
// 菜单跳转
// 视图改成路由级懒加载：首屏只加载当前页面，其余页面按需下载，
// 避免所有页面 + 组件库一次性打进同一个 1 MB 的 bundle。
import { createRouter, createWebHistory } from 'vue-router'
const ChatStudio = () => import('../views/ChatStudio.vue')
const PromptManager = () => import('../views/PromptManager.vue')
const KnowledgeBase = () => import('../views/KnowledgeBase.vue')
const RunLogs = () => import('../views/runlogs.vue')
export default createRouter({
history: createWebHistory(),
routes: [
{ path: '/', component: ChatStudio },
{ path: '/prompts', component: PromptManager },
{ path: '/knowledge', component: KnowledgeBase },
{ path: '/logs', component: RunLogs },
// 兜底路由：所有未匹配的路径重定向回首页，避免空白页
{ path: '/:pathMatch(.*)*', redirect: '/' }
]
})
