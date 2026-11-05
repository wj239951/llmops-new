import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  css: {
    preprocessorOptions: {
      scss: { api: 'modern-compiler' },
    },
  },
  // 拆包：视图已按路由懒加载，这里再把「不会变的基础库」和「体积最大的组件库」分开，
  // 避免升级业务代码时把 1 MB 的 vendor 缓存全部打掉。
  build: {
    rolldownOptions: {
      output: {
        advancedChunks: {
          groups: [
            { name: 'element-plus', test: /node_modules[\\/]element-plus/ },
            { name: 'vendor', test: /node_modules[\\/](vue|@vue|vue-router|axios)[\\/]/ },
          ],
        },
      },
    },
  },
  // 本机没有 Docker，用 vite preview 顶替 Dockerfile 里的 nginx 容器：
  // 同源 /api 反代到本机后端，行为与 nginx.conf 一致（SPA 回退由 preview 默认提供）。
  preview: {
    host: '127.0.0.1',
    port: 3000,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
