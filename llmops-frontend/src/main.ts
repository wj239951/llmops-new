import { createApp } from 'vue'
import {
  ElAlert,
  ElButton,
  ElForm,
  ElFormItem,
  ElInput,
  ElOption,
  ElPagination,
  ElSelect,
  ElSwitch,
  ElTable,
  ElTableColumn,
  ElUpload,
} from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'

// 按需注册：原来 app.use(ElementPlus) 会把整个组件库（≈1 MB JS）打进首屏 bundle。
// 这里只注册模板里真正用到的 12 个组件，其余交给 tree-shaking。
const components = [
  ElAlert,
  ElButton,
  ElForm,
  ElFormItem,
  ElInput,
  ElOption,
  ElPagination,
  ElSelect,
  ElSwitch,
  ElTable,
  ElTableColumn,
  ElUpload,
]

const app = createApp(App)
components.forEach((component) => {
  app.component(component.name as string, component)
})
app.use(router).mount('#app')
