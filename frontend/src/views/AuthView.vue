<template>
  <div class="auth-view">
    <el-card class="auth-card" shadow="never">
      <div class="brand">职引</div>
      <div class="subtitle">请先登录后使用求职助手</div>

      <el-tabs v-model="tab" stretch>
        <el-tab-pane label="登录" name="login">
          <el-form :model="loginForm" label-position="top" @submit.prevent="handleLogin">
            <el-form-item label="用户名">
              <el-input v-model="loginForm.username" autocomplete="username" />
            </el-form-item>
            <el-form-item label="密码">
              <el-input
                v-model="loginForm.password"
                type="password"
                show-password
                autocomplete="current-password"
                @keydown.enter.prevent="handleLogin"
              />
            </el-form-item>
            <el-button type="primary" class="submit-btn" :loading="loading" @click="handleLogin">
              登录
            </el-button>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="注册" name="register">
          <el-form :model="registerForm" label-position="top" @submit.prevent="handleRegister">
            <el-form-item label="用户名">
              <el-input v-model="registerForm.username" autocomplete="username" />
            </el-form-item>
            <el-form-item label="密码">
              <el-input
                v-model="registerForm.password"
                type="password"
                show-password
                autocomplete="new-password"
              />
            </el-form-item>
            <el-form-item label="确认密码">
              <el-input
                v-model="registerForm.confirmPassword"
                type="password"
                show-password
                autocomplete="new-password"
                @keydown.enter.prevent="handleRegister"
              />
            </el-form-item>
            <el-button type="primary" class="submit-btn" :loading="loading" @click="handleRegister">
              注册并登录
            </el-button>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const router = useRouter()
const route = useRoute()

const tab = ref<'login' | 'register'>('login')
const loading = ref(false)

const loginForm = reactive({
  username: '',
  password: '',
})

const registerForm = reactive({
  username: '',
  password: '',
  confirmPassword: '',
})

function nextPath() {
  return typeof route.query.redirect === 'string' ? route.query.redirect : '/'
}

async function handleLogin() {
  if (loading.value) return
  if (!loginForm.username.trim() || !loginForm.password.trim()) {
    ElMessage.warning('请输入用户名和密码')
    return
  }

  loading.value = true
  try {
    await userStore.login(loginForm.username.trim(), loginForm.password)
    ElMessage.success('登录成功')
    router.replace(nextPath())
  } finally {
    loading.value = false
  }
}

async function handleRegister() {
  if (loading.value) return
  if (!registerForm.username.trim() || !registerForm.password.trim()) {
    ElMessage.warning('请输入用户名和密码')
    return
  }
  if (registerForm.password !== registerForm.confirmPassword) {
    ElMessage.warning('两次输入的密码不一致')
    return
  }

  loading.value = true
  try {
    await userStore.register(registerForm.username.trim(), registerForm.password)
    ElMessage.success('注册成功，已自动登录')
    router.replace(nextPath())
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.auth-view {
  min-height: calc(100vh - 60px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.auth-card {
  width: 420px;
  border-radius: 12px;
}

.brand {
  font-size: 26px;
  font-weight: 700;
  color: #303133;
  text-align: center;
  margin-bottom: 6px;
}

.subtitle {
  font-size: 13px;
  color: #909399;
  text-align: center;
  margin-bottom: 14px;
}

.submit-btn {
  width: 100%;
}
</style>
