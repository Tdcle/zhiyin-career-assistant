<template>
  <el-header class="app-header">
    <div class="logo" @click="goHome">职引</div>

    <div class="current-page" v-if="route.path !== '/auth'">
      <h1 class="current-page-title">{{ currentPageTitle }}</h1>
    </div>
    <div class="current-page" v-else />

    <div class="user-info">
      <template v-if="userStore.isAuthenticated">
        <div class="user-pill">
          <el-avatar class="user-avatar" :size="34" :src="userAvatarSrc" />
          <div class="user-meta">
            <span class="user-name">{{ displayName }}</span>
            <span class="user-id">ID: {{ userStore.profile?.user_id }}</span>
          </div>
        </div>

        <el-dropdown trigger="click" @command="handleUserCommand">
          <button class="menu-btn" type="button" aria-label="用户菜单">
            <el-icon><ArrowDown /></el-icon>
          </button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="change-password">修改密码</el-dropdown-item>
              <el-dropdown-item command="logout" divided>退出登录</el-dropdown-item>
              <el-dropdown-item command="manage-memory">记忆管理</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </template>
      <span v-else class="no-user">未登录</span>
    </div>
  </el-header>

  <el-dialog
    v-model="passwordDialogVisible"
    title="修改密码"
    width="420px"
    :close-on-click-modal="false"
    destroy-on-close
  >
    <el-form label-position="top">
      <el-form-item label="当前密码">
        <el-input
          v-model="passwordForm.currentPassword"
          type="password"
          show-password
          autocomplete="current-password"
          @keydown.enter.prevent="submitPasswordChange"
        />
      </el-form-item>
      <el-form-item label="新密码">
        <el-input
          v-model="passwordForm.newPassword"
          type="password"
          show-password
          autocomplete="new-password"
        />
      </el-form-item>
      <el-form-item label="确认新密码">
        <el-input
          v-model="passwordForm.confirmPassword"
          type="password"
          show-password
          autocomplete="new-password"
          @keydown.enter.prevent="submitPasswordChange"
        />
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="closePasswordDialog">取消</el-button>
      <el-button type="primary" :loading="passwordSubmitting" @click="submitPasswordChange">
        确认修改
      </el-button>
    </template>
  </el-dialog>

  <el-dialog
    v-model="memoryDialogVisible"
    title="记忆管理"
    width="760px"
    :close-on-click-modal="false"
    destroy-on-close
  >
    <div class="memory-toolbar">
      <span class="memory-hint">修改或删除不准确的长期记忆。</span>
      <el-button size="small" :loading="memoryLoading" @click="loadMemories">刷新</el-button>
    </div>

    <el-empty v-if="!memoryLoading && memories.length === 0" description="暂无长期记忆" />

    <el-table
      v-else
      v-loading="memoryLoading"
      :data="memories"
      max-height="420"
      class="memory-table"
      row-key="id"
    >
      <el-table-column label="类型" width="150">
        <template #default="{ row }">
          <el-select v-if="editingMemoryId === row.id" v-model="memoryForm.factKey" size="small">
            <el-option
              v-for="option in memoryKeyOptions"
              :key="option.value"
              :label="option.label"
              :value="option.value"
            />
          </el-select>
          <el-tag v-else size="small">{{ memoryKeyLabel(row.fact_key) }}</el-tag>
        </template>
      </el-table-column>

      <el-table-column label="内容" min-width="260">
        <template #default="{ row }">
          <el-input
            v-if="editingMemoryId === row.id"
            v-model="memoryForm.factValue"
            type="textarea"
            :rows="2"
            maxlength="1000"
            show-word-limit
          />
          <div v-else class="memory-value">{{ row.fact_value }}</div>
        </template>
      </el-table-column>

      <el-table-column label="权重" width="96">
        <template #default="{ row }">
          <el-input-number
            v-if="editingMemoryId === row.id"
            v-model="memoryForm.importance"
            size="small"
            :min="1"
            :max="5"
            controls-position="right"
          />
          <span v-else>{{ row.importance }}</span>
        </template>
      </el-table-column>

      <el-table-column label="来源" width="120">
        <template #default="{ row }">
          <span class="memory-source">{{ row.source || '-' }}</span>
        </template>
      </el-table-column>

      <el-table-column label="操作" width="150" fixed="right">
        <template #default="{ row }">
          <template v-if="editingMemoryId === row.id">
            <el-button
              size="small"
              type="primary"
              :loading="memorySavingId === row.id"
              @click="saveMemory(row.id)"
            >
              保存
            </el-button>
            <el-button size="small" @click="cancelMemoryEdit">取消</el-button>
          </template>
          <template v-else>
            <el-button size="small" @click="startMemoryEdit(row)">修改</el-button>
            <el-button
              size="small"
              type="danger"
              link
              :loading="memoryDeletingId === row.id"
              @click="deleteMemory(row)"
            >
              删除
            </el-button>
          </template>
        </template>
      </el-table-column>
    </el-table>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRoute, useRouter } from 'vue-router'

import userAvatarSrc from '@/assets/user.png'
import { apiDeleteMemory, apiGetMemories, apiUpdateMemory, type MemoryFact } from '@/api/user'
import { useChatStore } from '@/stores/chat'
import { useInterviewStore } from '@/stores/interview'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const chatStore = useChatStore()
const interviewStore = useInterviewStore()
const route = useRoute()
const router = useRouter()

const passwordDialogVisible = ref(false)
const passwordSubmitting = ref(false)
const passwordForm = reactive({
  currentPassword: '',
  newPassword: '',
  confirmPassword: '',
})
const memoryDialogVisible = ref(false)
const memoryLoading = ref(false)
const memorySavingId = ref<number | null>(null)
const memoryDeletingId = ref<number | null>(null)
const memories = ref<MemoryFact[]>([])
const editingMemoryId = ref<number | null>(null)
const memoryForm = reactive({
  factKey: '',
  factValue: '',
  confidence: 0.75,
  importance: 3,
})
const memoryKeyOptions = [
  { label: '目标岗位', value: 'desired_role' },
  { label: '目标城市', value: 'desired_city' },
  { label: '经验要求', value: 'desired_experience' },
  { label: '薪资要求', value: 'desired_salary' },
  { label: '核心技能', value: 'core_skill' },
  { label: '偏好记录', value: 'preference_note' },
  { label: '经验记录', value: 'experience_note' },
  { label: '城市记录', value: 'location_note' },
  { label: '薪资记录', value: 'salary_note' },
  { label: '技能记录', value: 'skill_note' },
  { label: '学历记录', value: 'education_note' },
]
const memoryKeyLabels: Record<string, string> = Object.fromEntries(
  memoryKeyOptions.map((item) => [item.value, item.label]),
)

const currentPageTitle = computed(() => (route.path === '/interview' ? '模拟面试' : '求职大厅'))
const displayName = computed(() => userStore.currentUser || '未命名用户')

function goHome() {
  if (userStore.isAuthenticated) {
    router.push('/')
  } else {
    router.push('/auth')
  }
}

function resetPasswordForm() {
  passwordForm.currentPassword = ''
  passwordForm.newPassword = ''
  passwordForm.confirmPassword = ''
}

function closePasswordDialog() {
  if (passwordSubmitting.value) return
  passwordDialogVisible.value = false
  resetPasswordForm()
}

async function handleLogout() {
  await userStore.logout()
  chatStore.reset()
  interviewStore.reset()
  router.replace('/auth')
}

async function handleUserCommand(command: string | number | object) {
  if (command === 'change-password') {
    passwordDialogVisible.value = true
    return
  }
  if (command === 'manage-memory') {
    openMemoryDialog()
    return
  }
  if (command === 'logout') {
    await handleLogout()
  }
}

async function submitPasswordChange() {
  if (passwordSubmitting.value) return
  if (!passwordForm.currentPassword.trim() || !passwordForm.newPassword.trim() || !passwordForm.confirmPassword.trim()) {
    ElMessage.warning('请完整填写密码信息')
    return
  }
  if (passwordForm.newPassword.length < 6) {
    ElMessage.warning('新密码至少 6 位')
    return
  }
  if (passwordForm.newPassword !== passwordForm.confirmPassword) {
    ElMessage.warning('两次输入的新密码不一致')
    return
  }
  if (passwordForm.currentPassword === passwordForm.newPassword) {
    ElMessage.warning('新密码不能和当前密码相同')
    return
  }

  passwordSubmitting.value = true
  try {
    await userStore.changePassword(passwordForm.currentPassword, passwordForm.newPassword)
    ElMessage.success('密码修改成功')
    closePasswordDialog()
  } finally {
    passwordSubmitting.value = false
  }
}

function memoryKeyLabel(key: string) {
  return memoryKeyLabels[key] || key
}

async function openMemoryDialog() {
  memoryDialogVisible.value = true
  await loadMemories()
}

async function loadMemories() {
  memoryLoading.value = true
  try {
    const payload = await apiGetMemories()
    memories.value = payload.memories
  } finally {
    memoryLoading.value = false
  }
}

function startMemoryEdit(row: MemoryFact) {
  editingMemoryId.value = row.id
  memoryForm.factKey = row.fact_key
  memoryForm.factValue = row.fact_value
  memoryForm.confidence = row.confidence || 0.75
  memoryForm.importance = row.importance || 3
}

function cancelMemoryEdit() {
  editingMemoryId.value = null
  memoryForm.factKey = ''
  memoryForm.factValue = ''
  memoryForm.confidence = 0.75
  memoryForm.importance = 3
}

async function saveMemory(id: number) {
  if (!memoryForm.factKey.trim() || !memoryForm.factValue.trim()) {
    ElMessage.warning('记忆类型和内容不能为空')
    return
  }
  memorySavingId.value = id
  try {
    const updated = await apiUpdateMemory(id, {
      fact_key: memoryForm.factKey,
      fact_value: memoryForm.factValue,
      confidence: memoryForm.confidence,
      importance: memoryForm.importance,
    })
    memories.value = [updated, ...memories.value.filter((item) => item.id !== id && item.id !== updated.id)]
    cancelMemoryEdit()
    ElMessage.success('记忆已更新')
  } finally {
    memorySavingId.value = null
  }
}

async function deleteMemory(row: MemoryFact) {
  try {
    await ElMessageBox.confirm(`确定删除这条记忆：${row.fact_value}`, '删除记忆', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }

  memoryDeletingId.value = row.id
  try {
    await apiDeleteMemory(row.id)
    memories.value = memories.value.filter((item) => item.id !== row.id)
    if (editingMemoryId.value === row.id) {
      cancelMemoryEdit()
    }
    ElMessage.success('记忆已删除')
  } finally {
    memoryDeletingId.value = null
  }
}
</script>

<style scoped>
.app-header {
  display: grid;
  grid-template-columns: 180px 1fr 320px;
  align-items: center;
  gap: 14px;
  height: 60px;
  padding: 0 20px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
  border-bottom: 1px solid #e3eaf6;
}

.logo {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0.6px;
  color: #2f55d4;
  cursor: pointer;
}

.current-page {
  display: flex;
  justify-content: center;
  align-items: center;
  min-width: 0;
}

.current-page-title {
  margin: 0;
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
  font-size: 24px;
  font-weight: 700;
  letter-spacing: 1.2px;
  color: #182848;
  text-shadow: 0 1px 0 rgba(255, 255, 255, 0.8);
}

.user-info {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.user-pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 4px 12px 4px 4px;
  border-radius: 999px;
  border: 1px solid #dfe7f5;
  background: #ffffff;
  box-shadow: 0 3px 10px rgba(22, 49, 109, 0.08);
}

.user-avatar {
  border: 1px solid #d9e6ff;
}

.user-meta {
  display: flex;
  flex-direction: column;
  min-width: 0;
  line-height: 1.2;
}

.user-name {
  max-width: 120px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 13px;
  font-weight: 600;
  color: #273659;
}

.user-id {
  font-size: 11px;
  color: #7d8aa8;
}

.menu-btn {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: 1px solid #d6e2fb;
  background: #ffffff;
  color: #3057d5;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all 0.18s ease;
}

.menu-btn:hover {
  background: #f2f7ff;
  border-color: #bdd1fb;
}

.no-user {
  font-size: 13px;
  color: #98a3bb;
}

.memory-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.memory-hint {
  font-size: 13px;
  color: #66708a;
}

.memory-table {
  width: 100%;
}

.memory-value {
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.45;
}

.memory-source {
  font-size: 12px;
  color: #7d8aa8;
}
</style>
