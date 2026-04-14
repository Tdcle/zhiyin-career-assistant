<template>
  <div class="session-panel">
    <div class="header-row">
      <div class="section-title">会话列表</div>
      <el-button size="small" type="primary" :disabled="chatStore.streaming" @click="handleCreateSession">
        新建
      </el-button>
    </div>

    <el-scrollbar class="session-scroll" max-height="188px">
      <div v-if="chatStore.sessions.length" class="session-list">
        <button
          v-for="session in chatStore.sessions"
          :key="session.session_id"
          class="session-item"
          :class="{ active: session.session_id === chatStore.currentSessionId }"
          @click="handleSelectSession(session.session_id)"
        >
          <div class="session-main">
            <div class="session-title">{{ session.title }}</div>
            <div class="session-time">{{ formatTime(session.updated_at) }}</div>
          </div>
          <el-icon
            class="delete-icon"
            :class="{ disabled: chatStore.streaming }"
            @click.stop="handleDeleteSession(session.session_id)"
          >
            <Delete />
          </el-icon>
        </button>
      </div>

      <el-empty v-else :image-size="64" description="暂无会话" />
    </el-scrollbar>
  </div>
</template>

<script setup lang="ts">
import { ElMessageBox } from 'element-plus'

import { useChatStore } from '@/stores/chat'

const chatStore = useChatStore()

function formatTime(value: string): string {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return `${date.getMonth() + 1}-${date.getDate()} ${date.getHours().toString().padStart(2, '0')}:${date
    .getMinutes()
    .toString()
    .padStart(2, '0')}`
}

async function handleCreateSession() {
  await chatStore.createAndSelectSession()
}

async function handleSelectSession(sessionId: string) {
  if (sessionId === chatStore.currentSessionId) return
  await chatStore.selectSession(sessionId)
}

async function handleDeleteSession(sessionId: string) {
  if (chatStore.streaming) return
  try {
    await ElMessageBox.confirm('确认删除该会话吗？删除后不可恢复。', '删除会话', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
    await chatStore.deleteSession(sessionId)
  } catch {
    // user cancelled
  }
}
</script>

<style scoped>
.session-panel {
  padding-top: 4px;
}

.header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}

.section-title {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.session-scroll {
  border: 1px solid #ebeef5;
  border-radius: 6px;
  padding: 8px;
  background: #fff;
  overflow: hidden;
}

.session-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.session-item {
  width: 100%;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  background: #fff;
  padding: 8px 10px;
  min-height: 50px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.session-item:hover {
  border-color: #c6e2ff;
  background: #f5f9ff;
}

.session-item.active {
  border-color: #409eff;
  background: #ecf5ff;
}

.session-main {
  min-width: 0;
  text-align: left;
  flex: 1;
}

.session-title {
  font-size: 13px;
  color: #303133;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-time {
  margin-top: 3px;
  font-size: 12px;
  color: #909399;
}

.delete-icon {
  color: #f56c6c;
  flex-shrink: 0;
}

.delete-icon.disabled {
  color: #c0c4cc;
  cursor: not-allowed;
}
</style>
