<template>
  <div class="chat-box">
    <div class="input-wrapper">
      <el-input
        v-model="inputText"
        placeholder="例如：帮我找北京的大模型岗位"
        :disabled="chatStore.streaming"
        @keydown.enter.prevent="handleSend"
      >
        <template #append>
          <button
            class="send-btn"
            :disabled="chatStore.streaming || !inputText.trim()"
            @click="handleSend"
          >
            <el-icon v-if="chatStore.streaming" class="is-loading"><Loading /></el-icon>
            <span>{{ chatStore.streaming ? '发送中' : '发送' }}</span>
          </button>
        </template>
      </el-input>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'

import { createChatStream, type Job } from '@/api/chat'
import { useChatStore } from '@/stores/chat'
import { useUserStore } from '@/stores/user'

const chatStore = useChatStore()
const userStore = useUserStore()
const inputText = ref('')

async function handleSend() {
  const text = inputText.value.trim()
  if (!text || chatStore.streaming) return
  if (!userStore.isAuthenticated) {
    ElMessage.warning('请先登录')
    return
  }

  const sessionId = await chatStore.ensureActiveSession()

  inputText.value = ''
  chatStore.streaming = true
  chatStore.addMessage({ role: 'user', content: text })
  chatStore.addMessage({ role: 'assistant', content: '' })

  const es = createChatStream(text, sessionId)

  es.onmessage = (e) => {
    const event = JSON.parse(e.data) as { type: string; content?: string; jobs?: unknown[]; message?: string; hint?: string }
    if (event.type === 'token' && event.content) {
      chatStore.appendToLast(event.content)
    } else if (event.type === 'tool_call' && event.hint) {
      chatStore.appendToLast(`\n${event.hint}\n`)
    } else if (event.type === 'jobs') {
      chatStore.setJobs((event.jobs ?? []) as Job[])
    } else if (event.type === 'error') {
      ElMessage.error(event.message || '请求失败')
      es.close()
      chatStore.streaming = false
    } else if (event.type === 'done') {
      es.close()
      chatStore.streaming = false
      void chatStore.refreshSessions(chatStore.currentSessionId ?? undefined)
    }
  }

  es.onerror = () => {
    ElMessage.error('连接错误，请重试')
    es.close()
    chatStore.streaming = false
  }
}
</script>

<style scoped>
.chat-box {
  margin-top: 12px;
}

.input-wrapper {
  display: flex;
  align-items: center;
  gap: 12px;
}

.input-wrapper :deep(.el-input) {
  flex: 1;
}

.input-wrapper :deep(.el-input__wrapper) {
  border-radius: 4px 0 0 4px;
}

.input-wrapper :deep(.el-input-group__append) {
  padding: 0;
  border: none;
  background: transparent;
}

.send-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 0 4px 4px 0;
  cursor: pointer;
  font-size: 14px;
  background-color: #67c23a;
  color: white;
  transition: all 0.3s;
  white-space: nowrap;
  height: 100%;
}

.send-btn:hover:not(:disabled) {
  background-color: #85ce61;
}

.send-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
