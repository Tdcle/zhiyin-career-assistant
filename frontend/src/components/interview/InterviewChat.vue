<template>
  <div class="interview-chat">
    <!-- 消息列表 -->
    <div class="message-list" ref="listRef">
      <div
        v-for="(msg, idx) in interviewStore.messages"
        :key="idx"
        :class="['message-item', msg.role]"
      >
        <el-avatar :size="36" :src="msg.role === 'user' ? userAvatar : botAvatar" />
        <div class="bubble">{{ msg.content }}</div>
      </div>
      <div v-if="interviewStore.streaming" class="message-item assistant">
        <el-avatar :size="36" :src="botAvatar" />
        <div class="bubble typing"><span /><span /><span /></div>
      </div>
    </div>

    <!-- 输入框 -->
    <el-input
      v-model="inputText"
      placeholder="回答面试官的问题..."
      :disabled="interviewStore.streaming || interviewStore.phase === 'done'"
      class="chat-input"
      @keydown.enter.prevent="handleSend"
    >
      <template #append>
        <el-button
          type="primary"
          :loading="interviewStore.streaming"
          :disabled="!inputText.trim()"
          @click="handleSend"
        >
          回答
        </el-button>
      </template>
    </el-input>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { useInterviewStore } from '@/stores/interview'
import { createInterviewStream } from '@/api/interview'
import userAvatar from '@/assets/user.png'
import botAvatar from '@/assets/bot.png'

const interviewStore = useInterviewStore()
const inputText = ref('')
const listRef = ref<HTMLElement | null>(null)

watch(
  () => interviewStore.messages.length,
  () => nextTick(() => {
    if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight
  }),
)

async function handleSend() {
  const text = inputText.value.trim()
  if (!text || interviewStore.streaming) return

  inputText.value = ''
  interviewStore.addMessage({ role: 'user', content: text })
  interviewStore.addMessage({ role: 'assistant', content: '' })
  interviewStore.streaming = true

  const es = createInterviewStream(text)

  es.onmessage = (e) => {
    const event = JSON.parse(e.data)
    if (event.type === 'token') {
      interviewStore.appendToLast(event.content)
    } else if (event.type === 'score') {
      interviewStore.updateScore(event)
    } else if (event.type === 'error') {
      ElMessage.error(event.message)
      es.close()
      interviewStore.streaming = false
    } else if (event.type === 'done') {
      es.close()
      interviewStore.streaming = false
    }
  }

  es.onerror = () => {
    ElMessage.error('面试连接断开，请重试')
    es.close()
    interviewStore.streaming = false
  }
}
</script>

<style scoped>
.interview-chat {
  height: 100%;
  min-height: 0;
  padding: 0 12px 12px;
  display: flex;
  flex-direction: column;
}
.message-list {
  flex: 1;
  min-height: 0;
  height: auto;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 8px 4px;
}
.chat-input {
  margin: 8px 8px 0;
  width: calc(100% - 16px);
}
.message-item { display: flex; gap: 10px; align-items: flex-start; }
.message-item.user { flex-direction: row-reverse; }
.bubble {
  max-width: 75%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  background: #f0f2f5;
}
.message-item.user .bubble { background: #409eff; color: #fff; }
.typing { display: flex; gap: 4px; align-items: center; height: 36px; }
.typing span {
  width: 6px; height: 6px; border-radius: 50%;
  background: #909399; animation: bounce 1s infinite;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-6px); }
}
</style>
