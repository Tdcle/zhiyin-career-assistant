<template>
  <div class="message-list" ref="listRef">
    <div
      v-for="(msg, idx) in chatStore.messages"
      :key="idx"
      :class="['message-item', msg.role]"
    >
      <el-avatar :size="36" :src="msg.role === 'user' ? userAvatar : botAvatar" />
      <div class="bubble">{{ msg.content }}</div>
    </div>
    <div v-if="chatStore.streaming" class="message-item assistant">
      <el-avatar :size="36" :src="botAvatar" />
      <div class="bubble typing">
        <span /><span /><span />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { useChatStore } from '@/stores/chat'
import userAvatar from '@/assets/user.png'
import botAvatar from '@/assets/bot.png'

const chatStore = useChatStore()
const listRef = ref<HTMLElement | null>(null)

watch(
  () => chatStore.messages.length,
  () => nextTick(() => {
    if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight
  }),
)
</script>

<style scoped>
.message-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 12px 4px;
}
.message-item {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}
.message-item.user { flex-direction: row-reverse; }
.bubble {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  background: #f0f2f5;
}
.message-item.user .bubble { background: #409eff; color: #fff; }
/* 打字动画 */
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
