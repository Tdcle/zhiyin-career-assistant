<template>
  <div class="lobby-view">
    <el-row :gutter="16" class="lobby-row">
      <el-col :span="7" class="lobby-col">
        <el-card class="sidebar-card" shadow="never">
          <div class="sidebar-section">
            <SessionPanelPlaceholder />
          </div>
          <el-divider />
          <div class="sidebar-section">
            <ResumeUpload />
          </div>
          <el-divider />
          <div class="sidebar-section sidebar-jobs">
            <JobCardList compact :max-items="6" @start-interview="handleStartInterview" />
          </div>
        </el-card>
      </el-col>

      <el-col :span="17" class="lobby-col">
        <el-card class="main-card" shadow="never">
          <MessageList />
          <ChatBox />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'

import type { Job } from '@/api/chat'
import ChatBox from '@/components/chat/ChatBox.vue'
import JobCardList from '@/components/chat/JobCardList.vue'
import MessageList from '@/components/chat/MessageList.vue'
import ResumeUpload from '@/components/common/ResumeUpload.vue'
import SessionPanelPlaceholder from '@/components/common/SessionPanelPlaceholder.vue'
import { useChatStore } from '@/stores/chat'
import { useInterviewStore } from '@/stores/interview'

const router = useRouter()
const chatStore = useChatStore()
const interviewStore = useInterviewStore()

onMounted(async () => {
  try {
    await chatStore.initializeSessions()
  } catch {
    chatStore.reset()
  }
})

function handleStartInterview(job: Job) {
  interviewStore.reset()
  interviewStore.selectedJob = job
  router.push('/interview')
}
</script>

<style scoped>
.lobby-view {
  padding: 16px;
  height: calc(100vh - 60px);
  box-sizing: border-box;
}

.lobby-row {
  height: 100%;
}

.lobby-col {
  height: 100%;
  display: flex;
}

.sidebar-card {
  height: 100%;
  width: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-card :deep(.el-card__body) {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 14px;
}

.sidebar-section {
  flex: 0 0 auto;
}

.sidebar-jobs {
  flex: 1 1 auto;
  min-height: 0;
}

.main-card {
  height: 100%;
  width: 100%;
  display: flex;
  flex-direction: column;
}

.main-card :deep(.el-card__body) {
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 14px;
}
</style>
