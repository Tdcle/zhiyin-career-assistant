<template>
  <div class="interview-view">
    <!-- 加载中 -->
    <div v-if="interviewStore.phase === 'starting'" class="loading-mask">
      <el-icon class="loading-icon"><Loading /></el-icon>
      <p>正在连接面试官，准备中...</p>
    </div>

    <template v-else>
      <el-row class="interview-layout" :gutter="16">
        <!-- 左：雷达图 + 评分面板 -->
        <el-col :span="8" class="layout-col">
          <el-card class="panel-card">
            <RadarChart />
            <el-divider />
            <AssessmentPanel />
          </el-card>
        </el-col>

        <!-- 右：面试对话 -->
        <el-col :span="16" class="layout-col">
          <el-card class="chat-card">
            <template #header>
              <div class="interview-header">
                <span>{{ jobTitle }}</span>
                <div class="interview-actions">
                  <el-button type="primary" :loading="ending" @click="handleEnd">
                    结束面试并生成报告
                  </el-button>
                  <el-button @click="handleBack">返回求职大厅</el-button>
                </div>
              </div>
            </template>
            <InterviewChat />
          </el-card>
        </el-col>
      </el-row>
    </template>

    <!-- 报告弹窗 -->
    <el-dialog
      v-model="reportDialogVisible"
      class="report-dialog"
      title="📝 面试评估报告"
      width="920px"
      :close-on-click-modal="false"
      destroy-on-close
    >
      <!-- Loading 状态 -->
      <div v-if="ending && !interviewStore.finalReport" class="report-loading">
        <el-icon class="loading-icon"><Loading /></el-icon>
        <p>正在生成面试评估报告，请稍候...</p>
      </div>
      <!-- 报告内容 -->
      <div v-else class="report-content">{{ interviewStore.finalReport }}</div>
      <template #footer>
        <el-button @click="reportDialogVisible = false">关闭</el-button>
        <el-button type="primary" @click="handleBack">返回求职大厅</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

import RadarChart from '@/components/common/RadarChart.vue'
import AssessmentPanel from '@/components/interview/AssessmentPanel.vue'
import InterviewChat from '@/components/interview/InterviewChat.vue'

import { useInterviewStore } from '@/stores/interview'
import { createEndInterviewStream, apiStartInterview } from '@/api/interview'

const router = useRouter()
const interviewStore = useInterviewStore()
const ending = ref(false)
const reportDialogVisible = ref(false)
const loading = ref(false)

const jobTitle = computed(() =>
  (interviewStore.jobContext?.title as string) ?? interviewStore.selectedJob?.title ?? '模拟面试'
)

// 页面加载时自动启动面试
onMounted(async () => {
  // 如果已经有会话，说明已经初始化过了
  if (interviewStore.threadId) {
    return
  }
  
  // 如果没有选中的职位，返回求职大厅
  if (!interviewStore.selectedJob) {
    ElMessage.warning('请先选择职位')
    router.push('/')
    return
  }
  
  // 显示加载状态
  loading.value = true
  interviewStore.phase = 'starting'
  
  try {
    const result = await apiStartInterview(interviewStore.selectedJob.job_id)
    interviewStore.initSession(result as Parameters<typeof interviewStore.initSession>[0])
  } catch {
    ElMessage.error('启动面试失败，请重试')
    interviewStore.reset()
    router.push('/')
  } finally {
    loading.value = false
  }
})

async function handleEnd() {
  ending.value = true
  interviewStore.phase = 'ending'
  interviewStore.finalReport = ''
  reportDialogVisible.value = true

  const es = createEndInterviewStream()
  es.onmessage = (e) => {
    const event = JSON.parse(e.data)
    if (event.type === 'report') {
      interviewStore.finalReport += event.content
    } else if (event.type === 'done') {
      es.close()
      ending.value = false
      interviewStore.phase = 'done'
    } else if (event.type === 'error') {
      ElMessage.error(event.message)
      es.close()
      ending.value = false
    }
  }
  es.onerror = () => {
    ElMessage.error('生成报告失败，请重试')
    es.close()
    ending.value = false
  }
}

function handleBack() {
  interviewStore.reset()
  router.push('/')
}
</script>

<style scoped>
.interview-view {
  height: calc(100vh - 60px);
  min-height: calc(100vh - 60px);
  padding: 16px 20px;
  overflow: hidden;
}
.interview-layout {
  height: 100%;
}
.layout-col {
  height: 100%;
  min-height: 0;
  display: flex;
}
.loading-mask {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 60vh;
  gap: 16px;
  color: #909399;
}
.loading-icon {
  font-size: 48px;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
.panel-card {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}
.panel-card :deep(.el-card__body) {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}
.chat-card { 
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}
.chat-card :deep(.el-card__body) {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  padding: 0;
}
.interview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 16px;
  font-weight: 600;
}
.interview-actions { display: flex; gap: 8px; }

:deep(.report-dialog .el-dialog) {
  width: 920px;
  max-width: calc(100vw - 40px);
  height: 78vh;
  max-height: 78vh;
  display: flex;
  flex-direction: column;
}
:deep(.report-dialog .el-dialog__body) {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  display: flex;
  padding: 12px 16px 16px;
}
:deep(.report-dialog .el-dialog__footer) {
  border-top: 1px solid #ebeef5;
  padding-top: 12px;
}
.report-content {
  white-space: pre-wrap;
  font-size: 14px;
  line-height: 1.8;
  color: #303133;
  height: 100%;
  width: 100%;
  overflow-y: auto;
  padding: 16px;
  background-color: #f5f7fa;
  border-radius: 8px;
}

.report-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  gap: 16px;
  color: #909399;
}

.report-loading .loading-icon {
  font-size: 48px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>
