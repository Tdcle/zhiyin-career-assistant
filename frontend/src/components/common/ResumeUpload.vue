<template>
  <div class="resume-upload">
    <div class="section-title">简历管理</div>

    <div v-if="resumeInfo?.has_resume" class="existing-resume">
      <el-alert
        :title="`已上传简历：${resumeInfo.filename}`"
        type="success"
        :closable="false"
        show-icon
      />
      <div class="upload-hint">上传新简历将覆盖当前简历</div>
    </div>

    <el-upload
      :show-file-list="false"
      :before-upload="handleUpload"
      accept=".pdf,.png,.jpg,.jpeg"
      drag
      :disabled="uploading"
    >
      <el-icon><Upload /></el-icon>
      <div class="el-upload__text">
        {{ resumeInfo?.has_resume ? '拖拽或点击上传新简历（覆盖）' : '拖拽或点击上传简历' }}
      </div>
      <template #tip>
        <div class="el-upload__tip">支持 PDF / PNG / JPG / JPEG</div>
      </template>
    </el-upload>

    <el-alert
      v-if="status"
      :title="status"
      :type="statusType"
      :closable="false"
      style="margin-top: 8px"
      show-icon
    />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  apiGetResumeInfo,
  apiGetResumeTaskStatus,
  apiUploadResume,
  createResumeTaskStream,
  type ResumeInfo,
  type ResumeTaskStatusResponse,
} from '@/api/user'

const status = ref('')
const statusType = ref<'success' | 'error' | 'info' | 'warning'>('info')
const resumeInfo = ref<ResumeInfo | null>(null)
const uploading = ref(false)

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

async function loadResumeInfo() {
  try {
    resumeInfo.value = await apiGetResumeInfo()
  } catch {
    resumeInfo.value = { has_resume: false }
  }
}

function resolveTaskStatusMessage(
  task: ResumeTaskStatusResponse,
  isUpdate: boolean,
): { text: string; type: 'success' | 'error' | 'info' | 'warning'; done: boolean } {
  if (task.status === 'completed') {
    return {
      text: isUpdate ? '简历更新成功！' : '简历上传成功！',
      type: 'success',
      done: true,
    }
  }
  if (task.status === 'failed') {
    return {
      text: task.message || '简历处理失败',
      type: 'error',
      done: true,
    }
  }
  return {
    text: task.message || '简历处理中...',
    type: 'info',
    done: false,
  }
}

async function pollTaskStatusFallback(taskId: string, isUpdate: boolean) {
  const maxAttempts = 90
  for (let i = 0; i < maxAttempts; i += 1) {
    try {
      const task = await apiGetResumeTaskStatus(taskId)
      const resolved = resolveTaskStatusMessage(task, isUpdate)
      status.value = resolved.text
      statusType.value = resolved.type
      if (resolved.done) return
    } catch {
      status.value = '查询上传状态失败，正在重试...'
      statusType.value = 'warning'
    }
    await sleep(1000)
  }
  status.value = '简历处理超时，请稍后刷新查看状态'
  statusType.value = 'warning'
}

async function waitTaskBySSE(taskId: string, isUpdate: boolean) {
  await new Promise<void>((resolve) => {
    const source = createResumeTaskStream(taskId)
    let settled = false
    let fallbackStarted = false
    const timeout = window.setTimeout(async () => {
      if (settled) return
      source.close()
      await pollTaskStatusFallback(taskId, isUpdate)
      settled = true
      resolve()
    }, 180000)

    const finish = () => {
      if (settled) return
      settled = true
      window.clearTimeout(timeout)
      source.close()
      resolve()
    }

    source.onmessage = (event) => {
      if (!event.data) return
      let task: ResumeTaskStatusResponse | null = null
      try {
        task = JSON.parse(event.data) as ResumeTaskStatusResponse
      } catch {
        return
      }
      const resolved = resolveTaskStatusMessage(task, isUpdate)
      status.value = resolved.text
      statusType.value = resolved.type
      if (resolved.done) {
        finish()
      }
    }

    source.onerror = async () => {
      if (settled || fallbackStarted) return
      fallbackStarted = true
      window.clearTimeout(timeout)
      source.close()
      await pollTaskStatusFallback(taskId, isUpdate)
      finish()
    }
  })
}

async function handleUpload(file: File) {
  if (uploading.value) {
    status.value = '已有上传任务正在执行，请稍候'
    statusType.value = 'warning'
    return false
  }

  const isUpdate = !!resumeInfo.value?.has_resume
  uploading.value = true
  status.value = '简历正在上传...'
  statusType.value = 'info'

  try {
    const submit = await apiUploadResume(file)
    status.value = submit.message || '简历正在上传...'
    statusType.value = 'info'
    await waitTaskBySSE(submit.task_id, isUpdate)
    await loadResumeInfo()
  } catch {
    status.value = '简历上传失败'
    statusType.value = 'error'
  } finally {
    uploading.value = false
  }

  return false
}

onMounted(() => {
  loadResumeInfo()
})
</script>

<style scoped>
.resume-upload {
  padding-top: 4px;
}

.section-title {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 10px;
  color: #303133;
}

.existing-resume {
  margin-bottom: 12px;
}

.upload-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
  text-align: center;
}
</style>
