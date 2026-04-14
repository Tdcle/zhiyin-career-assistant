<template>
  <div v-if="visibleJobs.length" class="job-card-list" :class="{ compact: isCompact }">
    <div class="section-title">推荐职位列表</div>
    <el-row :gutter="isCompact ? 8 : 12" class="job-grid">
      <el-col
        v-for="job in visibleJobs"
        :key="job.job_id"
        :span="isCompact ? 24 : 8"
        class="job-col"
      >
        <el-card
          class="job-card"
          shadow="hover"
          @click="emit('startInterview', job)"
        >
          <div class="job-company">{{ job.company }}</div>
          <div class="job-title">{{ job.title }}</div>
          <div class="job-meta">
            <el-tag size="small" type="success">{{ job.salary }}</el-tag>
            <span class="job-tags">{{ formatTags(job.tags) }}</span>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import type { Job } from '@/api/chat'
import { useChatStore } from '@/stores/chat'

const props = withDefaults(defineProps<{ compact?: boolean; maxItems?: number }>(), {
  compact: false,
  maxItems: 6,
})

const chatStore = useChatStore()
const emit = defineEmits<{ (e: 'startInterview', job: Job): void }>()

const isCompact = computed(() => props.compact)
const visibleJobs = computed(() => chatStore.jobs.slice(0, props.maxItems))

function formatTags(tags: string) {
  return (tags || '').replace(/\|/g, ' / ')
}
</script>

<style scoped>
.job-card-list {
  margin-top: 4px;
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.section-title {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 10px;
  color: #303133;
}

.job-grid {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding-right: 2px;
}

.job-col {
  margin-bottom: 10px;
}

.job-card {
  cursor: pointer;
  transition: transform 0.15s;
}

.job-card:hover {
  transform: translateY(-2px);
}

.job-company {
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
}

.job-title {
  font-size: 15px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 8px;
}

.job-meta {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}

.job-tags {
  font-size: 12px;
  color: #606266;
}

.compact .job-title {
  font-size: 14px;
}

.compact .job-card :deep(.el-card__body) {
  padding: 10px 12px;
}
</style>
