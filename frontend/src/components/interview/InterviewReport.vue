<template>
  <el-card class="report-card" style="margin-top: 16px">
    <template #header>
      <span style="font-weight: 600">📝 面试评估报告</span>
    </template>
    <div class="report-content">{{ interviewStore.finalReport }}</div>
  </el-card>
</template>

<script setup lang="ts">
import { useInterviewStore } from '@/stores/interview'
const interviewStore = useInterviewStore()
</script>

<style scoped>
.report-card {}
.report-content {
  white-space: pre-wrap;
  font-size: 14px;
  line-height: 1.8;
  color: #303133;
}
</style>
