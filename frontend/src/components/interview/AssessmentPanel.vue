<template>
  <div class="assessment-panel">
    <div class="panel-title">📊 实时评估（100分制）</div>

    <!-- 评分条 -->
    <div class="score-item" v-for="item in scoreItems" :key="item.key">
      <span class="score-label">{{ item.label }}</span>
      <el-progress
        :percentage="scorecard[item.key] ?? 0"
        :color="progressColor(scorecard[item.key] ?? 0)"
        :stroke-width="10"
      />
    </div>

    <!-- 岗位匹配度 - 只显示JD和简历的契合度，不实时更新 -->
    <div v-if="interviewStore.matchAnalysis" class="analysis-section">
      <div class="analysis-title">岗位匹配度</div>
      <p class="analysis-text">{{ interviewStore.matchAnalysis }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useInterviewStore } from '@/stores/interview'

const interviewStore = useInterviewStore()
const scorecard = computed(() => interviewStore.scorecard)

const scoreItems = [
  { key: 'tech_depth' as const,        label: '技术深度' },
  { key: 'project_depth' as const,     label: '项目深度' },
  { key: 'experience_match' as const,  label: '经验匹配' },
  { key: 'communication' as const,     label: '沟通能力' },
  { key: 'jd_fit' as const,            label: 'JD 契合度' },
]

function progressColor(val: number) {
  if (val >= 80) return '#67c23a'
  if (val >= 60) return '#e6a23c'
  return '#f56c6c'
}
</script>

<style scoped>
.assessment-panel { padding: 4px 0; }
.panel-title { font-size: 14px; font-weight: 600; margin-bottom: 12px; color: #303133; }
.score-item { margin-bottom: 10px; }
.score-label { font-size: 13px; color: #606266; display: block; margin-bottom: 4px; }
.analysis-section { margin-top: 16px; }
.analysis-title { font-size: 13px; font-weight: 600; color: #303133; margin-bottom: 6px; }
.analysis-text { font-size: 13px; color: #606266; line-height: 1.6; white-space: pre-wrap; }
</style>
