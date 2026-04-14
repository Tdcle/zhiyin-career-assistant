<template>
  <div ref="chartRef" class="radar-chart" />
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'
import { useInterviewStore } from '@/stores/interview'

const interviewStore = useInterviewStore()
const chartRef = ref<HTMLElement | null>(null)
let chart: echarts.ECharts | null = null

const INDICATORS = [
  { name: '技术深度', max: 100 },
  { name: '项目深度', max: 100 },
  { name: '经验匹配', max: 100 },
  { name: '沟通能力', max: 100 },
  { name: 'JD 契合度', max: 100 },
]

function buildOption(scorecard: Record<string, number>) {
  return {
    radar: { indicator: INDICATORS },
    series: [{
      type: 'radar',
      data: [{
        value: [
          scorecard.tech_depth ?? 0,
          scorecard.project_depth ?? 0,
          scorecard.experience_match ?? 0,
          scorecard.communication ?? 0,
          scorecard.jd_fit ?? 0,
        ],
        name: '面试评分',
        areaStyle: { color: 'rgba(64,158,255,0.2)' },
        lineStyle: { color: '#409eff' },
        itemStyle: { color: '#409eff' },
      }],
    }],
  }
}

onMounted(() => {
  if (chartRef.value) {
    chart = echarts.init(chartRef.value)
    chart.setOption(buildOption(interviewStore.scorecard))
  }
})

watch(
  () => interviewStore.scorecard,
  (sc) => chart?.setOption(buildOption(sc)),
  { deep: true },
)

onUnmounted(() => chart?.dispose())
</script>

<style scoped>
.radar-chart {
  width: 100%;
  height: 280px;
}
</style>
