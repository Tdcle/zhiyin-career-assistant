import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Scorecard } from '@/api/interview'

export interface InterviewMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface Job {
  job_id: string
  company: string
  title: string
  salary: string
  tags: string
}

export const useInterviewStore = defineStore('interview', () => {
  const threadId = ref<string | null>(null)
  const jobContext = ref<Record<string, unknown>>({})
  const selectedJob = ref<Job | null>(null)
  const messages = ref<InterviewMessage[]>([])
  const scorecard = ref<Scorecard>({
    tech_depth: 0,
    project_depth: 0,
    experience_match: 0,
    communication: 0,
    jd_fit: 0,
  })
  const liveAssessmentMd = ref('')
  const radarImageUrl = ref<string | null>(null)
  const matchAnalysis = ref('')
  const finalReport = ref('')
  const streaming = ref(false)
  const phase = ref<'idle' | 'starting' | 'interviewing' | 'ending' | 'done'>('idle')

  function initSession(data: {
    thread_id: string
    job_context: Record<string, unknown>
    messages: InterviewMessage[]
    radar_image_url: string | null
    match_analysis: string
    live_assessment_md: string
    scorecard: Scorecard
  }) {
    threadId.value = data.thread_id
    jobContext.value = data.job_context
    messages.value = data.messages
    radarImageUrl.value = data.radar_image_url
    matchAnalysis.value = data.match_analysis
    liveAssessmentMd.value = data.live_assessment_md
    scorecard.value = data.scorecard
    phase.value = 'interviewing'
  }

  function addMessage(msg: InterviewMessage) {
    messages.value.push(msg)
  }

  function appendToLast(content: string) {
    const last = messages.value[messages.value.length - 1]
    if (last && last.role === 'assistant') {
      last.content += content
    }
  }

  function updateScore(data: { scorecard: Scorecard; live_assessment_md: string }) {
    scorecard.value = data.scorecard
    liveAssessmentMd.value = data.live_assessment_md
  }

  function reset() {
    threadId.value = null
    jobContext.value = {}
    selectedJob.value = null
    messages.value = []
    scorecard.value = { tech_depth: 0, project_depth: 0, experience_match: 0, communication: 0, jd_fit: 0 }
    liveAssessmentMd.value = ''
    radarImageUrl.value = null
    matchAnalysis.value = ''
    finalReport.value = ''
    streaming.value = false
    phase.value = 'idle'
  }

  return {
    threadId, jobContext, selectedJob, messages, scorecard, liveAssessmentMd,
    radarImageUrl, matchAnalysis, finalReport, streaming, phase,
    initSession, addMessage, appendToLast, updateScore, reset,
  }
})
