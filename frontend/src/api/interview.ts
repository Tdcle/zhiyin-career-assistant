import request from './request'

export interface Scorecard {
  tech_depth: number
  project_depth: number
  experience_match: number
  communication: number
  jd_fit: number
}

export interface StartInterviewResult {
  thread_id: string
  job_context: Record<string, unknown>
  messages: Array<{ role: string; content: string }>
  radar_image_url: string | null
  match_analysis: string
  live_assessment_md: string
  scorecard: Scorecard
}

export const apiStartInterview = (job_id: string): Promise<StartInterviewResult> =>
  request.post('/interview/start', { job_id })

export const apiGetInterviewState = (): Promise<Record<string, unknown>> =>
  request.get('/interview/state')

export const apiResetInterview = (): Promise<{ success: boolean }> =>
  request.post('/interview/reset')

/**
 * 面试 SSE 流式对话
 */
export function createInterviewStream(message: string): EventSource {
  const url = `/api/interview/stream?message=${encodeURIComponent(message)}`
  return new EventSource(url, { withCredentials: true })
}

/**
 * 结束面试并流式返回报告
 */
export function createEndInterviewStream(): EventSource {
  return new EventSource('/api/interview/end', { withCredentials: true })
}
