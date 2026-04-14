import request from './request'

export interface ResumeTaskSubmitResponse {
  success: boolean
  task_id: string
  status: string
  message: string
}

export interface ResumeTaskStatusResponse {
  task_id: string
  status: string
  message: string
  filename?: string
  user_id?: string
  updated_at?: string
}

export const apiUploadResume = (file: File): Promise<ResumeTaskSubmitResponse> => {
  const form = new FormData()
  form.append('file', file)
  return request.post('/user/resume', form)
}

export const apiGetResumeTaskStatus = (taskId: string): Promise<ResumeTaskStatusResponse> =>
  request.get(`/user/resume/task/${encodeURIComponent(taskId)}`)

export function createResumeTaskStream(taskId: string): EventSource {
  const url = `/api/user/resume/task/${encodeURIComponent(taskId)}/stream`
  return new EventSource(url, { withCredentials: true })
}

export interface ResumeInfo {
  has_resume: boolean
  filename?: string
  created_at?: string
}

export const apiGetResumeInfo = (): Promise<ResumeInfo> =>
  request.get('/user/resume')

export interface MemoryFact {
  id: number
  fact_key: string
  fact_value: string
  source: string
  is_active: boolean
  confidence: number
  importance: number
  expires_at?: string | null
  last_used_at?: string | null
  use_count: number
  meta: Record<string, unknown>
  created_at?: string | null
  updated_at?: string | null
}

export interface MemoryFactListResponse {
  memories: MemoryFact[]
}

export interface UpdateMemoryFactPayload {
  fact_key: string
  fact_value: string
  confidence: number
  importance: number
  meta?: Record<string, unknown>
}

export const apiGetMemories = (): Promise<MemoryFactListResponse> =>
  request.get('/user/memories')

export const apiUpdateMemory = (id: number, payload: UpdateMemoryFactPayload): Promise<MemoryFact> =>
  request.put(`/user/memories/${id}`, payload)

export const apiDeleteMemory = (id: number): Promise<{ success: boolean; message: string }> =>
  request.delete(`/user/memories/${id}`)
