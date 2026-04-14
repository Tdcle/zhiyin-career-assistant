import request from './request'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface Job {
  job_id: string
  title: string
  company: string
  salary: string
  tags: string
  [key: string]: unknown
}

export interface ChatSessionItem {
  session_id: string
  title: string
  updated_at: string
}

export interface ChatSessionListResponse {
  sessions: ChatSessionItem[]
  current_session_id: string | null
}

export const apiGetChatSessions = (): Promise<ChatSessionListResponse> =>
  request.get('/chat/sessions')

export const apiCreateChatSession = (title?: string): Promise<{ session: ChatSessionItem }> =>
  request.post('/chat/sessions', { title })

export const apiDeleteChatSession = (sessionId: string): Promise<{ success: boolean }> =>
  request.delete(`/chat/sessions/${encodeURIComponent(sessionId)}`)

export const apiGetChatHistory = (
  sessionId?: string,
): Promise<{ session_id: string; messages: ChatMessage[]; jobs: Job[] }> =>
  request.get('/chat/history', sessionId ? { params: { session_id: sessionId } } : undefined)

export const apiGetJobs = (sessionId?: string): Promise<{ jobs: Job[] }> =>
  request.get('/chat/jobs', sessionId ? { params: { session_id: sessionId } } : undefined)

/**
 * 创建 SSE 连接，发送消息并监听流式事件。
 * 调用方需自行 close() EventSource。
 */
export function createChatStream(message: string, sessionId: string): EventSource {
  const url =
    `/api/chat/stream?message=${encodeURIComponent(message)}&session_id=${encodeURIComponent(sessionId)}`
  return new EventSource(url, { withCredentials: true })
}
