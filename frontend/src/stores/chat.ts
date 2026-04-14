import { defineStore } from 'pinia'
import { ref } from 'vue'

import {
  apiCreateChatSession,
  apiDeleteChatSession,
  apiGetChatHistory,
  apiGetChatSessions,
  type ChatSessionItem,
  type Job,
} from '@/api/chat'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export type ChatSession = ChatSessionItem

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const jobs = ref<Job[]>([])
  const sessions = ref<ChatSession[]>([])
  const currentSessionId = ref<string | null>(null)
  const streaming = ref(false)
  const sessionsInitialized = ref(false)

  function setMessages(nextMessages: ChatMessage[]) {
    messages.value = [...nextMessages]
  }

  function addMessage(msg: ChatMessage) {
    messages.value.push(msg)
  }

  function appendToLast(content: string) {
    const last = messages.value[messages.value.length - 1]
    if (last && last.role === 'assistant') {
      last.content += content
    }
  }

  function setJobs(newJobs: Job[]) {
    jobs.value = [...newJobs]
  }

  async function refreshSessions(preferredSessionId?: string): Promise<string | null> {
    const payload = await apiGetChatSessions()
    sessions.value = payload.sessions

    const preferredExists =
      Boolean(preferredSessionId) &&
      payload.sessions.some((session) => session.session_id === preferredSessionId)

    const resolvedSessionId = preferredExists
      ? (preferredSessionId as string)
      : payload.current_session_id ?? payload.sessions[0]?.session_id ?? null

    currentSessionId.value = resolvedSessionId
    return resolvedSessionId
  }

  async function ensureActiveSession(): Promise<string> {
    if (currentSessionId.value) {
      return currentSessionId.value
    }

    const fromServer = await refreshSessions()
    if (fromServer) {
      return fromServer
    }

    const created = await apiCreateChatSession()
    currentSessionId.value = created.session.session_id
    await refreshSessions(created.session.session_id)
    return created.session.session_id
  }

  async function loadHistory(sessionId?: string): Promise<string> {
    const payload = await apiGetChatHistory(sessionId)
    currentSessionId.value = payload.session_id
    setMessages(payload.messages)
    setJobs(payload.jobs)
    await refreshSessions(payload.session_id)
    return payload.session_id
  }

  async function initializeSessions() {
    if (sessionsInitialized.value) {
      return
    }
    const sessionId = await ensureActiveSession()
    await loadHistory(sessionId)
    sessionsInitialized.value = true
  }

  async function selectSession(sessionId: string) {
    if (streaming.value) {
      return
    }
    await loadHistory(sessionId)
  }

  async function createAndSelectSession(title?: string) {
    if (streaming.value) {
      return
    }
    const created = await apiCreateChatSession(title)
    await refreshSessions(created.session.session_id)
    await loadHistory(created.session.session_id)
  }

  async function deleteSession(sessionId: string) {
    if (streaming.value) {
      return
    }
    const isCurrent = currentSessionId.value === sessionId
    await apiDeleteChatSession(sessionId)
    const nextSessionId = await refreshSessions()

    if (!isCurrent) {
      return
    }

    if (nextSessionId) {
      await loadHistory(nextSessionId)
      return
    }

    const created = await apiCreateChatSession()
    await loadHistory(created.session.session_id)
  }

  function reset() {
    messages.value = []
    jobs.value = []
    sessions.value = []
    currentSessionId.value = null
    streaming.value = false
    sessionsInitialized.value = false
  }

  return {
    messages,
    jobs,
    sessions,
    currentSessionId,
    streaming,
    setMessages,
    addMessage,
    appendToLast,
    setJobs,
    refreshSessions,
    ensureActiveSession,
    loadHistory,
    initializeSessions,
    selectSession,
    createAndSelectSession,
    deleteSession,
    reset,
  }
})
