import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import type { AuthTokenResponse, AuthUser } from '@/api/auth'
import { apiAuthMe, apiChangePassword, apiLogin, apiLogout, apiRegister } from '@/api/auth'
import { clearAccessToken, setAccessToken } from '@/api/request'

export const useUserStore = defineStore('user', () => {
  const profile = ref<AuthUser | null>(null)
  const initialized = ref(false)

  const isAuthenticated = computed(() => Boolean(profile.value))
  const currentUser = computed(() => profile.value?.username ?? null)

  function setSession(payload: AuthTokenResponse) {
    setAccessToken(payload.access_token)
    profile.value = payload.user
  }

  async function login(username: string, password: string) {
    const payload = await apiLogin({ username, password })
    setSession(payload)
    return payload.user
  }

  async function register(username: string, password: string) {
    const payload = await apiRegister({ username, password })
    setSession(payload)
    return payload.user
  }

  async function fetchMe(silent = false): Promise<AuthUser | null> {
    try {
      const me = await apiAuthMe(silent)
      profile.value = me
      return me
    } catch {
      profile.value = null
      clearAccessToken()
      if (!silent) {
        throw new Error('fetch user failed')
      }
      return null
    }
  }

  async function initializeAuth() {
    if (initialized.value) {
      return
    }
    await fetchMe(true)
    initialized.value = true
  }

  async function logout() {
    try {
      await apiLogout()
    } finally {
      clearAccessToken()
      profile.value = null
    }
  }

  async function changePassword(currentPassword: string, newPassword: string) {
    await apiChangePassword({
      current_password: currentPassword,
      new_password: newPassword,
    })
  }

  return {
    profile,
    initialized,
    isAuthenticated,
    currentUser,
    login,
    register,
    fetchMe,
    initializeAuth,
    logout,
    changePassword,
  }
})
