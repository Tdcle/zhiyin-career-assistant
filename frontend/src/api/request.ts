import axios from 'axios'
import { ElMessage } from 'element-plus'

const ACCESS_TOKEN_KEY = 'jobagent_access_token'

export function getAccessToken(): string | null {
  return localStorage.getItem(ACCESS_TOKEN_KEY)
}

export function setAccessToken(token: string) {
  localStorage.setItem(ACCESS_TOKEN_KEY, token)
}

export function clearAccessToken() {
  localStorage.removeItem(ACCESS_TOKEN_KEY)
}

const request = axios.create({
  baseURL: '/api',
  timeout: 30000,
  withCredentials: true,
})

request.interceptors.request.use((config) => {
  const token = getAccessToken()
  if (token) {
    config.headers = config.headers ?? {}
    ;(config.headers as Record<string, string>).Authorization = `Bearer ${token}`
  }
  return config
})

request.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const silent = Boolean((err.config as { silentError?: boolean } | undefined)?.silentError)
    if (!silent) {
      const msg = err.response?.data?.detail ?? err.message ?? '请求失败'
      ElMessage.error(msg)
    }
    return Promise.reject(err)
  },
)

export default request
