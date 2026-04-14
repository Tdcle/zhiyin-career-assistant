import request from './request'

export interface AuthUser {
  user_id: string
  username: string
}

export interface AuthTokenResponse {
  access_token: string
  token_type: string
  user: AuthUser
}

export interface LoginPayload {
  username: string
  password: string
}

export interface RegisterPayload {
  username: string
  password: string
}

export interface ChangePasswordPayload {
  current_password: string
  new_password: string
}

export const apiLogin = (payload: LoginPayload): Promise<AuthTokenResponse> =>
  request.post('/auth/login', payload)

export const apiRegister = (payload: RegisterPayload): Promise<AuthTokenResponse> =>
  request.post('/auth/register', payload)

export const apiAuthMe = (silent = false): Promise<AuthUser> =>
  request.get('/auth/me', silent ? ({ silentError: true } as Record<string, unknown>) : undefined)

export const apiLogout = (): Promise<{ success: boolean }> =>
  request.post('/auth/logout')

export const apiChangePassword = (payload: ChangePasswordPayload): Promise<{ success: boolean; message: string }> =>
  request.post('/auth/change-password', payload)
