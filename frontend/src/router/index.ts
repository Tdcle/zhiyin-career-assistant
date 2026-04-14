import { createRouter, createWebHistory } from 'vue-router'

import { useUserStore } from '@/stores/user'
import AuthView from '@/views/AuthView.vue'
import InterviewView from '@/views/InterviewView.vue'
import LobbyView from '@/views/LobbyView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/auth',
      name: 'auth',
      component: AuthView,
      meta: { title: '登录 / 注册', requiresAuth: false },
    },
    {
      path: '/',
      name: 'lobby',
      component: LobbyView,
      meta: { title: '求职大厅', requiresAuth: true },
    },
    {
      path: '/interview',
      name: 'interview',
      component: InterviewView,
      meta: { title: '模拟面试', requiresAuth: true },
    },
  ],
})

router.beforeEach(async (to) => {
  const userStore = useUserStore()
  await userStore.initializeAuth()

  const requiresAuth = Boolean(to.meta.requiresAuth)
  if (requiresAuth && !userStore.isAuthenticated) {
    return {
      path: '/auth',
      query: { redirect: to.fullPath },
    }
  }

  if (to.path === '/auth' && userStore.isAuthenticated) {
    const redirect = typeof to.query.redirect === 'string' ? to.query.redirect : '/'
    return redirect || '/'
  }

  return true
})

router.afterEach((to) => {
  document.title = `${to.meta.title ?? '职引'} | 职引`
})

export default router
