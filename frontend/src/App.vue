<template>
  <el-config-provider :locale="zhCn">
    <div class="app-shell">
      <AppHeader />
      <main class="app-main">
        <router-view />
      </main>
    </div>
  </el-config-provider>
</template>

<script setup lang="ts">
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import AppHeader from '@/components/layout/AppHeader.vue'
</script>

<style>
html,
body,
#app {
  height: 100%;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background: #f5f7fa;
}

.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-main {
  flex: 1;
  min-height: 0;
}
</style>
