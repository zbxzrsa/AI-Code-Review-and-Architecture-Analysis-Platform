# 性能预算与自动化检查

本指南定义前端应用的性能预算与检查策略，并提供在 CI 中集成的建议。

## 预算目标（默认建议）
- 首屏可交互（TTI）小于 `4000ms`
- 初次渲染的网络请求数小于 `30`
- 初次总下载体积小于 `3MB`
- LCP 小于 `2500ms`
- CLS 小于 `0.1`

## 自动化检查
- 端到端：使用 Playwright 执行 `frontend/testing/perf.spec.ts`
- 指标采集：在应用中启用 `uxMetrics.ts` 与 `PerfMonitorPanel.tsx`
- 构建体积：在打包阶段输出资产体积并进行阈值校验（可在 webpack 或构建脚本中添加）

## CI 集成建议
1. 安装依赖：`npm ci`
2. 构建生产包：`npm run build`
3. 启动服务器：`npm run preview`（或项目对应命令）
4. 运行 Playwright：`npx playwright test frontend/testing/perf.spec.ts`
5. 超过预算即失败，阻止合并与发布。

## 性能监控面板
- `PerfMonitorPanel`：实时查看 LCP/FCP/TTFB/CLS 与长任务统计
- 快捷键：`Ctrl + Shift + P` 切换显示

## 常见优化建议
- 代码分割：合理使用动态 `import()` 与路由级分割
- 资源压缩：启用 gzip/br 与图片压缩（WebP/AVIF）
- 缓存策略：静态资源 `immutable` 缓存；接口加 ETag/Cache-Control
- 懒加载：组件与数据按需加载，避免阻塞首屏
- 关键渲染路径：提取关键 CSS、优先加载必要脚本