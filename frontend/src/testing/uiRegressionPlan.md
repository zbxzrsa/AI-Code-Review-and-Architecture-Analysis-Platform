# UI 回归测试体系（计划）

## 工具选择

- 端到端：推荐 Playwright（跨浏览器、快、易配置）。
- 组件级：React Testing Library + Jest（交互为主、弱化实现细节）。
- 视觉回归：Playwright Screenshot + 像素对比（阈值控制）。

## 基本用例

1. 布局与样式
   - 不同断点（xs/sm/md/lg/xl）快照对比
   - 主要页面元素可见性与层级验证
2. 组件渲染
   - 条件渲染：空/加载/错误/成功四态验证
   - 列表渲染性能：大数据量渲染耗时记录
3. 交互
   - 点击、键盘与触摸事件行为
   - 表单验证与提交，包括错误提示与恢复
4. 状态同步
   - 并发请求与竞态处理（去重/取消）
   - 实时更新节流/防抖的正确性

## 示例片段（Playwright）

```ts
import { test, expect } from '@playwright/test';

test('首页在不同断点保持可用布局', async ({ page }) => {
  await page.goto('http://localhost:3000');
  for (const width of [360, 480, 768, 1024, 1280]) {
    await page.setViewportSize({ width, height: 800 });
    await expect(page.locator('header')).toBeVisible();
    await expect(page.locator('main')).toBeVisible();
    // 可添加快照对比
  }
});
```

## 集成建议

- CI 中按变更模块选择性运行用例、降低成本。
- 建立容忍阈值与重试策略，减少偶发失败的干扰。