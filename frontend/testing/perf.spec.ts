// @ts-nocheck
import { test, expect } from '@playwright/test';

// 示例性能基准：加载主页面的关键指标
test('首页加载性能预算', async ({ page }) => {
  const start = Date.now();
  await page.goto('/');

  // 等待首屏主要内容出现（根据项目的真实选择器调整）
  await page.waitForSelector('#root, main, [data-testid="app-ready"]', { timeout: 15000 });
  const tti = Date.now() - start; // 简化版TTI近似

  // 网络请求数与大小（示例：限制到30个请求以内）
  const requests: { url: string; size?: number }[] = [];
  page.on('requestfinished', async (req) => {
    try {
      const response = await req.response();
      const headers = response?.headers() || {};
      const size = Number(headers['content-length'] || 0);
      requests.push({ url: req.url(), size: isNaN(size) ? undefined : size });
    } catch {}
  });

  // 等待一点时间收集请求
  await page.waitForTimeout(2000);

  const totalRequests = requests.length;
  const totalBytes = requests.reduce((sum, r) => sum + (r.size || 0), 0);

  // 预算：TTI < 4000ms, 请求数 < 30, 首次总字节 < 3MB
  expect(tti).toBeLessThan(4000);
  expect(totalRequests).toBeLessThan(30);
  expect(totalBytes).toBeLessThan(3 * 1024 * 1024);
});