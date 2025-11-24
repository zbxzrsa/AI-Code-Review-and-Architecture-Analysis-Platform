export type UXMetric = {
  name: string;
  value: number;
  timestamp: number;
};

export type UXMetricsSnapshot = {
  lcp?: UXMetric;
  cls?: UXMetric;
  fcp?: UXMetric;
  ttfb?: UXMetric;
};

let metrics: UXMetricsSnapshot = {};
let observersStarted = false;

export function startUXObservers() {
  if (observersStarted) return;
  observersStarted = true;

  // Largest Contentful Paint
  try {
    const lcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const last = entries[entries.length - 1] as any;
      metrics.lcp = { name: 'LCP', value: last?.renderTime || last?.loadTime || 0, timestamp: Date.now() };
    });
    lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true } as any);
  } catch {}

  // Cumulative Layout Shift
  try {
    let clsValue = 0;
    const clsObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries() as any) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
          metrics.cls = { name: 'CLS', value: clsValue, timestamp: Date.now() };
        }
      }
    });
    clsObserver.observe({ type: 'layout-shift', buffered: true } as any);
  } catch {}

  // First Contentful Paint & TTFB
  try {
    const nav = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined;
    if (nav) {
      metrics.ttfb = { name: 'TTFB', value: nav.responseStart, timestamp: Date.now() };
      const paint = performance.getEntriesByType('paint') as PerformanceEntry[];
      const fcp = paint.find((p) => p.name === 'first-contentful-paint');
      if (fcp) metrics.fcp = { name: 'FCP', value: fcp.startTime, timestamp: Date.now() };
    }
  } catch {}
}

export function getUXMetrics(): UXMetricsSnapshot {
  return { ...metrics };
}

export function resetUXMetrics() {
  metrics = {};
}

export async function sendUXMetrics(endpoint: string) {
  try {
    const payload = getUXMetrics();
    await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[UXMetrics] send failed', e);
  }
}