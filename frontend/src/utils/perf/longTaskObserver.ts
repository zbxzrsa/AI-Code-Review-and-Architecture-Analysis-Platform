let started = false;
let totalDuration = 0;
let count = 0;

export function startLongTaskObserver() {
  if (started) return;
  started = true;
  try {
    // @ts-ignore PerformanceObserver types may not include 'longtask'
    const obs = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      for (const e of entries as any) {
        // e.duration in ms
        totalDuration += e.duration || 0;
        count += 1;
      }
    });
    // @ts-ignore type may not include 'longtask'
    obs.observe({ type: 'longtask', buffered: true });
  } catch {
    // ignore unsupported browsers
  }
}

export function getLongTaskStats(): { count: number; totalDuration: number } {
  return { count, totalDuration };
}

export function resetLongTaskStats() {
  totalDuration = 0;
  count = 0;
}