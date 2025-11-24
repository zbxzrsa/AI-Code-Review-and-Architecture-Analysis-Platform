type ErrorEventPayload = {
  type: 'error' | 'unhandledrejection';
  message: string;
  stack?: string;
  timestamp: number;
};

let endpoint: string | null = null;
let enabled = false;
let buffer: ErrorEventPayload[] = [];

export function initErrorMonitor(options: { endpoint?: string; enabled?: boolean } = {}) {
  endpoint = options.endpoint || null;
  enabled = options.enabled ?? true;
  if (!enabled) return;

  window.addEventListener('error', (e) => {
    const payload: ErrorEventPayload = {
      type: 'error',
      message: (e.error && e.error.message) || e.message || 'Unknown error',
      stack: (e.error && e.error.stack) || undefined,
      timestamp: Date.now()
    };
    push(payload);
  });

  window.addEventListener('unhandledrejection', (e) => {
    const reason: any = e.reason;
    const payload: ErrorEventPayload = {
      type: 'unhandledrejection',
      message: typeof reason === 'string' ? reason : reason?.message || 'Unknown rejection',
      stack: reason?.stack,
      timestamp: Date.now()
    };
    push(payload);
  });

  // 定时上报
  window.setInterval(flush, 5000);
}

function push(ev: ErrorEventPayload) {
  buffer.push(ev);
  if (buffer.length >= 10) flush();
}

async function flush() {
  if (!endpoint || buffer.length === 0) return;
  const payload = buffer.slice();
  buffer = [];
  try {
    await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events: payload })
    });
  } catch (e) {
    // swallow errors to avoid loops
    // eslint-disable-next-line no-console
    console.warn('[ErrorMonitor] send failed', e);
  }
}