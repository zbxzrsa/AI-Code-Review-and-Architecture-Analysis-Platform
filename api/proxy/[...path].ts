export const config = {
  runtime: 'edge',
};

const backend = process.env.BACKEND_URL!; // set per Vercel project/alias

export default async (req: Request, { params }: any) => {
  const url = new URL(req.url);
  const segs = Array.isArray(params.path) ? params.path.join('/') : params.path || '';
  const target = `${backend.replace(/\/$/, '')}/${segs}${url.search}`;

  const init: RequestInit = {
    method: req.method,
    headers: req.headers,
    body: ['GET', 'HEAD'].includes(req.method) ? undefined : await req.text(),
  };

  // Add timeout with AbortController
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 20000); // 20s timeout

  try {
    const res = await fetch(target, {
      ...init,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    // Handle 5xx with retry logic
    if (res.status >= 500) {
      // Simple retry with jittered backoff
      await new Promise(resolve => setTimeout(resolve, Math.random() * 200 + 200)); // 200-400ms

      try {
        const retryRes = await fetch(target, {
          ...init,
          signal: AbortSignal.timeout(20000), // Alternative timeout approach
        });

        return new Response(retryRes.body, {
          status: retryRes.status,
          headers: retryRes.headers,
        });
      } catch (retryError) {
        // Return original response if retry fails
        return new Response(res.body, {
          status: res.status,
          headers: res.headers,
        });
      }
    }

    return new Response(res.body, {
      status: res.status,
      headers: res.headers,
    });
  } catch (error) {
    clearTimeout(timeoutId);

    if (error.name === 'AbortError') {
      return new Response('Gateway Timeout', {
        status: 504,
        headers: { 'Content-Type': 'text/plain' },
      });
    }

    // Network/connection errors
    return new Response('Bad Gateway', {
      status: 502,
      headers: { 'Content-Type': 'text/plain' },
    });
  }
};

const backend = process.env.BACKEND_URL!; // set per Vercel project/alias

export default async (req: Request, { params }: any) => {
  const url = new URL(req.url);
  const segs = Array.isArray(params.path) ? params.path.join('/') : params.path || '';
  const target = `${backend.replace(/\/$/, '')}/${segs}${url.search}`;

  const init: RequestInit = {
    method: req.method,
    headers: req.headers,
    body: ['GET', 'HEAD'].includes(req.method) ? undefined : await req.text(),
  };

  const res = await fetch(target, init);
  return new Response(res.body, { status: res.status, headers: res.headers });
};
