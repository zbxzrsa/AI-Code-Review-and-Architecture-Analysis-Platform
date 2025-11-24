import { useCallback, useEffect, useRef, useState } from 'react';

export type AsyncStatus = 'idle' | 'loading' | 'error' | 'success';

interface UseAsyncStatusOptions<TArgs extends any[], TData> {
  fetcher: (...args: TArgs) => Promise<TData>;
  auto?: { enabled: boolean; args: TArgs };
  timeoutMs?: number;
}

export function useAsyncStatus<TArgs extends any[], TData>(options: UseAsyncStatusOptions<TArgs, TData>) {
  const { fetcher, auto, timeoutMs } = options;
  const [status, setStatus] = useState<AsyncStatus>('idle');
  const [data, setData] = useState<TData | null>(null);
  const [error, setError] = useState<unknown>(null);
  const requestIdRef = useRef(0);
  const timeoutRef = useRef<number | null>(null);

  const run = useCallback(async (...args: TArgs) => {
    const requestId = ++requestIdRef.current;
    setStatus('loading');
    setError(null);
    setData(null);

    if (timeoutRef.current) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (timeoutMs && timeoutMs > 0) {
      timeoutRef.current = window.setTimeout(() => {
        if (requestId === requestIdRef.current) {
          setStatus('error');
          setError(new Error('请求超时'));
        }
      }, timeoutMs);
    }

    try {
      const result = await fetcher(...args);
      if (requestId === requestIdRef.current) {
        setData(result);
        setStatus('success');
      }
      return result;
    } catch (e) {
      if (requestId === requestIdRef.current) {
        setError(e);
        setStatus('error');
      }
      throw e;
    } finally {
      if (timeoutRef.current) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    }
  }, [fetcher, timeoutMs]);

  useEffect(() => {
    if (auto?.enabled) {
      run(...auto.args).catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { status, data, error, run } as const;
}

export default useAsyncStatus;