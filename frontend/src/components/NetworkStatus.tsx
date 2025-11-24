import React, { useEffect, useMemo, useState } from 'react';
import { Tag, Button, Space } from 'antd';
 

// You can set a custom URL via env var `REACT_APP_CONNECTIVITY_CHECK_URL`
const DEFAULT_CHECK_URL = 'https://www.gstatic.com/generate_204';

const checkUrl = (typeof process !== 'undefined' && process.env && process.env.REACT_APP_CONNECTIVITY_CHECK_URL)
  ? process.env.REACT_APP_CONNECTIVITY_CHECK_URL!
  : DEFAULT_CHECK_URL;

async function checkConnectivity(): Promise<boolean> {
  try {
    // Use no-cors GET to avoid CORS errors; success if resolved
    await fetch(checkUrl, { method: 'GET', mode: 'no-cors' as any, cache: 'no-store' });
    return true;
  } catch (e) {
    // Fallback to navigator.onLine if fetch fails
    return typeof navigator !== 'undefined' ? navigator.onLine : false;
  }
}

const NetworkStatus: React.FC = () => {
  const t = (k: string, fb?: string) => fb || k;
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  const runCheck = async () => {
    setStatus('checking');
    const ok = await checkConnectivity();
    setStatus(ok ? 'online' : 'offline');
  };

  useEffect(() => {
    runCheck();
    const onOnline = () => setStatus('online');
    const onOffline = () => setStatus('offline');
    window.addEventListener('online', onOnline);
    window.addEventListener('offline', onOffline);
    return () => {
      window.removeEventListener('online', onOnline);
      window.removeEventListener('offline', onOffline);
    };
  }, []);

  const tag = useMemo(() => {
    if (status === 'checking') return <Tag color="gold">{'Checking network...'}</Tag>;
    if (status === 'online') return <Tag color="green">{'Online'}</Tag>;
    return <Tag color="red">{'Offline'}</Tag>;
  }, [status, t]);

  return (
    <Space>
      {tag}
      <Button size="small" onClick={runCheck}>{'Retry'}</Button>
    </Space>
  );
};

export default NetworkStatus;