import React, { useEffect, useMemo, useState } from 'react';
import { startUXObservers, getUXMetrics, UXMetricsSnapshot } from '../../utils/uxMetrics';
import { startLongTaskObserver, getLongTaskStats } from '../../utils/perf/longTaskObserver';

export interface PerfMonitorPanelProps {
  enabled?: boolean;
}

export function PerfMonitorPanel({ enabled = false }: PerfMonitorPanelProps) {
  const [flag, setFlag] = useState<boolean>(() => enabled || localStorage.getItem('perfMonitorEnabled') === 'true');
  const [metrics, setMetrics] = useState<UXMetricsSnapshot>({});
  const [longTasks, setLongTasks] = useState<{ count: number; totalDuration: number }>({ count: 0, totalDuration: 0 });

  useEffect(() => {
    startUXObservers();
    startLongTaskObserver();
    const id = window.setInterval(() => {
      setMetrics(getUXMetrics());
      setLongTasks(getLongTaskStats());
    }, 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const toggleCombo = e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'p';
      if (toggleCombo) {
        const next = !flag;
        setFlag(next);
        localStorage.setItem('perfMonitorEnabled', String(next));
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [flag]);

  if (!flag) return null;

  const entryStyle: React.CSSProperties = { display: 'flex', justifyContent: 'space-between', gap: 12 };
  const format = (n?: number) => (typeof n === 'number' ? `${Math.round(n)} ms` : '—');

  return (
    <div style={{
      position: 'fixed', right: 12, bottom: 120, zIndex: 99998,
      background: 'rgba(12,12,12,0.85)', color: '#fff', padding: '10px 12px', borderRadius: 8,
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
      boxShadow: '0 2px 8px rgba(0,0,0,0.25)', minWidth: 260
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>Performance Monitor Panel</div>
      <div style={entryStyle}><span>LCP</span><span>{format(metrics.lcp?.value)}</span></div>
      <div style={entryStyle}><span>FCP</span><span>{format(metrics.fcp?.value)}</span></div>
      <div style={entryStyle}><span>TTFB</span><span>{format(metrics.ttfb?.value)}</span></div>
      <div style={entryStyle}><span>CLS</span><span>{typeof metrics.cls?.value === 'number' ? metrics.cls.value.toFixed(3) : '—'}</span></div>
      <div style={{ marginTop: 8, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 8 }}>
        <div style={entryStyle}><span>Long Tasks Count</span><span>{longTasks.count}</span></div>
        <div style={entryStyle}><span>Total Long Task Duration</span><span>{Math.round(longTasks.totalDuration)} ms</span></div>
      </div>
      <div style={{ marginTop: 6, opacity: 0.9 }}>Shortcut: Ctrl + Shift + P to toggle</div>
    </div>
  );
}

export default PerfMonitorPanel;