import React, { useEffect, useMemo, useState } from 'react';

type Breakpoint = {
  name: string;
  min: number;
  max?: number;
};

const DEFAULT_BREAKPOINTS: Breakpoint[] = [
  { name: 'xs', min: 0, max: 479 },
  { name: 'sm', min: 480, max: 767 },
  { name: 'md', min: 768, max: 1023 },
  { name: 'lg', min: 1024, max: 1279 },
  { name: 'xl', min: 1280 }
];

function getBreakpoint(width: number, breakpoints: Breakpoint[]): string {
  for (const bp of breakpoints) {
    if (width >= bp.min && (bp.max === undefined || width <= bp.max)) {
      return bp.name;
    }
  }
  return 'unknown';
}

function checkStyleSheetAccess(): { total: number; readable: number; errors: number } {
  const sheets = Array.from(document.styleSheets || []);
  let readable = 0;
  let errors = 0;
  for (const sheet of sheets) {
    try {
      // Accessing cssRules may throw for cross-origin styles
      // eslint-disable-next-line @typescript-eslint/no-unused-expressions
      sheet.cssRules?.length;
      readable += 1;
    } catch (e) {
      errors += 1;
    }
  }
  return { total: sheets.length, readable, errors };
}

export interface UIDiagnosticsOverlayProps {
  enabled?: boolean;
  breakpoints?: Breakpoint[];
}

export function UIDiagnosticsOverlay({ enabled = false, breakpoints = DEFAULT_BREAKPOINTS }: UIDiagnosticsOverlayProps) {
  const [flag, setFlag] = useState<boolean>(() => {
    return enabled || localStorage.getItem('uiDiagnosticsEnabled') === 'true';
  });

  const [viewport, setViewport] = useState({ width: window.innerWidth, height: window.innerHeight });
  const [styleInfo, setStyleInfo] = useState(checkStyleSheetAccess());

  useEffect(() => {
    function onResize() {
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    }
    window.addEventListener('resize', onResize);
    const id = window.setInterval(() => setStyleInfo(checkStyleSheetAccess()), 3000);
    return () => {
      window.removeEventListener('resize', onResize);
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const toggleCombo = e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'd';
      if (toggleCombo) {
        const next = !flag;
        setFlag(next);
        localStorage.setItem('uiDiagnosticsEnabled', String(next));
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [flag]);

  const bp = useMemo(() => getBreakpoint(viewport.width, breakpoints), [viewport.width, breakpoints]);

  if (!flag) return null;

  const dpr = Math.round(window.devicePixelRatio * 100) / 100;
  const orientation = window.matchMedia('(orientation: portrait)').matches ? 'portrait' : 'landscape';

  return (
    <div style={{
      position: 'fixed',
      right: 12,
      bottom: 12,
      zIndex: 99999,
      background: 'rgba(20,20,20,0.85)',
      color: '#fff',
      padding: '10px 12px',
      borderRadius: 8,
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
      boxShadow: '0 2px 8px rgba(0,0,0,0.25)'
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>UI Diagnostics Panel</div>
      <div>Viewport: {viewport.width} × {viewport.height} ({orientation})</div>
      <div>DPR: {dpr}</div>
      <div>Breakpoint: {bp}</div>
      <div>Stylesheets: {styleInfo.readable}/{styleInfo.total} readable, errors {styleInfo.errors}</div>
      <div style={{ marginTop: 6, opacity: 0.9 }}>Shortcut: Ctrl + Shift + D to toggle</div>
    </div>
  );
}

export default UIDiagnosticsOverlay;