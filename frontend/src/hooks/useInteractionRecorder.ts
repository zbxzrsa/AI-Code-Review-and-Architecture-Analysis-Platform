import { useEffect, useRef, useState } from 'react';

type RecordedEvent = {
  t: number; // timestamp offset
  type: 'click' | 'keydown' | 'touchstart';
  payload: Record<string, any>;
};

interface RecorderOptions {
  autosave?: boolean;
  storageKey?: string;
}

export function useInteractionRecorder(options: RecorderOptions = {}) {
  const { autosave = true, storageKey = 'interactionRecording' } = options;
  const [recording, setRecording] = useState(false);
  const startedAtRef = useRef<number>(0);
  const eventsRef = useRef<RecordedEvent[]>([]);

  useEffect(() => {
    if (!recording) return;
    startedAtRef.current = performance.now();

    function onClick(e: MouseEvent) {
      eventsRef.current.push({ t: performance.now() - startedAtRef.current, type: 'click', payload: {
        x: e.clientX, y: e.clientY, targetSelector: getSelector(e.target as Element)
      }});
    }
    function onKey(e: KeyboardEvent) {
      eventsRef.current.push({ t: performance.now() - startedAtRef.current, type: 'keydown', payload: {
        key: e.key, code: e.code, ctrl: e.ctrlKey, shift: e.shiftKey, alt: e.altKey
      }});
    }
    function onTouch(e: TouchEvent) {
      const t = e.touches[0];
      eventsRef.current.push({ t: performance.now() - startedAtRef.current, type: 'touchstart', payload: {
        x: t.clientX, y: t.clientY
      }});
    }

    window.addEventListener('click', onClick, { capture: true });
    window.addEventListener('keydown', onKey, { capture: true });
    window.addEventListener('touchstart', onTouch, { capture: true });

    return () => {
      window.removeEventListener('click', onClick, { capture: true } as any);
      window.removeEventListener('keydown', onKey, { capture: true } as any);
      window.removeEventListener('touchstart', onTouch, { capture: true } as any);
    };
  }, [recording]);

  function start() {
    eventsRef.current = [];
    setRecording(true);
  }

  function stop() {
    setRecording(false);
    const snapshot = eventsRef.current.slice();
    if (autosave) localStorage.setItem(storageKey, JSON.stringify(snapshot));
    return snapshot;
  }

  function playbackFromStorage(speed = 1) {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return;
    const events: RecordedEvent[] = JSON.parse(raw);
    let base = performance.now();
    for (const ev of events) {
      window.setTimeout(() => dispatch(ev), ev.t / speed);
    }
    function dispatch(ev: RecordedEvent) {
      if (ev.type === 'click') {
        const el = querySelectorSafe(ev.payload.targetSelector);
        if (el) el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      } else if (ev.type === 'keydown') {
        document.dispatchEvent(new KeyboardEvent('keydown', {
          key: ev.payload.key, code: ev.payload.code, ctrlKey: ev.payload.ctrl, shiftKey: ev.payload.shift, altKey: ev.payload.alt, bubbles: true
        }));
      } else if (ev.type === 'touchstart') {
        // 触摸事件的回放在桌面浏览器中可能受限，这里仅作为占位模拟
      }
    }
  }

  return { recording, start, stop, playbackFromStorage } as const;
}

function getSelector(el: Element | null): string {
  if (!el) return '';
  const id = el.id ? `#${el.id}` : '';
  const cls = el.className ? `.${String(el.className).replace(/\s+/g, '.')}` : '';
  return `${el.tagName.toLowerCase()}${id}${cls}`;
}

function querySelectorSafe(sel: string): Element | null {
  try { return document.querySelector(sel); } catch { return null; }
}

export default useInteractionRecorder;