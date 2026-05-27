import { useEffect, useState } from 'react';
import { analyzeFrame } from '../services/api';

const INTERVAL = parseInt(import.meta.env.VITE_FRAME_INTERVAL_MS, 10) || 2000;

/**
 * Polls the pipeline endpoint at a fixed interval, returning the latest
 * analysis frame and a connection flag for the topbar status indicator.
 *
 * @param {() => Promise<Blob|null>} captureBlob
 * @param {boolean} enabled
 *
 * Each effect run owns its own `cancelled` and `timer` — no shared refs
 * across re-mounts, so toggling enabled or re-rendering can never spawn
 * a parallel polling chain.
 */
export function useFrameAnalysis(captureBlob, enabled) {
  const [analysis, setAnalysis] = useState(null);
  const [connected, setConnected] = useState(null);

  useEffect(() => {
    if (!enabled) {
      setAnalysis(null);
      return;
    }

    let cancelled = false;
    let timer = null;

    const tick = async () => {
      if (cancelled) return;

      let blob;
      try {
        blob = await captureBlob();
      } catch {
        blob = null;
      }
      if (cancelled) return;

      // No frame available (camera not ready) — wait and try again
      if (!blob) {
        timer = setTimeout(tick, INTERVAL);
        return;
      }

      try {
        const data = await analyzeFrame(blob);
        if (cancelled) return;
        setConnected(true);
        setAnalysis(data);
      } catch {
        if (cancelled) return;
        setConnected(false);
        // Keep showing previous frame's data, do NOT clear
      }

      // Schedule the next tick ONLY if we're still alive.
      // Single source of truth — no shared in-flight state.
      if (!cancelled) {
        timer = setTimeout(tick, INTERVAL);
      }
    };

    tick();

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    };
  }, [enabled, captureBlob]);

  return { analysis, connected };
}