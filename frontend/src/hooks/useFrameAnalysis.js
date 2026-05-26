import { useEffect, useRef, useState } from 'react';
import { analyzeFrame } from '../services/api';

const INTERVAL = parseInt(import.meta.env.VITE_FRAME_INTERVAL_MS, 10) || 5000; //Changing it to every 5 seconds. I think 2 was too fast

/**
 * Polls the pipeline endpoint at a fixed interval, returning the latest
 * analysis frame and a connection flag for the topbar status indicator.
 *
 * @param {() => Promise<Blob|null>} captureBlob
 * @param {boolean} enabled
 */
export function useFrameAnalysis(captureBlob, enabled) {
  const [analysis, setAnalysis] = useState(null);
  const [connected, setConnected] = useState(null);
  const inFlight = useRef(false);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!enabled) {
      if (timerRef.current) clearTimeout(timerRef.current);
      setAnalysis(null);
      return;
    }

    let cancelled = false;

    const run = async () => {
      if (cancelled || inFlight.current) {
        timerRef.current = setTimeout(run, INTERVAL);
        return;
      }
      inFlight.current = true;
      try {
        const blob = await captureBlob();
        if (!blob) {
          inFlight.current = false;
          timerRef.current = setTimeout(run, INTERVAL);
          return;
        }
        const data = await analyzeFrame(blob);
        setConnected(true);
        setAnalysis(data);
      } catch {
        setConnected(false);
        // Keep showing the previous frame's data, do NOT clear
      } finally {
        inFlight.current = false;
        if (!cancelled) timerRef.current = setTimeout(run, INTERVAL);
      }
    };

    run();
    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [enabled, captureBlob]);

  return { analysis, connected };
}
