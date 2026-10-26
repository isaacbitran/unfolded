import { useFrame } from '@react-three/fiber';

/** Exposes a rolling frame-time average on window.__perf for the dev HUD and Playwright. */
export function PerfProbe() {
  useFrame((_, dt) => {
    const w = window as unknown as { __perf?: { ms: number; frames: number } };
    const p = (w.__perf ??= { ms: 16.7, frames: 0 });
    p.ms = p.ms * 0.95 + dt * 1000 * 0.05;
    p.frames++;
  });
  return null;
}
