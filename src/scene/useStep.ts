import { useFrame } from '@react-three/fiber';
import { useRef } from 'react';
import { story } from '../state/store';
import { clamp01 } from './theme';

/** Crossfade envelope for step k at story position s: in over [k−0.1, k+0.02], out over [k+0.98, k+1.1]. */
export function stepVisibility(k: number, s: number): number {
  const fadeIn = clamp01((s - (k - 0.1)) / 0.12);
  const fadeOut = 1 - clamp01((s - (k + 0.98)) / 0.12);
  return Math.min(fadeIn, fadeOut);
}

/**
 * Run `cb(p, vis)` every frame for step k: p = progress inside the step (clamped 0..1),
 * vis = crossfade opacity (0 when the step is far away; callers should early-out).
 */
export function useStep(k: number, cb: (p: number, vis: number, s: number) => void) {
  const ref = useRef(cb);
  ref.current = cb;
  useFrame(() => {
    const s = story.s;
    const vis = stepVisibility(k, s);
    ref.current(clamp01(s - k), vis, s);
  });
}
